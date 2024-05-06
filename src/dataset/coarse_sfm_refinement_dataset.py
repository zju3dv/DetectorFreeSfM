import os
import os.path as osp
from pathlib import Path
from loguru import logger
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from .utils import (
    read_rgb,
)

from ..colmap.read_write_model import (
    read_images_binary,
    read_cameras_binary,
    read_points3d_binary,
    write_model,
)
from ..post_optimization.utils.geometry_utils import *


class CoarseColmapDataset(Dataset):

    def __init__(
        self,
        args,
        image_lists,
        covis_pairs,
        colmap_results_dir,  # before refine results
        save_dir,
        only_basename_in_colmap=False,
        vis_path=None,
        verbose=True
    ):
        """
        Parameters:
        ---------------
        image_lists: ['path/to/image/0.png', 'path/to/image/1.png]
        covis_pairs: List or path
        colmap_results_dir: The directory contains images.bin(.txt) point3D.bin(.txt)...
        """
        super().__init__()
        self.img_list = image_lists

        self.colmap_results_dir = colmap_results_dir
        self.colmap_refined_save_dir = save_dir
        self.vis_path = vis_path

        self.img_resize = args['img_resize']
        self.df = args['df'] # 8
        self.feature_track_assignment_strategy = args['feature_track_assignment_strategy']
        self.verbose = verbose
        self.state = True
        self.preload = args['img_preload']

        if isinstance(covis_pairs, list):
            self.pair_list = covis_pairs
        else:
            # Load pairs: 
            with open(covis_pairs, 'r') as f:
                self.pair_list = f.read().rstrip('\n').split('\n')

        self.frame_ids = list(range(len(self.img_list)))

        # Load colmap coarse results:
        is_colmap_valid = osp.exists(osp.join(colmap_results_dir))

        if not is_colmap_valid:
            logger.warning(f"COLMAP is not valid, current COLMAP path: {osp.join(colmap_results_dir)}")
            self.state = False
            return

        self.colmap_images = read_images_binary(
            osp.join(colmap_results_dir, "images.bin")
        )
        self.colmap_3ds = read_points3d_binary(
            osp.join(colmap_results_dir, "points3D.bin")
        )

        self.colmap_cameras = read_cameras_binary(
            osp.join(colmap_results_dir, "cameras.bin")
        )

        (
            self.frameId2colmapID_dict,
            self.colmapID2frameID_dict,
        ) = self.get_frameID2colmapID(self.frame_ids, self.img_list, self.colmap_images, only_basename_in_colmap=only_basename_in_colmap)

        # Get intrinsic and extrinsics:
        self.image_intrin_extrins = {} # {img_id: {'extrin': [R, t], 'intrin: 3*3}}
        for img_id, colmap_image in self.colmap_images.items():
            extrinsic = get_pose_from_colmap_image(colmap_image) # w2c, [R, t]
            intrinsic = get_intrinsic_from_colmap_camera(
                self.colmap_cameras[colmap_image.camera_id]
            )
            self.image_intrin_extrins[img_id] = {'intrin': intrinsic, 'extrin': extrinsic}

        logger.info(f"Load colmap results finish!") if verbose else None

        # Verification:
        if (
            len(self.colmap_3ds) == 0
            or len(self.colmap_cameras) == 0
            or len(self.colmap_images) == 0
        ):
            self.state = False
        

        # Get keyframes and feature track(3D points) assignment
        logger.info("Building keyframes begin....")
        if self.feature_track_assignment_strategy == "midium_scale":
            (
                self.keyframe_dict,
                self.point_cloud_assigned_imgID_kptID,
            ) = self.get_keyframes_by_scale(self.colmap_images, self.colmap_3ds, verbose=self.verbose, scale_strategy='middle')
        else:
            raise NotImplementedError

        if self.preload:
            self.img_dict = {}
            for img in image_lists:
                self.img_dict[img] = read_rgb(
                    img,
                    (self.img_resize,) if self.img_resize is not None else None,
                    resize_no_larger_than=True,
                    pad_to=None,
                    df=self.df,
                    ret_scales=True,
                )

    def extract_corresponding_frames(self, colmap_frame_dict):
        """
        Update: {related_frameID: list}
        """
        for colmap_frameID, frame_info in colmap_frame_dict.items():
            related_frameID = []
            if not frame_info["is_keyframe"]:
                continue
            all_kpt_status = frame_info["all_kpt_status"]
            point_cloud_idxs = all_kpt_status[all_kpt_status >= 0]
            for point_cloud_idx in point_cloud_idxs:
                # Get related feature track
                image_ids = self.colmap_3ds[point_cloud_idx].image_ids
                point2D_idxs = self.colmap_3ds[point_cloud_idx].point2D_idxs

                related_frameID.append(image_ids)

            all_related_frameID = np.concatenate(related_frameID)
            unique_frameID, counts = np.unique(all_related_frameID, return_counts=True)

            self_idx = np.squeeze(
                np.argwhere(unique_frameID == colmap_frameID)
            ).tolist()  # int
            unique_frameID = unique_frameID.tolist()
            unique_frameID.pop(self_idx)  # pop self index
            frame_info.update({"related_frameID": unique_frameID})
    
    def build_initial_depth_pose(self, colmap_frame_dict):
        """
        Build initial pose for each registred frame, and build initial depth for each keyframe
        Update:
        initial_depth_pose_kpts: {colmap_frameID: {"initial_pose": [R: np.array 3*3, t: np.array 3],
                                              "intrinsic": np.array 3*3
                                              "keypoints": np.array [N_all,2] or None,
                                              "is_keyframe": bool, 
                                              "initial_depth": np.array N_all or None,
                                              "all_kpt_status": np.array N_all or None}}
        NOTE: None is when frame is not keyframe.
        NOTE: Intrinsic is from colmap
        """
        for colmap_frameID, colmap_image in self.colmap_images.items():
            initial_pose = self.image_intrin_extrins[colmap_frameID]['extrin']
            intrinisic = self.image_intrin_extrins[colmap_frameID]['intrin']
            keypoints = self.colmap_images[colmap_frameID].xys  # N_all*2

            if colmap_frameID in self.keyframe_dict:
                is_keyframe = True
                all_kpt_status = self.keyframe_dict[colmap_frameID]["state"]  # N_all

                occupied_mask = all_kpt_status >= 0
                point_cloud_idxs = all_kpt_status[
                    occupied_mask
                ]  # correspondence 3D index of keypoints, N
                point_cloud = np.concatenate(
                    [
                        self.colmap_3ds[point_cloud_idx].xyz[None]
                        for point_cloud_idx in point_cloud_idxs
                    ]
                )  # N*3
                reprojected_kpts, initial_depth_ = project_point_cloud_to_image(
                    intrinisic, initial_pose, point_cloud
                )

                initial_depth = np.ones((keypoints.shape[0],)) * -1
                initial_point_cloud = np.ones((keypoints.shape[0], 3)) * -1
                initial_depth[occupied_mask] = initial_depth_
                initial_point_cloud[occupied_mask] = point_cloud

            else:
                initial_point_cloud, initial_depth, keypoints, all_kpt_status = None, None, None, None
                is_keyframe = False

            colmap_frame_dict.update(
                {
                    colmap_frameID: {
                        "initial_pose": initial_pose,
                        "initial_point_cloud": initial_point_cloud,
                        "intrinsic": intrinisic,
                        "keypoints": keypoints,
                        "is_keyframe": is_keyframe,
                        "initial_depth": initial_depth,
                        "all_kpt_status": all_kpt_status,
                    }
                }
            )

    def get_frameID2colmapID(self, frame_IDs, frame_names, colmap_images, only_basename_in_colmap=False):
        # frame_id equal to frame_idx
        frameID2colmapID_dict = {}
        colmapID2frameID_dict = {}
        for frame_ID in frame_IDs:
            frame_name = frame_names[frame_ID]
            frame_name = osp.basename(frame_name) if only_basename_in_colmap else frame_name

            for colmap_image in colmap_images.values():
                if frame_name == colmap_image.name:
                    # Registrated scenario
                    frameID2colmapID_dict[frame_ID] = colmap_image.id
                    colmapID2frameID_dict[colmap_image.id] = frame_ID
                    break
            if frame_ID not in frameID2colmapID_dict:
                # -1 means not registrated
                frameID2colmapID_dict[frame_ID] = -1
        return frameID2colmapID_dict, colmapID2frameID_dict

    def get_keyframes_by_scale(self, colmap_images, colmap_3ds, verbose=True, scale_strategy='largest'):
        # Prepare required data structures:
        p3D_observed = {} # {img_id: {p3d_idx set}}
        img_p3d_to_p2d = {}
        colmap_images_state = {} # {colmap_imageID:{state: np.array [N]}}
        for img_id, colmap_image in colmap_images.items():
            state_value = -2 * np.ones((colmap_image.point3D_ids.shape[0],))
            colmap_unregisted_mask = colmap_image.point3D_ids == -1
            state_value[colmap_unregisted_mask] = -1

            colmap_images_state[img_id] = {'state': state_value}

        colmap_3d_states = {}
        # Build keypoints state and colmap 3D state.
        for point3D_id, point_3d in tqdm(colmap_3ds.items(), disable=not verbose):
            observed_images = point_3d.image_ids.tolist()

            intrins = []
            extrins = []
            points3D = []

            for i, img_id in enumerate(observed_images):
                intrins.append(self.image_intrin_extrins[img_id]['intrin']) # 3*3
                extrins.append(convert_pose2T(self.image_intrin_extrins[img_id]['extrin'])) # 4*4
                points3D.append(point_3d.xyz[None]) # 1 * 3
            intrins, extrins, points3D = map(lambda x: np.stack(x, axis=0), [intrins, extrins, points3D]) # N * 3 * 3
            kpt_reproj, depth = project_point_cloud_to_image(intrins, extrins, points3D) # depth: N * 1
            f = intrins[:, 0,0] # N
            scales = (f / (depth[:, 0] + 1e-4)).tolist()

            # Assign the feature track to the node with scale strategy
            index = np.argsort(np.array(scales))
            if scale_strategy == 'largest':
                assigned_idx = index[-1]
            elif scale_strategy == 'smallest':
                assigned_idx = index[0]
            elif scale_strategy == 'middle':
                assigned_idx = index[int(len(index) // 2)]
            elif scale_strategy == 'random':
                assigned_idx = np.random.choice(index)
            else:
                raise NotImplementedError
            assigned_img_id, assigned_point2D_idx = point_3d.image_ids[assigned_idx], point_3d.point2D_idxs[assigned_idx]

            # Update 3D state:
            colmap_3d_states[point3D_id] = (assigned_img_id, assigned_point2D_idx)

            # Update 2D state:
            for idx, (img_id, point2d_idx) in enumerate(zip(point_3d.image_ids.tolist(), point_3d.point2D_idxs.tolist())):
                if idx == assigned_idx:
                    assert (point2d_idx == assigned_point2D_idx) and (img_id == assigned_img_id)
                    colmap_images_state[assigned_img_id]['state'][assigned_point2D_idx] = point3D_id
                else:
                    colmap_images_state[img_id]['state'][point2d_idx] = -3

        # Get keyframes:
        keyframe_dict = {}
        for colmap_image_id, colmap_image_state in colmap_images_state.items():
            assert np.sum(colmap_image_state["state"] == -2) == 0
            keyframe_dict[colmap_image_id] = (colmap_image_state['state'][colmap_image_state['state'] >= 0]).astype(np.int32)

        return keyframe_dict, colmap_3d_states
   
    def update_kpts_by_current_model_projection(self, fix_ref_node=True):
        for image_id, image in self.colmap_images.items():
            # Load pose and intrinsics:
            extrinsic = get_pose_from_colmap_image(image) # w2c
            intrinsic = get_intrinsic_from_colmap_camera(
                self.colmap_cameras[image.camera_id]
            )

            # Find corresponding 3D points:
            point_3d_ids = image.point3D_ids
            point_2d_idx = np.arange(0,point_3d_ids.shape[0])
            registrated_mask = point_3d_ids > -1
            point3D_coords = []
            reference_node_mask = []
            for idx, point3D_id in enumerate(point_3d_ids[registrated_mask].tolist()):
                point3D_coords.append(self.colmap_3ds[point3D_id].xyz)
                ref_frame_id, ref_kpt_id = self.point_cloud_assigned_imgID_kptID[point3D_id]
                if ref_frame_id == image_id and ref_kpt_id == point_2d_idx[registrated_mask][idx]:
                    reference_node_mask.append(1)
                else:
                    reference_node_mask.append(0)

            point3D_coords = np.stack(point3D_coords) if len(point3D_coords) > 0 else np.empty((0,3)) # N*3
            reference_node_mask = np.array(reference_node_mask, dtype=np.bool)

            # Project:
            proj2D, depth = project_point_cloud_to_image(intrinsic, extrinsic, point3D_coords)

            # Update:
            if fix_ref_node:
                self.colmap_images[image_id].xys[registrated_mask][~reference_node_mask] = proj2D[~reference_node_mask]
            else:
                self.colmap_images[image_id].xys[registrated_mask] = proj2D
    
    def update_refined_kpts_to_colmap_multiview(self, fine_match_results):
        for bag_results in fine_match_results:
            for refined_pts in bag_results:
                location, image_id, pt2d_idx = refined_pts[:2], int(refined_pts[2]), int(refined_pts[3])

                pt3d_id = self.colmap_images[image_id].point3D_ids[pt2d_idx]
                duplicate_idxs = np.concatenate(np.where(self.colmap_images[image_id].point3D_ids == pt3d_id), axis=0)

                self.colmap_images[image_id].xys[duplicate_idxs, :] = location + 0.5
    
    def save_colmap_model(self, save_dir):
        # Write results to colmap file format
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        write_model(
            self.colmap_cameras,
            self.colmap_images,
            self.colmap_3ds,
            save_dir,
            ext=".bin",
        )
        write_model(
            self.colmap_cameras,
            self.colmap_images,
            self.colmap_3ds,
            save_dir,
            ext=".txt",
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        img_name = self.img_list[idx]
        if self.preload:
            img_scale = self.img_dict[img_name]
        else:
            img_scale = read_rgb(
                img_name,
                (self.img_resize,) if self.img_resize is not None else None,
                resize_no_larger_than=True,
                # pad_to=self.img_resize,
                pad_to=None,
                df=self.df,
                ret_scales=True,
            )

        img, scale, original_hw = map(lambda x: x, img_scale)  # with dataloader operation
        data = {
            "image": img,  # 1*H*W because no dataloader operation, if batch: 1*H*W
            "scale": scale,  # 2
            "f_name": img_name,
            "img_name": img_name,
            "frameID": idx,
            "img_path": [img_name],
        }
        return data

