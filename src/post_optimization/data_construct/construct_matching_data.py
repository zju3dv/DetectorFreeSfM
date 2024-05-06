from cv2 import transpose
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from time import time
from loguru import logger
from src.utils.ray_utils import chunks_balance
from src.post_optimization.utils.geometry_utils import project_point_cloud_to_image, convert_pose2T, transform_point_cloud_to_camera

class FeatureTrackStatus:
    def __init__(
        self,
        points3D_assignment,
        points3D,
        frame_dict,
        max_track_length=9,
        max_num_img_in_bag=16,
        verbose=False,
    ):
        self.point3D_assignment = points3D_assignment
        self.frame_dict = frame_dict
        self.point3D_ids_2_idx_dict = {}
        self.track_length = []
        self.total_track_length = 0
        reference_img_ids = []
        query_imgs_ids = []
        for idx, point3D_id in enumerate(points3D_assignment.keys()):
        # for idx, (point3D_id, point3D) in enumerate(points3D.items()):
            point3D = points3D[point3D_id]
            self.point3D_ids_2_idx_dict[point3D_id] = idx
            # self.track_length.append(len(point3D.point2D_idxs)) # Duplicate image ids exists
            assigned_img_id, assigned_point_idx = points3D_assignment[point3D_id]
            reference_img_ids.append(assigned_img_id)

            image_ids = point3D.image_ids
            query_img_ids = list(
                set(image_ids) - {assigned_img_id}
            )  # Remove duplicated image ids
            query_imgs_ids.append(query_img_ids)
            track_length = len(query_img_ids) + 1
            self.total_track_length += track_length - 1 # total number of query nodes
            self.track_length.append(track_length)


        self.max_track_length = max_track_length
        self.max_num_img_in_bag = max_num_img_in_bag if max_num_img_in_bag is not None else max_track_length

        # Status, which will updated constantly
        # NOTE: occupied is marked as -1
        self.reference_img_ids = reference_img_ids  # ["image_id"]
        self.query_imgs_ids = query_imgs_ids  # [['image_id"...]]
        self.verbose = verbose  # For debug
        self.pt3d_ids = list(self.point3D_ids_2_idx_dict.keys()) # For fast index

    def find_current_max_track_id(self):
        max_index = np.argmax(np.array(self.track_length))
        assert self.reference_img_ids[max_index] != -1
        return self.pt3d_ids[max_index]

    def get_images_from_trackID_and_update(self, point3D_id):
        """
        Intend for the current longest track
        """
        idx = self.point3D_ids_2_idx_dict[point3D_id]
        referece_img_id = self.reference_img_ids[idx]
        track_length = self.track_length[idx]
        if len(self.query_imgs_ids[idx]) > self.max_track_length - 1:
            query_img_ids = self.query_imgs_ids[idx][: (self.max_track_length - 1)]
            del self.query_imgs_ids[idx][
                : (self.max_track_length - 1)
            ]  # Update query image status
            self.total_track_length -= self.max_track_length - 1
            self.track_length[idx] -= self.max_track_length - 1
        else:
            query_img_ids = self.query_imgs_ids[idx]
            # Update status:
            self.query_imgs_ids[idx] = []  # Empty
            self.total_track_length -= self.track_length[idx] - 1
            self.track_length[idx] = 1
        return referece_img_id, query_img_ids, track_length

    def get_images_from_trackID(self, point3D_id):
        idx = self.point3D_ids_2_idx_dict[point3D_id]
        referece_img_id = self.reference_img_ids[idx]
        query_image_ids = self.query_imgs_ids[idx]
        track_length = self.track_length[idx]
        return referece_img_id, query_image_ids, track_length

    def update_track_status(self, point3D_id, excluded_img_ids):
        idx = self.point3D_ids_2_idx_dict[point3D_id]
        query_img_ids = self.query_imgs_ids[idx]
        assert (
            len(set(excluded_img_ids) - set(query_img_ids)) == 0
        )  # need to all included
        self.query_imgs_ids[idx] = list(
            set(query_img_ids) - set(excluded_img_ids)
        )
        self.total_track_length -= len(excluded_img_ids)
        self.track_length[idx] -= len(excluded_img_ids)

    def get_relevant_tracks(self, exclude_track_id, bag_image_ids):
        relevant_tracks = []  # [track_id]
        track_corresponding_imgs = []  # [[ref_img_id, [query_images]]]
        for image_id in bag_image_ids:
            # Get processed track ids
            processed_track_ids = self.frame_dict[image_id]
            for track_id in processed_track_ids:
                if track_id == exclude_track_id:
                    continue
                if track_id not in self.point3D_assignment:
                    # Eliminate tracks which will process by other workers
                    continue

                (
                    reference_img_id,
                    query_img_ids,
                    track_length,
                ) = self.get_images_from_trackID(track_id)

                if track_length == 1:
                    # Already empty
                    continue

                assert reference_img_id == image_id  # Correction check
                common_img_ids = set(query_img_ids) & set(bag_image_ids)
                exclude_img_ids = set(query_img_ids) - set(bag_image_ids)
                # if len(common_img_ids) == len(bag_image_ids) or len(exclude_img_ids) >= 3:
                exclude_value = 0  # NOTE: this number should not larger than 3, otherwise cause too much bags

                add_quota = self.max_num_img_in_bag - len(bag_image_ids)
                if add_quota > 0 and len(exclude_img_ids) != 0:
                    # Can add some images to bag
                    extra_added = list(exclude_img_ids)[:add_quota] # TODO: constraint max track length?
                    bag_image_ids += extra_added
                    exclude_img_ids -= set(extra_added)
                    common_img_ids |= set(extra_added)

                if (
                    (len(common_img_ids) == len(query_img_ids)
                    or len(exclude_img_ids) >= exclude_value) 
                    and (len(common_img_ids) != 0)
                    # and (len(common_img_ids) >= (len(exclude_img_ids)))
                    # and ((len(bag_image_ids) - len(common_img_ids)) < (len(common_img_ids) * 2)) # to aviod padding too much?
                    # and ((len(common_img_ids) > len(bag_image_ids) // 2) or track_length < 3) # to aviod padding too much?
                    # and ((len(common_img_ids) > len(bag_image_ids) // 2)) # to aviod padding too much?
                ):
                    # accpet
                    self.update_track_status(
                        track_id, common_img_ids
                    )  # Since common images will be excluded
                    relevant_tracks.append(track_id)
                    track_corresponding_imgs.append([image_id, list(common_img_ids)])
                else:
                    continue
        return relevant_tracks, track_corresponding_imgs, bag_image_ids


    def is_empty(self):
        print(self.total_track_length) if self.verbose else None
        return self.total_track_length != 0

class MatchingMultiviewData(Dataset):
    """
    Construct image bags for MultiviewMatcher
    """

    def __init__(self, colmap_image_dataset, config, worker_split_idxs=None) -> None:
        super().__init__()
        self.max_track_length = config['max_track_length']
        self.chunk = config['chunk']  # Used to limit number of tracks in a bag

        # Colmap info
        self.colmap_image_dataset = colmap_image_dataset
        self.colmap_frame_dict = colmap_image_dataset.keyframe_dict
        self.point3d_assignment_all = colmap_image_dataset.point_cloud_assigned_imgID_kptID
        self.colmap_3ds = colmap_image_dataset.colmap_3ds
        self.colmap_images = colmap_image_dataset.colmap_images
        self.colmap_cameras = colmap_image_dataset.colmap_cameras
        self.colmap_intrin_extrin = colmap_image_dataset.image_intrin_extrins
        if worker_split_idxs is None:
            self.point3d_assignment = self.point3d_assignment_all
        else:
            self.point3d_assignment = {} 
            all_assignment = list(self.point3d_assignment_all.items())
            t0 = time()
            for idx in worker_split_idxs:
                track_id, assignment = all_assignment[idx]
                self.point3d_assignment[track_id] = assignment
            t1 = time()
            logger.info(f"Split takes: {t1-t0}")

        # Split track and assign bags
        t0 = time()
        logger.info(f"Assign bags begin!")
        self.image_bags = self.assign_bags()
        t1 = time()
        logger.info(f"Assign bags takes: {t1 - t0}")
        self.image_bags = self.chunk_bags()
        t2 = time()
        logger.info(f"Chunk bags takes: {t2 - t1}")

    def chunk_bags(self):
        chunked_bags = []
        for image_bag in self.image_bags:
            if len(image_bag["track_ids"]) > self.chunk:
                n_split = (len(image_bag["track_ids"]) // self.chunk) + 1
                chunked_track_ids = chunks_balance(image_bag["track_ids"], n_split)
                chunked_corres_img_ids = chunks_balance(
                    image_bag["track_corresponding_imgs"], n_split
                )
                chunked_bags += [
                    {
                        "bag_image_ids": image_bag["bag_image_ids"],
                        "track_ids": track_ids,
                        "track_corresponding_imgs": corres_img_ids,
                    }
                    for track_ids, corres_img_ids in zip(
                        chunked_track_ids, chunked_corres_img_ids
                    )
                ]
            else:
                chunked_bags += [image_bag]
        
        return chunked_bags

    def assign_bags(self):
        """
        while status data not empty:
            1. sort by feature track length and find the largest one
            2. get top track length nodes(must include reference node) as bag
            3. get relevant tracks, which reference node within the bag; and all query node within bag(which is good) or len(query node with in bag) < 1/2 track_length
            4. update status
        """
        verbose = False
        status = FeatureTrackStatus(
            self.point3d_assignment,
            self.colmap_3ds,
            self.colmap_frame_dict,
            max_track_length=self.max_track_length,
            verbose=verbose,
        )
        bags = []
        while status.is_empty():
            point3D_id = status.find_current_max_track_id()
            (
                referece_img_id,
                query_img_ids,
                _,
            ) = status.get_images_from_trackID_and_update(point3D_id)
            image_bag = [referece_img_id] + query_img_ids
            relevant_tracks, corresponding_imgs, image_bag = status.get_relevant_tracks(point3D_id, image_bag)
            track_ids = [point3D_id] + relevant_tracks
            corresponding_imgs = [[referece_img_id, query_img_ids]] + corresponding_imgs
            bags.append(
                {
                    "bag_image_ids": image_bag,
                    "track_ids": track_ids,
                    "track_corresponding_imgs": corresponding_imgs,
                }
            )
        return bags

    def buildDataBag(self, data_list):
        # data0: dict, data1: dict
        data = {}
        preserve_data_name_convert = {"image": "images", "scale": "scales"}
        for i, data_part in enumerate(data_list):
            for key, value in data_part.items():
                if key not in preserve_data_name_convert:
                    continue
                else:
                    key_new = preserve_data_name_convert[key]

                if key_new in data:
                    data[key_new] += [value]
                else:
                    data[key_new] = [value]

        for key, value in data.items():
            if key is 'images':
                continue
            data[key] = torch.stack(value, dim=0)  # N * ..
        return data
    
    def get_point_scale(self, img_id, point3D_id):
        point3D_coord = self.colmap_3ds[point3D_id].xyz # 3
        intrin = self.colmap_intrin_extrin[img_id]['intrin'] # 3*3
        extrin = self.colmap_intrin_extrin[img_id]['extrin'] # [R,t]
        reproj_pt2d, depth = project_point_cloud_to_image(intrin, extrin, point3D_coord[None])

        f = intrin[0,0]
        return f / (depth[0] + 1e-4)
    
    def get_relative_view_point(self, src_img_id, dst_img_id, point3D_id):
        point3D_coord = self.colmap_3ds[point3D_id].xyz # 3
        src_extrin = convert_pose2T(self.colmap_intrin_extrin[src_img_id]['extrin']) # 4*4
        dst_extrin = convert_pose2T(self.colmap_intrin_extrin[dst_img_id]['extrin']) # 4*4

        f = transform_point_cloud_to_camera(src_extrin, point3D_coord[None]) # [1 * 3]
        relative_pose = src_extrin @ np.linalg.inv(dst_extrin)
        t = relative_pose[:3, [3]].T # [1 * 3]

        a = f - t # [1 * 3]

        f_norm, t_norm, a_norm = map(lambda x: np.linalg.norm(x, axis=-1, keepdims=True), [f, t, a])

        alpha = np.arccos((f @ t.T) / (f_norm * t_norm + 1e-6))
        beta = np.arccos((a @ (-1 * t.T)) / (a_norm * t_norm + 1e-6))
        gamma = np.pi - alpha - beta
        view_point_vector = (t / (t_norm + 1e-6)) * gamma
        return view_point_vector[0] # [3]
    

    def __len__(self):
        return len(self.image_bags)

    def __getitem__(self, index):
        image_bag_info = self.image_bags[index]
        bag_image_ids = image_bag_info["bag_image_ids"]
        single_image_list = [
            self.colmap_image_dataset[
                self.colmap_image_dataset.colmapID2frameID_dict[image_id]
            ]
            for image_id in bag_image_ids
        ]
        colmap_img_id2bag_img_idx = {
            img_id: img_idx
            for img_id, img_idx in zip(bag_image_ids, np.arange(len(bag_image_ids)))
        }

        data = self.buildDataBag(single_image_list)
        M_tracks = len(image_bag_info["track_ids"])
        N_max_query_node = len(bag_image_ids) - 1
        (
            reference_node,
            reference_movable_mask,
            reference_pt2d_idxs,
            reference_image_ids,
            query_nodes,
            query_mask,
            referece_image_idx,
            reference_points_scales,
            query_image_idxs,
            query_image_ids,
            query_points_scales,
            query_pt2d_idxs,
            relative_view_points,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])

        for track_id, corresponding_imgs in sorted(zip(
            image_bag_info["track_ids"], image_bag_info["track_corresponding_imgs"]
        ), key=lambda x:len(x[1][1]), reverse=True) : # order by track length (descending)
            reference_img_id_, query_img_ids_ = corresponding_imgs
            # Find corresponding points id
            assert self.point3d_assignment[track_id][0] == reference_img_id_
            reference_pt2d_idx = self.point3d_assignment[track_id][1]

            referece_image_idx.append(colmap_img_id2bag_img_idx[reference_img_id_])
            reference_pt2d_idxs.append(reference_pt2d_idx)
            reference_image_ids.append(reference_img_id_)

            reference_point_scale = self.get_point_scale(reference_img_id_, track_id)
            reference_points_scales.append(reference_point_scale)

            reference_node.append(
                self.colmap_images[reference_img_id_].xys[reference_pt2d_idx]
            )

            (
                query_nodes_,
                query_mask_,
                query_image_idxs_,
                query_image_ids_,
                query_pt_scales_,
                query_pt2d_idxs_,
                relative_view_points_,
            ) = ([], [], [], [], [], [], [])
            for idx in range(N_max_query_node):
                if idx >= len(query_img_ids_):
                    # Padding
                    query_nodes_.append(np.array([1, 1]))
                    query_mask_.append(False)
                    query_image_idxs_.append(-1)
                    query_image_ids_.append(-1)
                    query_pt2d_idxs_.append(-1)
                    query_pt_scales_.append(reference_point_scale)
                    relative_view_points_.append(np.array([0, 0, 0]))

                else:
                    query_img_id_ = query_img_ids_[idx]
                    query_img_idx_ = np.where(
                        self.colmap_3ds[track_id].image_ids == query_img_id_
                    )
                    query_pt2d_idx_ = self.colmap_3ds[track_id].point2D_idxs[
                        query_img_idx_
                    ]
                    query_node_ = np.mean(self.colmap_images[query_img_id_].xys[query_pt2d_idx_], axis=0) # [2]

                    query_nodes_.append(
                        query_node_
                    )  # [2]
                    query_mask_.append(True)
                    query_image_idxs_.append(colmap_img_id2bag_img_idx[query_img_id_])
                    query_image_ids_.append(query_img_id_)
                    query_pt2d_idxs_.append(query_pt2d_idx_[0])
                    query_pt_scales_.append(self.get_point_scale(query_img_id_, track_id))
                    relative_view_points_.append(self.get_relative_view_point(reference_img_id_, query_img_id_, track_id))

            query_nodes.append(np.stack(query_nodes_))
            query_mask.append(np.stack(query_mask_))
            query_image_idxs.append(np.stack(query_image_idxs_))
            query_image_ids.append(np.stack(query_image_ids_))
            query_pt2d_idxs.append(np.stack(query_pt2d_idxs_))
            query_points_scales.append(np.stack(query_pt_scales_))
            relative_view_points.append(np.stack(relative_view_points_))

        # Type convert:
        (
            reference_node,
            reference_pt2d_idxs,
            reference_image_ids,
            query_nodes,
            query_mask,
            referece_image_idx,
            reference_points_scales,
            query_image_idxs,
            query_image_ids,
            query_pt2d_idxs,
            query_points_scales,
            relative_view_points
        ) = map(
            lambda x: torch.from_numpy(np.stack(x)),
            [
                reference_node,
                reference_pt2d_idxs,
                reference_image_ids,
                query_nodes,
                query_mask,
                referece_image_idx,
                reference_points_scales,
                query_image_idxs,
                query_image_ids,
                query_pt2d_idxs,
                query_points_scales,
                relative_view_points
            ],
        )

        scales_absolute = torch.cat([reference_points_scales[..., None], query_points_scales], dim=-1) # B * N_track * N_view
        if 'scales' in data:
            scales_absolute /= data['scales'][..., 0][None]
        scales_relative = scales_absolute / (scales_absolute[..., [0]]) # 1 * M * N

        # Fill the relative point of the reference node with zero:
        relative_view_points = torch.cat([torch.zeros((relative_view_points.shape[0], 1, 3)), relative_view_points], dim=-2) # [1, M, N, 3]
        data.update(
            # M is number of track, while N is track length
            {
                "query_points": reference_node.to(torch.float32) - 0.5,  # [M, 2] # M tracks
                "reference_points_coarse": query_nodes.transpose(
                    0, 1
                ).to(torch.float32) - 0.5,  # [N-1, M, 2]
                "track_valid_mask": query_mask.transpose(0, 1),  # [N-1, M,]
                "query_img_idxs": referece_image_idx,  # [M]
                "reference_img_idxs": query_image_idxs.transpose(0, 1),  # [N-1, M]
                "scales_relative": scales_relative.transpose(0, 1), # [N, M]
                "view_point_vector": relative_view_points.transpose(0, 1), # [N, M, 3]

                # Infos used to update model points locations
                "query_img_ids": reference_image_ids, # [M]
                "query_pt2d_idxs": reference_pt2d_idxs, # [M]
                "reference_img_ids": query_image_ids.transpose(0, 1),  # [N-1, M]
                "reference_pt2d_idxs": query_pt2d_idxs.transpose(0, 1),  # [N-1, M]
            }
        )
        return data
