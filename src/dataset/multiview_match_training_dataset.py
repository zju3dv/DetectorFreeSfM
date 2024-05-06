import pickle
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .utils import read_grayscale, read_depth_megadepth, read_rgb

class MultiviewMatchingDataset(Dataset):
    def __init__(
        self,
        data_root,
        scene_info_path,
        mode="train",
        img_resize=800,
        depth_max_size=2000,
        coarse_scale=8,
        augmentor=None,
        padding=False,
        sort_type='largest_scale',
        df=8,
        percent=1.0,
        **kwargs
    ) -> None:
        super().__init__()
        with open(scene_info_path, "rb") as f:
            scene_info = pickle.load(f)

        # Preserve required data:
        self.scene_name = scene_info["scene_name"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.image_tuples = scene_info["image_tuples"]
        self.overlap_scores = scene_info['overlap_score']
        self.img_global_scales = scene_info['scales']
        del scene_info

        self.data_root = data_root
        self.sort_type = sort_type
        self.mode = mode

        self.img_resize = img_resize
        self.depth_max_size = depth_max_size
        self.coarse_scale = coarse_scale
        self.padding = padding  # for batch_size > 1, padding must be used
        self.df = df
        self.augmentor = augmentor

        self.image_tuples = self.image_tuples[:: int(1 / percent)]

    def __len__(self):
        return len(self.image_tuples)

    def __getitem__(self, index):
        image_tuple = self.image_tuples[index]
        overlap_score = self.overlap_scores[index]
        img_global_scale = self.img_global_scales[index]

        # Sort images by scale, use image with smallest scale as reference image:
        sort_index = np.argsort(np.array(img_global_scale) * -1) # Decending
        if self.sort_type == 'largest_scale':
            sort_index = np.concatenate([sort_index[[0]], np.random.permutation(sort_index[1:])])
        elif self.sort_type == 'smallest_scale':
            sort_index = np.concatenate([sort_index[[-1]], np.random.permutation(sort_index[:-1])])
        elif self.sort_type == 'middle_scale':
            middle_index = len(sort_index) // 2
            sort_index = np.concatenate([sort_index[[middle_index]], np.random.permutation(np.delete(sort_index, middle_index))])
        elif self.sort_type == 'random':
            sort_index = np.random.permutation(sort_index)
        else:
            raise NotImplementedError
        image_tuple = np.array(image_tuple)[sort_index]

        images, depth_maps, intrinsics, poses, img_scales, img_original_hw, pad_masks = [], [], [], [], [], [], []
        # Load images:
        for image_id in image_tuple:
            image_path = osp.join(self.data_root ,self.image_paths[image_id])
            depth_path = osp.join(self.data_root, self.depth_paths[image_id])
            R, t = self.poses[image_id]
            T = np.eye(4)
            T[:3,:3] = R
            T[:3, 3] = t
            pose = torch.from_numpy(T).to(torch.float32)
            intrinsic = torch.from_numpy(self.intrinsics[image_id]).to(torch.float32)

            ts_img, scale, original_hw, ts_mask = read_rgb(
                image_path,
                resize=(self.img_resize,),
                pad_to=self.img_resize if self.padding else None,
                ret_scales=True,
                ret_pad_mask=True,
                augmentor=self.augmentor,
                df=self.df,
            ) # scales: [h_origin / h_resize, w_origin / w_resize]

            # Load depth
            ts_depth = read_depth_megadepth(depth_path, pad_to=self.depth_max_size)

            # Append:
            images.append(ts_img)
            depth_maps.append(ts_depth)
            intrinsics.append(intrinsic)
            poses.append(pose)
            img_scales.append(scale)
            img_original_hw.append(original_hw)
            pad_masks.append(ts_mask)
        
        if self.padding:
            # Images with same size:
            images = torch.stack(images, dim=0) # N * 1 * H * W
            pad_masks = torch.stack(pad_masks)

        depth_maps, intrinsics, poses, img_scales, img_original_hw = map(lambda x: torch.stack(x, dim=0), [depth_maps, intrinsics, poses, img_scales, img_original_hw])

        # Get relative poses:
        relative_poses = []
        for i in range(len(image_tuple)):
            if i == 0:
                relative_poses.append(torch.eye(4))
            else:
                relative_poses.append(poses[i] @ poses[0].inverse())
        relative_poses = torch.stack(relative_poses, dim=0)

        data =  {
            "scene_name": self.scene_name,
            "images": images,
            "depth": depth_maps,
            "scales": img_scales, # N * 2
            "original_hw": img_original_hw, # N * 2
            "intrinsics": intrinsics,
            "extrinsics": poses, # w2c N * 4 * 4
            "relative_poses": relative_poses, # id_0 2 id_j
        }

        if self.padding:
            data.update({'masks': pad_masks})

        return data
