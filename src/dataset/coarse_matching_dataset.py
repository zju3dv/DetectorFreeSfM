import os.path as osp
import torch
from loguru import logger

from torch.utils.data import Dataset
from .utils import (
    read_grayscale, read_rgb
)

class CoarseMatchingDataset(Dataset):
    """Build image pairs dataset for SfM"""

    def __init__(
        self,
        args,
        image_lists,
        covis_pairs,
        subset_ids,
    ):
        """
        Parameters:
        ---------------
        """
        super().__init__()
        self.img_dir = image_lists
        self.img_resize = args['img_resize']
        self.df = args['df']
        self.pad_to = args['pad_to']
        self.img_dict = {}
        self.preload = args['img_preload']
        self.subset_ids = subset_ids # List

        if isinstance(covis_pairs, list):
            self.pair_list = covis_pairs
        else:
            assert osp.exists(covis_pairs)
            # Load pairs: 
            with open(covis_pairs, 'r') as f:
                self.pair_list = f.read().rstrip('\n').split('\n')

        self.img_read_func = read_grayscale if args['img_type'] == 'grayscale' else read_rgb
        if self.preload:
            logger.info(f"Will preload {len(self.img_dir)} Images for Matching")
            for image_path in self.img_dir:
                image_scale = self.img_read_func(
                    image_path,
                    (self.img_resize,) if self.img_resize is not None else None,
                    df=self.df,
                    pad_to=self.pad_to,
                    ret_scales=True,
                )
                self.img_dict[image_path] = image_scale
            logger.info("Preload Finish!")


    def __len__(self):
        return len(self.subset_ids)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        pair_idx = self.subset_ids[idx]
        if self.preload:
            img_path0, img_path1 = self.pair_list[pair_idx].split(' ')
            img_scale0 = self.img_dict[img_path0]
            img_scale1 = self.img_dict[img_path1]
        else:
            img_path0, img_path1 = self.pair_list[pair_idx].split(' ')
            img_scale0 = self.img_read_func(
                img_path0,
                (self.img_resize,) if self.img_resize is not None else None,
                df=self.df,
                pad_to=self.pad_to,
                ret_scales=True,
            )
            img_scale1 = self.img_read_func(
                img_path1,
                (self.img_resize,) if self.img_resize is not None else None,
                pad_to=self.pad_to,
                df=self.df,
                ret_scales=True,
            )

        img0, scale0, original_hw0 = img_scale0
        img1, scale1, original_hw1 = img_scale1

        data = {
            "image0": img0,
            "image1": img1,
            "scale0": scale0,  # 1*2
            "scale1": scale1,
            "f_name0": osp.basename(img_path0).rsplit('.', 1)[0],
            "f_name1": osp.basename(img_path1).rsplit('.', 1)[0],
            "frameID": pair_idx,
            "pair_key": (img_path0, img_path1),
        }

        return data