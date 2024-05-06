from loguru import logger
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data.dataloader import DataLoader
from torch import distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler, RandomSampler
from pathlib import Path
from tqdm import tqdm
from typing import Iterable
import numpy as np
import os.path as osp
from src.dataset.sampler.sampler import RandomConcatSampler
from src.dataset.multiview_match_training_dataset import MultiviewMatchingDataset

def get_local_split(items: np.ndarray, world_size: int, rank: int, seed: int):
    n_items = len(items)
    items_permute = np.random.RandomState(seed).permutation(items)
    if n_items % world_size == 0:
        padded_items = items_permute
    else:
        padding = np.random.RandomState(seed).choice(items,
                                                    world_size - (n_items % world_size),
                                                    replace=True)
        padded_items = np.concatenate([items_permute, padding])
        assert len(
            padded_items) % world_size == 0, f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank = len(padded_items) // world_size
    local_items = padded_items[n_per_rank * rank: n_per_rank * (rank+1)]
    # local_items = padded_items[rank:len(padded_items):word_size]
    return local_items

class MultiviewMatcherDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.scene_info_dir = kwargs['scene_info_dir'] # directory contains scene_id.pkl
        self.dataset_path = kwargs['dataset_path']
        self.train_list_path = kwargs['train_list_path'] # txt file contains training scene
        self.val_list_path = kwargs['val_list_path']
        # image path; depth path

        assert osp.exists(self.scene_info_dir), self.scene_info_dir
        assert osp.exists(self.dataset_path), self.dataset_path

        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]

        # Data related
        self.train_percent = kwargs["train_percent"]
        self.val_percent = kwargs["val_percent"]

        self.sort_type = kwargs['sort_type']
        self.img_pad = kwargs["img_pad"]
        self.img_resize = kwargs["img_resize"]
        self.depth_max_size = kwargs['depth_max_size']
        self.df = kwargs["df"]
        self.coarse_scale = kwargs["coarse_scale"]
        self.augmentor = None

        # Loader parameters:
        self.train_loader_params = {
            "batch_size": self.batch_size,
            # "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.val_loader_params = {
            "batch_size": 1,
            # "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.test_loader_params = {
            "batch_size": 1,
            # "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

        # Sampler:
        self.data_sampler = kwargs['data_sampler'] # 
        self.n_samples_per_subset = kwargs['n_samples_per_subset']  # 200
        self.subset_replacement = kwargs['subset_sample_replacement']  # True
        self.shuffle = kwargs['shuffle_within_epoch_subset']  # True
        self.repeat = kwargs['repeat_sample']  # 1
        
        # RandomSampler
        self.replacement = kwargs['replacement']  # False, whether draw with replacement or not.
        self.num_samples = kwargs['num_samples']  # None, can be n_samples_per_subset * n_subsets_per_gpu

        self.random_seed = kwargs['random_seed']


    def prepare_data(self):
        pass

    def build_concat_dataset(self,
                             data_root: str,
                             scene_names: Iterable[str],
                             scene_info_dir: str,
                             mode: str):
        datasets = []
        augmentor = self.augmentor if mode == 'train' else None
        for scene_name in tqdm(scene_names, desc=f'[rank:{self.rank}] Loading {mode} datasets',
                             disable=int(self.rank) != 0):
            scene_info_path = osp.join(scene_info_dir, scene_name + '.pkl')
            # `ScanNetDatasetNpz`/`MegaDepthDatasetNpz` load all data from npz_path in __init__, which might take times.
            datasets.append(
                MultiviewMatchingDataset(data_root, scene_info_path, mode=mode,
                                    img_resize=self.img_resize, depth_max_size=self.depth_max_size, 
                                    coarse_scale=self.coarse_scale, sort_type=self.sort_type,
                                    augmentor=augmentor, padding=self.img_pad, df=self.df))
        return ConcatDataset(datasets)


    def setup_dataset(self, mode='train'):
        """ Setup train / validation / test set"""
        if mode == 'train':
            # Load train scenes:
            with open(self.train_list_path, 'r') as f:
                train_scenes_list = [name.split('\n')[0] for name in f.readlines()]
            local_scene_names = get_local_split(train_scenes_list, self.world_size, self.rank, self.random_seed)
        else:
            with open(self.val_list_path, 'r') as f:
                val_scenes_list = [name.split('\n')[0] for name in f.readlines()]
            local_scene_names = val_scenes_list
        logger.info(f'[rank:{self.rank}] #scenes assigned: {len(local_scene_names)} | Start loading {mode} datasets...')

        dataset_builder = self.build_concat_dataset
        dataset = dataset_builder(self.dataset_path, local_scene_names, self.scene_info_dir, mode=mode)
        return dataset

    def setup(self, stage=None):
        """ Setup train & val / test dataset. This method will be called by PL automatically
        or called by user manually to build up the data set of specific stage.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase. (regulation set by PL)
        """
        assert stage in ['fit', 'test']
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(
                f"[rank:{self.rank}] world_size: {self.world_size} | data_loading: {'local'}")
        except AssertionError as a_e:
            logger.warning(str(a_e) + " (wolrd_size will be set to 1 and rank = 0)")
            self.world_size = 1
            self.rank = 0
        
        if stage == 'fit':
            self.train_dataset = self.setup_dataset(mode='train')
            self.val_dataset = self.setup_dataset(mode='val')
        else:
            self.test_dataset = self.setup_dataset(mode='test')

    def train_dataloader(self):
        """for training dataset, use custom sampler instead of DistributedSampler"""
        assert self.data_sampler in ['scene_balance', 'random', 'normal', 'none']
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init.')

        if self.data_sampler == 'scene_balance':
            # RandomConcatSampler (custom sampler, in a scene balance fashion, assume dataset is splitted to ranks)
            sampler = RandomConcatSampler(self.train_dataset, self.n_samples_per_subset, self.subset_replacement,
                                          self.shuffle, self.repeat, self.random_seed)
        elif self.data_sampler == 'random':
            sampler = RandomSampler(self.train_dataset, replacement=self.replacement,
                                    num_samples=self.num_samples, generator=torch.manual_seed(self.random_seed))
        elif self.data_sampler == 'normal':  # for debugging / overfitting models
            raise NotImplementedError
            sampler = DistributedSampler(self.train_dataset, shuffle=False)
        else:  # 'none'
            sampler = None

        return DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(dataset=self.val_dataset, sampler=sampler, **self.val_loader_params)

    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(dataset=self.test_dataset, sampler=sampler, **self.test_loader_params)
