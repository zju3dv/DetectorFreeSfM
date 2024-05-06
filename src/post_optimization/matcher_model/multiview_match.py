from itertools import chain
import os
import math
from loguru import logger
from typing import ChainMap
from src.utils.ray_utils import ProgressBar, chunk_index, chunk_index_balance
from .multiview_match_worker import *


def multiview_matcher(cfgs, dataset_cfgs, colmap_image_dataset, rewindow_size_factor=None, model_idx=None, visualize_dir=None, use_ray=False, ray_cfg=None, verbose=True):
    matcher = build_model(cfgs["model"], rewindow_size_factor, model_idx)

    if not use_ray:
        fine_match_results = matchWorker(
            colmap_image_dataset,
            matcher,
            subset_track_idxs=None,
            visualize=cfgs["visualize"],
            visualize_dir=visualize_dir,
            dataset_cfgs=dataset_cfgs,
            verbose=verbose
        )
    else:
        if ray_cfg is not None:
            # Cfg overwrite
            cfgs['ray'] = {**cfgs['ray'], **dict(ray_cfg)}

        # Initial ray:
        cfg_ray = cfgs["ray"]
        if cfg_ray["slurm"]:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                num_gpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_gpus_per_worker"]),
                local_mode=cfg_ray["local_mode"], ignore_reinit_error=True
            )

        pb = ProgressBar(cfg_ray["n_workers"], "Matching feature tracks...") if verbose else None
        all_subset_ids = chunk_index_balance(
            len(colmap_image_dataset.point_cloud_assigned_imgID_kptID),
            cfg_ray["n_workers"], shuffle=True
        )

        dataset_remote = ray.put(colmap_image_dataset)
        remote_func = matchWorker_ray_wrapper.options(num_cpus=cfg_ray["n_cpus_per_worker"], num_gpus=cfg_ray["n_gpus_per_worker"])
        obj_refs = [
            remote_func.remote(
                dataset_remote,
                matcher,
                subset_ids,
                visualize=cfgs["visualize"],
                visualize_dir=visualize_dir,
                verbose=verbose,
                dataset_cfgs=dataset_cfgs,
                pba=pb.actor if pb is not None else None,
            )
            for subset_ids in all_subset_ids
        ]
        pb.print_until_done() if pb is not None else None
        results = ray.get(obj_refs)
        fine_match_results = list(chain(*results))

    logger.info("Matcher finish!")
    return fine_match_results
