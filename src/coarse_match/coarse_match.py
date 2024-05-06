import os
import ray
import math
import os.path as osp
from loguru import logger
from typing import ChainMap

from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from src.utils.data_io import save_h5, load_h5
from .utils.merge_kpts import Match2Kpts
from .coarse_match_worker import *

cfgs = {
    "data": {
             "img_resize": 1200, 
             "df": 8, 
             "pad_to": None,
             "img_type": "grayscale", # ['grayscale', 'rgb']
             "img_preload": True,
            },
    "matcher": {
        "model": {
            "matcher": 'loftr',
            "type": "coarse_only", #['coarse_only', 'coarse_fine]
            "match_thr": 0.2,
            "matchformer":
            {
                "cfg_path_coarse_only": "third_party/MatchFormer/config/matchformer_coarse_only.py",
                "cfg_path_coarse_fine": "third_party/MatchFormer/config/matchformer_coarse_fine.py",
                "weight_path": "weight/outdoor-large-LA.ckpt",
            },
            "aspanformer":{
                "cfg_path_coarse_only": "third_party/aspantransformer/configs/aspan/outdoor/aspan_test_coarse_only.py",
                "cfg_path_coarse_fine": "third_party/aspantransformer/configs/aspan/outdoor/aspan_test.py",
                "weight_path": "weight/aspanformer_weights/outdoor.ckpt",
            },
            "loftr_official":
            {
                "cfg_path_coarse_only": "third_party/LoFTR/configs/loftr/outdoor/loftr_ds_coarse_only.py",
                "cfg_path_coarse_fine": "third_party/LoFTR/configs/loftr/outdoor/loftr_ds.py",
                "weight_path": "weight/outdoor_ds.ckpt",
            },
            "seed": 666
        },
        "round_matches_ratio": 4,
        "pair_name_split": " ",
    },
    "coarse_match_debug": True,
    "ray": {
        "slurm": False,
        "n_workers": 8, # 16
        "n_cpus_per_worker": 2,
        "n_gpus_per_worker": 0.5,
        "local_mode": False,
    },
}


def detector_free_coarse_matching(
    image_lists,
    covis_pairs_out,
    feature_out,
    match_out,
    img_resize=None,
    img_preload=False,
    matcher='loftr',
    match_type='coarse_only', # ['coarse_only', 'coarse_fine']
    match_thr=0.2,
    match_round_ratio=None,
    use_ray=False,
    ray_cfg=None,
    verbose=True
):

    # Cfg overwrite:
    cfgs['matcher']['model']['type'] = match_type
    cfgs['matcher']['model']['match_thr'] = match_thr
    cfgs['matcher']['model']['matcher'] = matcher
    cfgs['matcher']['round_matches_ratio'] = match_round_ratio
    cfgs['data']['img_resize'] = img_resize
    cfgs['data']['img_preload'] = img_preload
    if 'loftr' in matcher:
        cfgs['data']['df'] = 8
        cfgs['data']['pad_to'] = None
    elif matcher == 'matchformer':
        cfgs['data']['df'] = 8
        cfgs['data']['pad_to'] = -1 # Two image must with same size
    elif matcher == 'aspanformer':
        cfgs['data']['df'] = None # Will pad inner the matching module
        cfgs['data']['pad_to'] = None

    # Construct directory
    base_dir = feature_out.rsplit("/", 1)[0]
    os.makedirs(base_dir, exist_ok=True)
    cache_dir = osp.join(feature_out.rsplit("/", 1)[0], "raw_matches.h5")

    if isinstance(covis_pairs_out, list):
        pair_list = covis_pairs_out
    else:
        assert osp.exists(covis_pairs_out)
        # Load pairs: 
        with open(covis_pairs_out, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')

    if use_ray:
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
                local_mode=cfg_ray["local_mode"],
                ignore_reinit_error=True,
            )

        # Matcher runner
        if not cfgs["coarse_match_debug"] and osp.exists(cache_dir):
            matches = load_h5(cache_dir, transform_slash=True)
            logger.info("Caches raw matches loaded!")
        else:
            pb = ProgressBar(len(pair_list), "Matching image pairs...") if verbose else None
            all_subset_ids = chunk_index(
                len(pair_list), math.ceil(len(pair_list) / cfg_ray["n_workers"])
            )
            remote_func = match_worker_ray_wrapper.options(num_cpus=cfg_ray["n_cpus_per_worker"], num_gpus=cfg_ray["n_gpus_per_worker"])
            obj_refs = [
                remote_func.remote(
                    subset_ids, image_lists, covis_pairs_out, cfgs, pb.actor if pb is not None else None, verbose
                )
                for subset_ids in all_subset_ids
            ]
            pb.print_until_done() if pb is not None else None
            results = ray.get(obj_refs)
            matches = dict(ChainMap(*results))
            logger.info("Matcher finish!")

            logger.info(f"Raw matches cach begin")
            logger.info(f"Raw matches cach finish: {cache_dir}")

        # Combine keypoints
        n_imgs = len(image_lists)
        pb = ProgressBar(n_imgs, "Combine keypoints") if pb is not None else None
        all_kpts = Match2Kpts(
            matches, image_lists, name_split=cfgs["matcher"]["pair_name_split"]
        )
        sub_kpts = chunks(all_kpts, math.ceil(n_imgs / cfg_ray["n_workers"]))
        obj_refs = [
            keypoints_worker_ray_wrapper.remote(sub_kpt, pb.actor if pb is not None else None, verbose=verbose)
            for sub_kpt in sub_kpts
        ]
        pb.print_until_done() if pb is not None else None
        keypoints = dict(ChainMap(*ray.get(obj_refs)))
        logger.info("Combine keypoints finish!")

        # Convert keypoints match to keypoints indexs
        logger.info("Update matches")
        obj_refs = [
            update_matches(
                sub_matches,
                keypoints,
                merge=True if match_round_ratio == 1 else False,
                verbose=verbose,
                pair_name_split=cfgs["matcher"]["pair_name_split"],
            )
            for sub_matches in split_dict(matches, math.ceil(len(matches) / 1))
        ]
        updated_matches = dict(ChainMap(*obj_refs))

        # Post process keypoints:
        keypoints = {k: v for k, v in keypoints.items() if isinstance(v, dict)}
        pb = ProgressBar(len(keypoints), "Post-processing keypoints...") if pb is not None else None
        obj_refs = [
            transform_keypoints_ray_wrapper.remote(sub_kpts, pb.actor if pb is not None else None, verbose=verbose)
            for sub_kpts in split_dict(
                keypoints, math.ceil(len(keypoints) / cfg_ray["n_workers"])
            )
        ]
        pb.print_until_done() if pb is not None else None
        kpts_scores = ray.get(obj_refs)
        final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
        final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    else:
        # Matcher runner
        if not cfgs["coarse_match_debug"] and osp.exists(cache_dir):
            matches = load_h5(cache_dir, transform_slash=True)
            logger.info("Caches raw matches loaded!")
        else:
            all_ids = np.arange(0, len(pair_list))

            matches = match_worker(all_ids, image_lists, covis_pairs_out, cfgs, verbose=verbose)
            logger.info("Matcher finish!")

            logger.info(f"Raw matches cach begin: {cache_dir}")
            save_h5(matches, cache_dir, verbose=verbose)

        # Combine keypoints
        n_imgs = len(image_lists)
        logger.info("Combine keypoints!")
        all_kpts = Match2Kpts(
            matches, image_lists, name_split=cfgs["matcher"]["pair_name_split"]
        )
        sub_kpts = chunks(all_kpts, math.ceil(n_imgs / 1))  # equal to only 1 worker
        obj_refs = [keypoint_worker(sub_kpt, verbose=verbose) for sub_kpt in sub_kpts]
        keypoints = dict(ChainMap(*obj_refs))

        # Convert keypoints match to keypoints indexs
        logger.info("Update matches")
        obj_refs = [
            update_matches(
                sub_matches,
                keypoints,
                merge=True if match_round_ratio == 1 else False,
                verbose=verbose,
                pair_name_split=cfgs["matcher"]["pair_name_split"],
            )
            for sub_matches in split_dict(matches, math.ceil(len(matches) / 1))
        ]
        updated_matches = dict(ChainMap(*obj_refs))

        # Post process keypoints:
        keypoints = {
            k: v for k, v in keypoints.items() if isinstance(v, dict)
        }
        logger.info("Post-processing keypoints...")
        kpts_scores = [
            transform_keypoints(sub_kpts, verbose=verbose)
            for sub_kpts in split_dict(keypoints, math.ceil(len(keypoints) / 1))
        ]
        final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
        final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Reformat keypoints_dict and matches_dict
    # from (abs_img_path0 abs_img_path1) -> (img_name0, img_name1)
    keypoints_renamed = {}
    for key, value in final_keypoints.items():
        keypoints_renamed[osp.basename(key)] = value

    matches_renamed = {}
    for key, value in updated_matches.items():
        name0, name1 = key.split(cfgs["matcher"]["pair_name_split"])
        new_pair_name = cfgs["matcher"]["pair_name_split"].join(
            [osp.basename(name0), osp.basename(name1)]
        )
        matches_renamed[new_pair_name] = value.T

    save_h5(keypoints_renamed, feature_out)
    save_h5(matches_renamed, match_out)

    return final_keypoints, updated_matches
