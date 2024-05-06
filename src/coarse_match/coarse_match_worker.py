import ray
import os
from loguru import logger

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm

from src.utils.misc import lower_config
from src.utils.torch_utils import update_state_dict, STATE_DICT_MAPPER

from .utils.merge_kpts import agg_groupby_2d
from .utils.detector_wrapper import DetectorWrapper
from ..dataset.coarse_matching_dataset import CoarseMatchingDataset

def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

def build_model(args):
    pl.seed_everything(args['seed'])
    logger.info(f"Use {args['matcher']} as coarse matcher")
    model_type = args['type']
    match_thr = args['match_thr']
    if args['matcher'] == "loftr_official":
        from third_party.LoFTR.src.loftr.loftr import LoFTR
        from third_party.LoFTR.src.config.default import get_cfg_defaults as get_cfg_defaults_loftr

        matcher_args = args['loftr_official']
        cfg = get_cfg_defaults_loftr()
        cfg.merge_from_file(matcher_args[f'cfg_path_{model_type}'])
        match_cfg = lower_config(cfg)
        match_cfg['loftr']['match_coarse']['thr'] = match_thr
        match_cfg['loftr']['coarse']['temp_bug_fix'] = False
        matcher = LoFTR(config=match_cfg['loftr'])
        # load checkpoints
        state_dict = torch.load(matcher_args['weight_path'], map_location="cpu")["state_dict"]
        matcher.load_state_dict(state_dict, strict=True)

        detector = DetectorWrapper()
        detector.eval()
        matcher.eval()

    elif args['matcher'] == 'aspanformer':
        from third_party.aspantransformer.src.ASpanFormer.aspanformer import ASpanFormer
        from third_party.aspantransformer.src.config.default import get_cfg_defaults as get_cfg_defaults_aspanformer

        matcher_args = args['aspanformer']
        config = get_cfg_defaults_aspanformer()
        config.merge_from_file(matcher_args[f'cfg_path_{model_type}'])
        _config = lower_config(config)
        _config['aspan']['match_coarse']['thr'] = match_thr
        matcher = ASpanFormer(config=_config['aspan'], online_resize=True)
        state_dict = torch.load(matcher_args['weight_path'], map_location='cpu')['state_dict']
        matcher.load_state_dict(state_dict,strict=False)

        detector = DetectorWrapper()
        detector.eval()
        matcher.eval()

    elif args['matcher'] == 'matchformer':
        from third_party.MatchFormer.model.matchformer import Matchformer
        from third_party.MatchFormer.config.defaultmf import get_cfg_defaults as get_cfg_defaults_matchformer

        matcher_args = args['matchformer']
        config = get_cfg_defaults_matchformer()
        config.merge_from_file(matcher_args[f'cfg_path_{model_type}'])
        _config = lower_config(config)
        _config['matchformer']['match_coarse']['thr'] = match_thr
        matcher = Matchformer(config=_config['matchformer'],)
        state_dict = torch.load(matcher_args['weight_path'], map_location='cpu')
        matcher.load_state_dict(state_dict,strict=True)

        detector = DetectorWrapper()
        detector.eval()
        matcher.eval()
    else:
        raise NotImplementedError

    return detector, matcher

def extract_preds(data):
    """extract predictions assuming bs==1"""
    m_bids = data["m_bids"].cpu().numpy()
    assert (np.unique(m_bids) == 0).all()
    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()
    mconfs = data["mconf"].cpu().numpy()

    return mkpts0, mkpts1, mconfs

@torch.no_grad()
def extract_matches(data, detector=None, matcher=None):
    detector(data)
    matcher(data)

    mkpts0, mkpts1, mconfs = extract_preds(data)
    return (mkpts0, mkpts1, mconfs)


@torch.no_grad()
def match_worker(subset_ids, image_lists, covis_pairs_out, cfgs, pba=None, verbose=True):
    """extract matches from part of the possible image pair permutations"""
    args = cfgs['matcher']
    detector, matcher = build_model(args['model'])
    detector.cuda()
    matcher.cuda()
    matches = {}
    # Build dataset:
    dataset = CoarseMatchingDataset(cfgs["data"], image_lists, covis_pairs_out, subset_ids)
    dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)

    tqdm_disable = True
    if not verbose:
        assert pba is None
    else:
        if pba is None:
            tqdm_disable = False

    # match all permutations
    for data in tqdm(dataloader, disable=tqdm_disable):
        f_name0, f_name1 = data['pair_key'][0][0], data['pair_key'][1][0]
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        mkpts0, mkpts1, mconfs = extract_matches(
            data_c,
            detector=detector,
            matcher=matcher,
        )

        # Round mkpts to grid-level to construct feature tracks for the later SfM
        if args['model']['type'] is not 'coarse_only' and args['round_matches_ratio'] is not None:
            mkpts0 = np.round((mkpts0 / data['scale0'][:, [1, 0]]) / args['round_matches_ratio']) * args['round_matches_ratio'] * data['scale0'][:, [1, 0]]
            mkpts1 = np.round((mkpts1 / data['scale1'][:, [1, 0]]) / args['round_matches_ratio']) * args['round_matches_ratio'] * data['scale1'][:, [1, 0]]

        # Extract matches (kpts-pairs & scores)
        matches[args['pair_name_split'].join([f_name0, f_name1])] = np.concatenate(
            [mkpts0, mkpts1, mconfs[:, None]], -1
        )  # (N, 5)

        if pba is not None:
            pba.update.remote(1)
    return matches

@ray.remote(num_cpus=1, num_gpus=0.5, max_calls=1)  # release gpu after finishing
def match_worker_ray_wrapper(*args, **kwargs):
    return match_worker(*args, **kwargs)

def keypoint_worker(name_kpts, pba=None, verbose=True):
    """merge keypoints associated with one image.
    """
    keypoints = {}

    if verbose:
        name_kpts = tqdm(name_kpts) if pba is None else name_kpts
    else:
        assert pba is None

    for name, kpts in name_kpts:
        kpt2score = agg_groupby_2d(kpts[:, :2].astype(int), kpts[:, -1], agg="sum")
        kpt2id_score = {
            k: (i, v)
            for i, (k, v) in enumerate(
                sorted(kpt2score.items(), key=lambda kv: kv[1], reverse=True)
            )
        }
        keypoints[name] = kpt2id_score

        if pba is not None:
            pba.update.remote(1)
    return keypoints

@ray.remote(num_cpus=1)
def keypoints_worker_ray_wrapper(*args, **kwargs):
    return keypoint_worker(*args, **kwargs)


def update_matches(matches, keypoints, merge=False, pba=None, verbose=True, **kwargs):
    # convert match to indices
    ret_matches = {}

    if verbose:
        matches_items = tqdm(matches.items()) if pba is None else matches.items()
    else:
        assert pba is None
        matches_items = matches.items()

    for k, v in matches_items:
        mkpts0, mkpts1 = (
            map(tuple, v[:, :2].astype(int)),
            map(tuple, v[:, 2:4].astype(int)),
        )
        name0, name1 = k.split(kwargs['pair_name_split'])
        _kpts0, _kpts1 = keypoints[name0], keypoints[name1]

        mids = np.array(
            [
                [_kpts0[p0][0], _kpts1[p1][0]]
                for p0, p1 in zip(mkpts0, mkpts1)
                if p0 in _kpts0 and p1 in _kpts1
            ]
        )

        if len(mids) == 0:
            mids = np.empty((0, 2))

        def _merge_possible(name):  # only merge after dynamic nms (for now)
            return f'{name}_no-merge' not in keypoints

        if merge and _merge_possible(name0) and _merge_possible(name1):
            merge_ids = []
            mkpts0, mkpts1 = map(tuple, v[:, :2].astype(int)), map(tuple,  v[:, 2:4].astype(int))
            for p0, p1 in zip(mkpts0, mkpts1): 
                if (*p0, -2) in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[(*p1, -2)][0]])
                elif p0 in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[p0][0], _kpts1[(*p1, -2)][0]])
                elif (*p0, -2) in _kpts0 and p1 in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[p1][0]]) 
            merge_ids = np.array(merge_ids)

            if len(merge_ids) == 0:
                merge_ids = np.empty((0, 2))
                logger.warning("merge failed! No matches have been merged!")
            else:
                logger.info(f'merge successful! Merge {len(merge_ids)} matches')
            
            try:
                mids_multiview = np.concatenate([mids, merge_ids], axis=0)
            except ValueError:
                import ipdb; ipdb.set_trace()
        
            mids = np.unique(mids_multiview, axis=0)
        else:
            assert (
                len(mids) == v.shape[0]
            ), f"len mids: {len(mids)}, num matches: {v.shape[0]}"

        ret_matches[k] = mids.astype(int)  # (N,2)
        if pba is not None:
            pba.update.remote(1)

    return ret_matches

@ray.remote(num_cpus=1)
def update_matches_ray_wrapper(*args, **kwargs):
    return update_matches(*args, **kwargs)


def transform_keypoints(keypoints, pba=None, verbose=True):
    """assume keypoints sorted w.r.t. score"""
    ret_kpts = {}
    ret_scores = {}

    if verbose:
        keypoints_items = tqdm(keypoints.items()) if pba is None else keypoints.items()
    else:
        assert pba is None
        keypoints_items = keypoints.items()

    for k, v in keypoints_items:
        v = {_k: _v for _k, _v in v.items() if len(_k) == 2}
        kpts = np.array([list(kpt) for kpt in v.keys()]).astype(np.float32)
        scores = np.array([s[-1] for s in v.values()]).astype(np.float32)
        if len(kpts) == 0:
            logger.warning("corner-case n_kpts=0 exists!")
            kpts = np.empty((0,2))
        ret_kpts[k] = kpts
        ret_scores[k] = scores
        if pba is not None:
            pba.update.remote(1)
    return ret_kpts, ret_scores

@ray.remote(num_cpus=1)
def transform_keypoints_ray_wrapper(*args, **kwargs):
    return transform_keypoints(*args, **kwargs)
