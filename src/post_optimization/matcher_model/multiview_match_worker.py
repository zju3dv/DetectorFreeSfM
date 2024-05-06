import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import ray
from ray.actor import ActorHandle
from tqdm import tqdm
from omegaconf import OmegaConf
from loguru import logger

from src.MultiviewMatcher.MultiviewMatcher import MultiviewMatcher
from src.utils.data_io import dict_to_cuda
from ..data_construct import MatchingMultiviewData


def build_model(args, rewindow_size_factor=None, model_idx=None):
    cfg = OmegaConf.load(args['cfg_path'][model_idx] if model_idx is not None else args['cfg_path'][0])
    pl.seed_everything(args['seed'])
    matcher_cfg = cfg['model']['multiview_refinement']
    if rewindow_size_factor is not None:
        current_window_size = matcher_cfg['multiview_transform']['window_size']
        window_size_rescaled = ((current_window_size // 2) - 1 * rewindow_size_factor) * 2 + 1
        if window_size_rescaled < 7:
            window_size_rescaled = 7
        matcher_cfg['backbone']['s2dnet']["window_size"] = window_size_rescaled
        matcher_cfg['multiview_transform']["window_size"] = window_size_rescaled
        matcher_cfg['multiview_matching_test']["window_size"] = window_size_rescaled

        current_left_win_move_size = matcher_cfg['multiview_matching_test']["left_point_movement_window_size"]
        if current_left_win_move_size is not None:
            current_left_win_move_size_rescaled = ((current_left_win_move_size // 2) - 1 * rewindow_size_factor) * 2 + 1
            if current_left_win_move_size_rescaled < 3:
                current_left_win_move_size_rescaled = 3
            matcher_cfg['multiview_matching_test']["left_point_movement_window_size"] = current_left_win_move_size_rescaled

    matcher = MultiviewMatcher(config=matcher_cfg, test=True).eval()
    # load checkpoints
    model_path = args['weight_path'][model_idx] if model_idx is not None else args['weight_path'][0]
    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        logger.info(f"Load model from path: {args['weight_path']}")
        for k in list(state_dict.keys()):
            if "matcher." in k:
                state_dict[k.replace("matcher.", "")] = state_dict.pop(k)
            else:
                state_dict.pop(k)
        
        for k in list(state_dict.keys()):
            if 'loftr_coarse' in k:
                state_dict.pop(k)
            if 'loftr_fine' in k:
                state_dict[k.replace("loftr_fine", "fine_transformer")] = state_dict.pop(k)
        matcher.load_state_dict(state_dict, strict=True)
    else:
        logger.warning(f"Model path is None!")
    return matcher

@torch.no_grad()
def extract_results(
    data,
    matcher=None,
):
    # 1. inference
    matcher(data)
    # 2. extract match and refined poses
    reference_points_refined = data['query_points_refined'].cpu().numpy() # 1 * n_track * 2
    reference_img_ids = data['query_img_ids'].cpu().numpy() # 1 * n_track
    reference_pt2D_idxs = data['query_pt2d_idxs'].cpu().numpy() # 1 * n_track
    ref_movable_mask = data['query_movable_mask'].cpu().numpy()

    query_points_refined = data['reference_points_refined'][-1].cpu().numpy() # 1 * n_view-1 * n_track * 2
    query_img_ids = data['reference_img_ids'].cpu().numpy() # 1 * n_view-1 * n_track
    query_pt2D_idxs = data['reference_pt2d_idxs'].cpu().numpy() # 1 * n_view-1 * n_track
    mask = data['track_valid_mask'].cpu().numpy() # # 1 * n_view-1 * n_track
    assert query_points_refined.shape[0] == 1
    # M * 2, M, M
    if "time" in data:
        time = data['time']
    else:
        time = None
    return [query_points_refined[mask], query_img_ids[mask], query_pt2D_idxs[mask]], \
        [reference_points_refined[ref_movable_mask], reference_img_ids[ref_movable_mask], reference_pt2D_idxs[ref_movable_mask]], time


class UpdatedQueryPts:
    def __init__(self, colmap_images) -> None:
        self.updated_dict = {img_id : {} for img_id in colmap_images.keys()}
    
    def find_movable_and_update(self, data):
        # Determine moveable mask:
        query_moveable_mask = []
        query_node = []
        for idx, (query_img_id, query_pt2d_idx) in enumerate(zip(data['query_img_ids'][0].numpy(), data['query_pt2d_idxs'][0])):
            if query_pt2d_idx in self.updated_dict[query_img_id]:
                # Already moved:
                query_moveable_mask.append(False)
                query_node.append(
                    self.updated_dict[query_img_id][query_pt2d_idx]
                ) # (2)
            else:
                # Not move yet
                query_moveable_mask.append(True)
                query_node.append(data['query_points'][0, idx])
        data.update({'query_points': torch.stack(query_node).to(torch.float32)[None], "query_movable_mask": torch.from_numpy(np.stack(query_moveable_mask))[None]}) # 1 * M

    def update_query_pts(self, kpts_refined, image_ids, pt2D_idxs):
        for kpt2D, image_id, pt2D_idx in zip(kpts_refined, image_ids, pt2D_idxs):
            self.updated_dict[image_id][pt2D_idx] = kpt2D

@torch.no_grad()
def matchWorker(colmap_dataset, matcher, subset_track_idxs=None, visualize=False, visualize_dir=None, pba: ActorHandle = None, dataset_cfgs=None, verbose=True):
    """extract matches from part of the possible image pair permutations"""

    multiview_matching_dataset = MatchingMultiviewData(colmap_dataset, dataset_cfgs, worker_split_idxs=subset_track_idxs)
    dataloader = DataLoader(multiview_matching_dataset, num_workers=4, pin_memory=True)
    
    matcher.cuda()
    results_list = []
    running_time = []
    query_updated_buffer = UpdatedQueryPts(multiview_matching_dataset.colmap_images)
    if not verbose:
        assert pba is None
    
    for data in tqdm(dataloader, disable=not verbose):
        query_updated_buffer.find_movable_and_update(data)
        data_c = dict_to_cuda(data)

        [query_points_refined, query_img_ids, query_pt2D_idxs], [ref_points_refined, ref_img_ids, ref_pt2D_idxs], time = extract_results(
            data_c, matcher=matcher
        )
        query_updated_buffer.update_query_pts(ref_points_refined, ref_img_ids, ref_pt2D_idxs)

        # 3. extract results
        points2D_refined = np.concatenate([query_points_refined, ref_points_refined], axis=0)
        img_ids = np.concatenate([query_img_ids, ref_img_ids], axis=0)
        pt2D_idxs = np.concatenate([query_pt2D_idxs, ref_pt2D_idxs], axis=0)
        results_list.append(np.concatenate([points2D_refined, img_ids[:, None], pt2D_idxs[:, None]], axis=1)) # M * 4

        if time is not None:
            running_time.append(time)

    if pba is not None:
        pba.update.remote(1)

    if len(running_time) != 0:
        logger.warning(f"Current each part running time evaluation does not support multiple workers.")
        time_each_part = np.array(running_time) # N * m, m is number of parts
        print(f"Mean time of each part is: {np.mean(time_each_part, axis=0)}")
        print(f"Total time of each part is: {np.sum(time_each_part, axis=0)}")
    return results_list

@ray.remote(num_cpus=1, num_gpus=0.25, max_calls=1)  # release gpu after finishing
def matchWorker_ray_wrapper(*args, **kwargs):
    return matchWorker(*args, **kwargs)