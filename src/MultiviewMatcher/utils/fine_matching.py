from einops.einops import rearrange, repeat
from loguru import logger

import torch
import torch.nn as nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from time import time

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm
    NOTE: use a separate class for d2d (sprase/dense flow) ?
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fine_detector = config['detector']
        self._type = config['s2d']['type']
        
        self.obtain_offset_method = config['s2d']['obtain_offset_method']

        self.left_point_movement = config['left_point_movement_window_size']
        self.best_left_strategy = config['best_left_strategy']
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)

    def forward(self, features_reference_crop, features_query_crop, query_points_coarse, reference_points_coarse, data, scales_relative_reference=None, track_mask=None, query_movable_mask=None, keypoint_relocalized_offset=None, return_all_ref_heatmap=False):
        """
        Args:
            features_reference_crop: B * n_track * WW * C or B * n_track * WW * C (single view matching scenario)
            features_query_crop: B * n_track * (n_view - 1) * WW * C
            query_points_coarse: B * N_track * 2
            reference_points_coarse: B * n_track * (n_view - 1) * 2
            scales_relative_reference: B * n_track * (n_view -1)
            mask: B * n_track * (n_view - 1)
            query_movable_mask: B * N_track
            query_movable_mask: B * N_track
            keypoint_relocalized_offset: B * N_track, use keypoint detector to relocalize feature tracks
            data (dict)
        Update:
            data (dict):{
                "query_points_refined": [B, n_track, 2]
                "reference_points_refined": [B, n_view-1, n_track, 2]
                'std': [B, n_view-1, n_track]}
        """
        self.device = features_reference_crop.device
        B, n_track, n_query, WW, C = features_query_crop.shape
        self.W, self.WW, self.C = data['W'], WW, C

        # Re-format:
        features_reference_crop = rearrange(features_reference_crop, "b t w c -> (b t) w c")

        features_query_crop = rearrange(features_query_crop, 'b t n w c -> (b t) n w c') # [M, n_view-1, WW, C]
        track_mask = rearrange(track_mask, 'b t n -> (b t) n') if track_mask is not None else None
        query_movable_mask = rearrange(query_movable_mask, 'b t -> (b t)') if query_movable_mask is not None else None
        keypoint_relocalized_offset = rearrange(keypoint_relocalized_offset, 'b t c -> (b t) c') if keypoint_relocalized_offset is not None else None

        features_reference_selected = self.select_left_point(features_reference_crop, data) # [M, C] or [M, n_view-1, C] (single view matching scenario)
        
        coords_normed, heatmap_all_ref, std = self.predict_s2d(features_reference_selected, features_query_crop, data) # [M, n_view-1, 2], [M, n_view-1]

        if self.left_point_movement is not None:
            query_offset_norm, coords_normed, std, heatmap = self._obtain_left_normalized_offset(coords_normed, std, heatmap_all_ref, track_mask=track_mask, movable_mask=query_movable_mask, keypoint_relocalized_offset=keypoint_relocalized_offset)
        else:
            query_offset_norm = None
            heatmap = heatmap_all_ref

        # De-format:
        coords_normed = rearrange(coords_normed, '(b t) n c -> b t n c', b=B)
        query_offset_norm = rearrange(query_offset_norm, '(b t) c -> b t c', b=B) if query_offset_norm is not None else None
        std = rearrange(std, '(b t) n -> b n t', b=B) if std is not None else None
        heatmap = rearrange(heatmap, '(b t) n w0 w1 -> b n t w0 w1', b=B) if heatmap is not None else None

        data.update({'fine_local_heatmap_pred': heatmap})

        if query_offset_norm is not None:
            query_points_refined = self.build_moved_query(query_offset_norm, query_points_coarse, data)
        else:
            query_points_refined = query_points_coarse

        # compute absolute kpt coords: B * n_track * n_view-1 * 2
        referece_points_refined = self.build_mkpts(coords_normed, reference_points_coarse, data, scales_relative_reference)

        if return_all_ref_heatmap:
            heatmap_all_ref = rearrange(heatmap_all_ref, '(b t) m n w0 w1 -> b t n m w0 w1', b=B) if heatmap_all_ref is not None else None
            return query_points_refined, referece_points_refined.transpose(1,2), std, heatmap_all_ref,  # B * n_view-1 * n_track * 2, B * n_view-1 * n_track

        else:
            return query_points_refined, referece_points_refined.transpose(1,2), std, None,  # B * n_view-1 * n_track * 2, B * n_view-1 * n_track
        
    def select_left_point(self, feat_f0, data):
        L = feat_f0.shape[-2] # [M, WW, C] or [M, n_view-1, WW, C]
        W = int(L ** .5)
        assert L % 2 == 1
        
        left_point_movement = self.left_point_movement
        if left_point_movement is None:
            feat_f0_picked = feat_f0[..., L//2, :]
        else:
            assert not self.training
            assert left_point_movement > 0 and left_point_movement % 2 == 1 and left_point_movement <= W
            if len(feat_f0.shape) == 3:
                feat_f0 = rearrange(feat_f0, 'm (h w) c -> m h w c', h=W)
            else:
                raise NotImplementedError
            
            feat_f0_picked = feat_f0[..., (W//2 - left_point_movement//2):(W//2 + left_point_movement//2 + 1), (W//2 - left_point_movement//2):(W//2 + left_point_movement//2 + 1), :]
            feat_f0_picked = feat_f0_picked.flatten(-3, -2)

        return feat_f0_picked
        
    def predict_s2d(self, feat_f0_picked, feat_f1, data):
        # compute normalized coords ([-1, 1]) of right patches
        if self._type == 'heatmap':
            coords_normed = self._s2d_heatmap(feat_f0_picked, feat_f1, data)
        else:
            raise NotImplementedError()
        return coords_normed
    
    def _obtain_left_normalized_offset(self, coord_normed_tentative, std, heatmap, track_mask=None, movable_mask=None, keypoint_relocalized_offset=None):
        """
        Args:
            coord_normed_tentative: M * L * N * 2
            std: M * L * N
            query_mask: M * N, used to mask invalid (padded) nodes in track
            movable_mask: M, used to determine whether movable
            keypoint_relocalized_offset: M * 2, used to determine whether movable
            heatmap: M * L * N * W * W
        """
        t0 = time()
        L = coord_normed_tentative.shape[1]
        left_mv_win_size = int(L**.5)
        if track_mask is None:
            logger.warning(f'No mask provided')
            track_mask = torch.ones_like(std, dtype=torch.bool)
        else:
            track_mask = repeat(track_mask, 'm n -> m l n', l=L)
        if self.best_left_strategy == "smallest_mean_std":
            score = masked_mean(std, track_mask, dim=-1) # M * L, std smaller is better
            # score = torch.softmax(score * -1, dim=-1) * -1
        elif self.best_left_strategy == 'max_conf/std':
            # Take local maxium and distribution into consideration.
            conf, _ = heatmap.flatten(-2, -1).max(dim=-1) # M * L * N
            score = masked_mean(conf * 2 / (std + 1e-6), track_mask, dim=-1) # larger is better, M * L
            score *= -1 # smaller is better
        else:
            raise NotImplementedError
        best_value, best_index = torch.min(score, dim=-1)

        # Use detected keypoints to relocalize:
        if keypoint_relocalized_offset is not None:
            keypoint_relocalized_mask = keypoint_relocalized_offset.sum(-1) != 0
            keypoint_relocalized_index = ((keypoint_relocalized_offset / (left_mv_win_size // 2) + 1) / 2) * (left_mv_win_size - 1) # convert center coordinate to top-left
            keypoint_relocalized_index = keypoint_relocalized_index.round().clamp(0, left_mv_win_size-1).long()
            best_index = torch.where(keypoint_relocalized_mask, keypoint_relocalized_index[:, 1] * left_mv_win_size + keypoint_relocalized_index[:, 0], best_index)

        # best_index[~movable_mask] = L // 2
        best_index = torch.where(movable_mask, best_index, L//2)

        left_offset_x = best_index % left_mv_win_size
        left_offset_y = best_index // left_mv_win_size
        left_offset = torch.stack([left_offset_x, left_offset_y], dim=-1)

        left_offset_norm = (left_offset / (left_mv_win_size - 1)) * 2 - 1 # NOTE: in the left window size coordinate

        # Select offset corresponding track:
        m_ids = torch.arange(coord_normed_tentative.shape[0], device=coord_normed_tentative.device)
        coord_normed_selected, std_selected, heatmap_selected = map(lambda x: x[m_ids, best_index], [coord_normed_tentative, std, heatmap])

        return left_offset_norm, coord_normed_selected, std_selected, heatmap_selected

    def _obtain_normalized_offset(self, heatmap):
        """
        Args:
            heatmap: B * L * N * W * W or B * N * W * W
        """
        std = None
        M, N, W = heatmap.shape[:3]
        obtain_offset_method = self.obtain_offset_method
        if obtain_offset_method == 'argsoftmax':
            coords_normalized, std = argsoftmax(heatmap)
        else:
            raise NotImplementedError
        return coords_normalized, std
        
    def _s2d_heatmap(self, feat_ref_picked, feat_query, data):
        W, WW, C = self.W, self.WW, self.C
        M, n_query = feat_query.shape[:2]
        
        if len(feat_ref_picked.shape) == 2:
            # Multiview matching scenario
            sim_matrix = torch.einsum('mc,mnrc->mnr', feat_ref_picked, feat_query)
        elif len(feat_ref_picked.shape) == 3 and self.left_point_movement != 0:
            # Multiview matching and anchor point movement scenario
            sim_matrix = torch.einsum('mlc,mnrc->mlnr', feat_ref_picked, feat_query)
        else:
            raise NotImplementedError

        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=-1) # [M, n_query, W, W] or [M, L, n_query, W, W]
        if len(heatmap.shape) == 3:
            heatmap = heatmap.view(M, n_query, W, W)
        elif len(heatmap.shape) == 4:
            L = heatmap.shape[1] # n tentative movement
            heatmap = heatmap.view(M, L, n_query, W, W)
        else:
            raise NotImplementedError

        coords_normalized, std = self._obtain_normalized_offset(heatmap)
        return coords_normalized, heatmap, std
    
    def build_moved_query(self, coords_normed, query_points_coarse, data):
        """
        Args:
            coords_normed: B * n_tracks * 2
            query_points_coarse: B * n_tracks * 2 (in original scale)
        Return:
            query_points_refined: B * n_tracks * 2
        """
        W, scales_origin_to_fine = self.left_point_movement, data['scales_origin_to_fine_query']
        window_size = (W // 2)
        query_points_refined = query_points_coarse + (coords_normed * window_size * scales_origin_to_fine)
        return query_points_refined
    
    def build_mkpts(self, coords_normed, reference_points_coarse, data, scales_relative_reference=None):
        """
        Args:
            coords_normed: B * n_tracks * n_view-1 * 2
            reference_points_coarse: B * n_tracks * n_view-1 * 2 (in original scale)
            scales_relative: B * n_track * n_view-1, only available when scale align is enabled
        Return:
            reference_points_refined: B * n_tracks * n_view-1 * 2
        """
        # scale_origin_to_fine: B * n_view * (n_track-1) * 2
        W, WW, C, scales_origin_to_fine = self.W, self.WW, self.C, data['scales_origin_to_fine_reference']
        
        # mkpts1_f
        if scales_relative_reference is not None:
            window_size = (W // 2) * scales_relative_reference[..., None]
        else:
            window_size = (W // 2)
        referece_points_refined = reference_points_coarse + (coords_normed * window_size * scales_origin_to_fine.transpose(1,2))
        return referece_points_refined

def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)

def argsoftmax(heatmap):
    """
    Args:
        heatmap: M, L, N, W, W or B, N, W, W
    """
    flattened = False
    if len(heatmap.shape) == 4:
        M, N, W = heatmap.shape[:3]
    elif len(heatmap.shape) == 5:
        M, L, N_, W = heatmap.shape[:4]
        N = L * N_
        heatmap = heatmap.flatten(1,2)
        flattened = True

    WW = W ** 2
    # compute coordinates from heatmap
    coords_normalized = dsnt.spatial_expectation2d(heatmap, True)  # [M, n_query, 2]
    grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, 1, -1, 2)  # [1, 1, WW, 2]

    # compute std over <x, y>
    var = torch.sum(grid_normalized**2 * heatmap.view(M, N, WW, 1), dim=-2) - coords_normalized**2  # [M, n_query, 2]
    std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M, n_query]  clamp needed for numerical stability

    if flattened:
        # left point movement scenario
        coords_normalized, std = map(lambda x: rearrange(x, 'm (l n) c -> m l n c', l=L), [coords_normalized, std[..., None]])
        std = std[..., 0]
    return coords_normalized, std