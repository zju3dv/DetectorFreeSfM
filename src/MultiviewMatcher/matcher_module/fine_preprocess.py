import torch
import torch.nn as nn
from einops.einops import rearrange
from roi_align.roi_align import RoIAlign
    
class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.crop_size = config['crop_size']
        self.W = self.config['window_size']  # window size of fine-level (cf_res[-1])
        assert self.crop_size >= self.W
        self.enable_rescaled_crop = config['enable_rescaled_crop']

        self.sparse = config['sparse']
        
        self.roi_align_custom = RoIAlign(self.crop_size, self.crop_size, transform_fpcoor=False)
            
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            try:
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # pyre-ignore
                    nn.init.constant_(m.bias, 0)
            except:
                pass
        
    def forward_chunked(self, feature, feature_idx, sample_points, img_idxs, data, scales_relative=None, full_feature_map_patches_num=0):
        data.update({'W': self.W})

        B, n_view, n_track, C = sample_points.shape

        features = feature
        all_sample_points_xy = sample_points.reshape(-1, C)
        assert B == 1, f"Currently only support batchsize == 1"

        points_corresponding_img_idxs = img_idxs.reshape(-1) # (B * n_view * n_track)

        feature_belonging_mask = points_corresponding_img_idxs == feature_idx
        feature_belonging_mask_sum = feature_belonging_mask.sum()
        if feature_belonging_mask_sum <= full_feature_map_patches_num: # sparse feature map
            sparse = True
            corresponding_idxs = torch.zeros((feature_belonging_mask_sum,), device=feature_belonging_mask.device).long()

            if self.enable_rescaled_crop:
                raise ValueError
            else:
                scales_relative = None
            features_crop = self._forward_single_scale(features, corresponding_idxs, all_sample_points_xy[feature_belonging_mask], data, box_scales=scales_relative) # M * WW * C
        else: # full feature map
            sparse = False
            features_crop = features
        return features_crop, feature_belonging_mask, sparse # M * WW * C

    def forward(self, features, sample_points, img_idxs, data, scales_relative=None):
        data.update({'W': self.W})
        if isinstance(features[0], list):
            raise NotImplementedError

        B, n_view, n_track, C = sample_points.shape

        # From (B * n_view) * n_track * 2 to M * 2
        # Reformat:
        features = rearrange(features, 'b n c h w -> (b n) c h w')
        all_sample_points_xy = sample_points.reshape(-1, C)
        assert B == 1, f"Currently only support batchsize == 1"

        if self.sparse:
            points_corresponding_img_idxs = img_idxs.reshape(-1) # (B * n_view * n_track)
            if self.enable_rescaled_crop:
                raise ValueError
            else:
                scales_relative = None
            features_crop = self._forward_single_scale(features, points_corresponding_img_idxs, all_sample_points_xy, data, box_scales=scales_relative) # M * WW * C
            # De-format:
            features_crop = rearrange(features_crop, "(b n t) w c -> b n t w c", b=B, n=n_view, t=n_track) # B * n_track * n_view * WW * C
        else:
            features_crop = features
        return features_crop, self.sparse
    
    def _forward_single_scale(self, features, b_ids, points_xy, data, box_scales=None):
        features_crop = self._extract_local_patches(features, points_xy, b_ids, scales=box_scales)
        features_crop = rearrange(features_crop, 'n c w h -> n (w h) c') # B' * WW * C
        return features_crop
    
    def _extract_local_patches(
            self,
            features,  # (N, C, H, W) # 1/2 feature map
            keypoints,  # [L, 2] # keypoint in feature map scale
            bids,  # [L, 1]
            scales=None,
            aligned=False,
        ):
        bids = bids.unsqueeze(-1) if len(bids.shape) == 1 else bids
        redius = self.crop_size // 2
        if scales is not None:
            redius *= scales[:, None]
        boxes = torch.cat([bids, keypoints - redius, keypoints + redius], dim=-1).to(torch.float32) # L*5
        unfold_features = self.roi_align_custom(features, boxes[:,1:], bids.to(torch.int32))
        return unfold_features