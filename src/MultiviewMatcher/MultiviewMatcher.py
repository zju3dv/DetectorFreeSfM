from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from time import time

from src.utils.profiler import PassThroughProfiler

from .backbone import (
    build_backbone,
    _extract_backbone_feats,
)
from .matcher_module import LocalFeatureTransformer, FinePreprocess
from .utils.fine_matching import FineMatching

class MultiviewMatcher(nn.Module):
    def __init__(self, config, profiler=None, debug=False, test=False, plotting_vis=None, test_speed=False):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.debug = debug
        self.plotting_vis = plotting_vis
        self.test_speed = test_speed

        self.n_steps = self.config["n_matching_steps"]
        self.enable_multiview_scale_align = self.config["enable_multiview_scale_align"]
        assert self.n_steps != 0
        # Modules
        self.backbone = build_backbone(self.config["backbone"])

        self.fine_preprocess = FinePreprocess(
            self.config["multiview_transform"],
        )

        self.fine_transformer = LocalFeatureTransformer(self.config["multiview_transform"])
        self.fine_matching = FineMatching(
            self.config["multiview_matching_train"]
            if not test
            else self.config["multiview_matching_test"]
        )

        self.backbone_pretrained = self.config["backbone"]["pretrained"]
        if self.backbone_pretrained is not None:
            logger.info(f"Load pretrained backbone from {self.backbone_pretrained}")
            ckpt = torch.load(self.backbone_pretrained, "cpu")["state_dict"]
            for k in list(ckpt.keys()):
                if "backbone" in k:
                    newk = k[k.find("backbone") + len("backbone") + 1 :]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.backbone.load_state_dict(ckpt, strict=True)

            if self.config["backbone"]["pretrained_fix"]:
                for param in self.backbone.parameters():
                    param.requires_grad = False

    def forward(self, data, chunk_track=1000, chunk_backbone_img=True):
        if isinstance(data["images"], torch.Tensor):
            B, N_img, c, h, w = data["images"].shape
            device = data["images"].device
        else:
            # NOTE: no img padding scenario
            N_img = len(data["images"])
            B, c, h, w = data["images"][0].shape
            device = data["images"][0].device

        fine_scale = torch.full(
            (B, N_img, 2),
            self.config["backbone"]["resolution"][-1],
            device=device,
        )  # resize to fine, B * n_view * 2
        scales = (
            fine_scale * data["scales"][:, :, [1, 0]]
            if "scales" in data
            else fine_scale
        )  # B * n_view * 2

        scales_relative = (
            data["scales_relative"].clamp(
                1 / (self.fine_preprocess.W // 2), (self.fine_preprocess.W // 2)
            )
            if "scales_relative" in data
            else None
        )

        for step in range(self.n_steps):
            current_reference_location = (
                data["reference_points_coarse"]
                if step == 0
                else data["reference_points_refined"][-1].clone().detach()
            )  # B * n_view-1 * n_track
            all_sample_points = torch.cat(
                [data["query_points"][:, None], current_reference_location], dim=1
            )
            img_idxs = torch.cat(
                [data["query_img_idxs"][:, None], data["reference_img_idxs"]], dim=1
            )  # B * n_view * n_track

            # NOTE: currently only support bs==1
            B, n_view, n_track = img_idxs.shape
            all_sample_points_scales = ((scales.view(-1, 2))[img_idxs.view(-1)]).view(
                B, n_view, n_track, 2
            )
            all_sample_points /= all_sample_points_scales  # B * n_view * n_track * 2

            scales_relative_current_step = (
                scales_relative ** ((step + 1) / self.n_steps)
                if scales_relative is not None
                else None
            )

            if self.training or not chunk_backbone_img:
                chunk_view_list = [n_view]
                num_track_list = [n_track]
            else:
                max_view = 16 # Hard code
                max_view_tracks = max_view * chunk_track
                unique_keys, counts = torch.unique(data["track_valid_mask"].sum(-2).max(0)[0], sorted=True, return_counts=True)
                unique_keys, counts = unique_keys.flip(0), counts.flip(0)
                chunk_view_list, num_track_list = [], []
                i = 0
                while i < unique_keys.shape[0]:
                    chunk_view_list.append(unique_keys[i] + 1)
                    valid_view = unique_keys[i] + 1
                    if valid_view * counts[i] <= max_view_tracks:
                        num_track = counts[i]
                        i += 1
                    else:
                        counts[i] -= max_view_tracks // valid_view
                        num_track = max_view_tracks // valid_view
                    num_track_list.append(num_track)

            if self.training or not chunk_backbone_img:
                assert isinstance(data["images"], torch.Tensor)
                with self.profiler.record_function("MultiviewMatcher/fine-process"):
                    image_patches, sparse = self.fine_preprocess(
                        data["images"],
                        all_sample_points,
                        img_idxs,
                        data,
                        scales_relative_current_step,
                    )  # B * n_view * n_track * WW * C
                    
                    if sparse:
                        W = int(image_patches.shape[-2] ** 0.5)
                        image_patches = rearrange(
                            image_patches, "b n t (h w) c -> (b n t) c h w", h=W, w=W
                        )

                # Extract features:
                with self.profiler.record_function("MultiviewMatcher/backbone"):
                    # Images: B * N_img * 1 * H * W
                    features = self.backbone(
                        image_patches,
                        rearrange(scales_relative_current_step, "b n t -> (b n t)") if self.enable_multiview_scale_align and not self.training else None,
                        sparse,
                        rearrange(all_sample_points, "b n t c -> (b n t) c"),
                        rearrange(img_idxs, "b n t -> (b n t)"),
                    )  # [(B * n_view * n_track) * C * H * W]
                    features = rearrange(
                        features[-1],
                        "(b n t) c h w -> b t n (h w) c",
                        t=n_track,
                        n=n_view,
                    )

                features = _extract_backbone_feats(
                    features, self.config["backbone"]
                )
                features_crop_reference, features_crop_query = (
                    features[:, :, 0],
                    features[:, :, 1:],
                )

            elif chunk_backbone_img:
                features_crops, feature_belonging_num, image_patches_all = [], [], []
                original_order_indexs = torch.full(
                    (B * n_track * n_view,), -1, device=device, dtype=torch.long
                )

                if self.test_speed:
                    crop_img_time = 0
                    feature_extract_time = 0

                # NOTE: To aviod OOM at test time.
                for img_idx in range(N_img):
                    crop_size = self.config["multiview_transform"]['crop_size']
                    full_feature_map_patches_num = 100000 * h * w / (crop_size)**2 # To enable sparse feature patch crop

                    if self.test_speed:
                        torch.cuda.synchronize()
                        crop_img_t0 = time()

                    with self.profiler.record_function("MultiviewMatcher/fine-process"):
                        (
                            image_patches,
                            image_patch_belonging_mask,
                            sparse,
                        ) = self.fine_preprocess.forward_chunked(
                            data["images"][:, img_idx]
                            if isinstance(data["images"], torch.Tensor)
                            else data["images"][img_idx],
                            img_idx,
                            all_sample_points,
                            img_idxs,
                            data,
                            scales_relative_current_step,
                            full_feature_map_patches_num,
                        )  # M * WW * C

                        if image_patches.shape[0] == 0 or image_patch_belonging_mask.sum() == 0:
                            continue

                        if sparse:
                            W = int(image_patches.shape[-2] ** 0.5)
                            image_patches = rearrange(
                                image_patches, "m (h w) c -> m c h w", h=W, w=W
                            )  # NOTE: M is (b n t) order

                    if self.test_speed:
                        torch.cuda.synchronize()
                        feat_extract_t0 = time()
                        crop_img_time += (feat_extract_t0 - crop_img_t0)

                    # Extract features:
                    with self.profiler.record_function("MultiviewMatcher/backbone"):
                        # Images: B * N_img * 1 * H * W
                        features = self.backbone(
                            image_patches,
                            rearrange(scales_relative_current_step, "b n t -> (b n t)")[
                                image_patch_belonging_mask
                            ] if self.enable_multiview_scale_align else None,
                            sparse,
                            rearrange(all_sample_points, "b n t c -> (b n t) c")[
                                image_patch_belonging_mask
                            ], 
                            None
                        )  # [M * C * H * W]
                        features = rearrange(features[-1], "m c h w -> m (h w) c")
                        image_patches = rearrange(image_patches, "m c h w -> m h w c") if self.plotting_vis is not None else None

                    if self.test_speed:
                        torch.cuda.synchronize()
                        feat_extract_t1 = time()
                        feature_extract_time += (feat_extract_t1 - feat_extract_t0)

                    features = _extract_backbone_feats(
                        features, self.config["backbone"]
                    )

                    original_order_indexs[image_patch_belonging_mask] = torch.arange(
                        sum(feature_belonging_num),
                        sum(feature_belonging_num) + features.shape[0],
                        device=device,
                        dtype=torch.long,
                    )
                    features_crops.append(features)
                    image_patches_all.append(image_patches) if self.plotting_vis is not None else None
                    feature_belonging_num.append(features.shape[0])
                
                del features
                features_crops = torch.cat(features_crops, dim=0)[
                    original_order_indexs
                ]  # (B * n_view * n_track) * WW * C
                features_crops = rearrange(
                    features_crops, "(b n t) w c -> b t n w c", n=n_view, t=n_track
                )  # B * n_track * n_view * WW * C
                image_patches = rearrange(
                    torch.cat(image_patches_all, dim=0)[original_order_indexs],
                    "(b n t) h w c -> b t n h w c",
                    n=n_view,
                    t=n_track,
                ) if self.plotting_vis is not None else None
                features_crop_reference, features_crop_query = (
                    features_crops[:, :, 0],
                    features_crops[:, :, 1:],
                )

            (
                query_points_refined_chunked,
                reference_points_refined_chunked,
                std_chunked, transformed_features, heatmap_ref_all
            ) = ([], [], [], [], [])

            if self.test_speed:
                transform_feat_time = 0
                matching_time = 0
            i = 0
            for it in range(len(num_track_list)):
                num_track = num_track_list[it]
                chunk_view = chunk_view_list[it]
                data.update(
                    {
                        "scales_origin_to_fine_reference": all_sample_points_scales[
                            :, 1:chunk_view, i : (i + num_track)
                        ],
                        "scales_origin_to_fine_query": all_sample_points_scales[
                            :, 0, i : (i + num_track)
                        ],
                        "scales_relative": scales_relative_current_step[
                            :, :chunk_view, i : (i + num_track)
                        ],
                        "scales_relative_reference": scales_relative_current_step[
                            :, 1:chunk_view, i : (i + num_track)
                        ],
                        "view_point_vector_chunk": data["view_point_vector"][
                            :, :chunk_view, i : (i + num_track)
                        ],
                    }
                )  # B * (n_view-1) * n_track * 2

                features_crop_reference_super_patch = features_crop_reference[
                    :, i : (i + num_track)
                ]

                # Perform multiview transformer:
                with self.profiler.record_function("MultiviewMatcher/transformer"):
                    if self.test_speed:
                        torch.cuda.synchronize()
                        transform_t0 = time()
                    if self.config["multiview_transform"]["enable"]:
                        (
                            features_crop_reference_transformed,
                            features_crop_query_transformed,
                        ) = self.fine_transformer(
                            features_crop_reference_super_patch,
                            features_crop_query[:, i : (i + num_track), :chunk_view-1],
                            data,
                            query_mask=data["track_valid_mask"].transpose(1, 2)[
                                :, i : (i + num_track), :chunk_view-1
                            ]
                            if "track_valid_mask" in data
                            else None,
                        )
                    else:
                        features_crop_reference_transformed = features_crop_reference_super_patch
                        features_crop_query_transformed = features_crop_query[:, i : (i + num_track), :chunk_view-1]

                    if self.test_speed:
                        torch.cuda.synchronize()
                        transform_t1 = time()
                        transform_feat_time += (transform_t1 - transform_t0)

                # Perform multiview refinement:
                with self.profiler.record_function("MultiviewMatcher/matching"):
                    if self.test_speed:
                        torch.cuda.synchronize()
                        matching_t0 = time()

                    (
                        query_points_refined,
                        reference_points_refined,
                        std,
                        heatmap_all
                    ) = self.fine_matching(
                        features_crop_reference_transformed,
                        features_crop_query_transformed,
                        data["query_points"][:, i : (i + num_track)],
                        current_reference_location.transpose(1, 2)[
                            :, i : (i + num_track), :chunk_view-1
                        ],
                        data,
                        scales_relative_reference=scales_relative_current_step.transpose(1,2)[:, i : (i + num_track), 1:chunk_view] if self.enable_multiview_scale_align else None,
                        track_mask=data["track_valid_mask"].transpose(1, 2)[
                            :, i : (i + num_track), :chunk_view-1
                        ],
                        query_movable_mask=data["query_movable_mask"][
                            :, i : (i + num_track)
                        ] if 'query_movable_mask' in data else None,
                        keypoint_relocalized_offset=data['keypoint_relocalized_offset'][:, i : (i + num_track)] if 'keypoint_relocalized_offset' in data else None,
                        return_all_ref_heatmap=True if self.plotting_vis is 'heatmap_patch' else False,
                    )  # B * n_view-1 * n_track * 2
                    query_points_refined_chunked.append(query_points_refined)
                    reference_points_refined_chunked.append(F.pad(reference_points_refined, pad=(0,0,0,0,0,n_view-chunk_view), value=0))
                    std_chunked.append(F.pad(std, pad=(0,0,0,n_view-chunk_view), value=0))
                    transformed_features.append(torch.cat([features_crop_reference_transformed[:, :, None], F.pad(features_crop_query_transformed, pad=(0,0,0,0,0,n_view-chunk_view), value=0)], dim=2)) if self.plotting_vis is 'feature_patch' else None
                    heatmap_ref_all.append(F.pad(heatmap_all, pad=(0,0,0,0,0,0,1,n_view-chunk_view), value=0)) if self.plotting_vis is 'heatmap_patch' else None

                    if self.test_speed:
                        torch.cuda.synchronize()
                        matching_t1 = time()
                        matching_time += (matching_t1 - matching_t0)
                
                i += num_track

            query_points_refined = torch.cat(query_points_refined_chunked, dim=1)
            reference_points_refined = torch.cat(
                reference_points_refined_chunked, dim=2
            )
            std = torch.cat(std_chunked, dim=2) if std is not None else None

            data["query_points_refined"] = query_points_refined
            if "reference_points_refined" in data:
                data["reference_points_refined"].append(
                    reference_points_refined
                )  # B * n_view-1 * n_track * 2
                data["std"].append(std)  # B * n_view-1 * n_track
            else:
                data["reference_points_refined"] = [reference_points_refined]
                data["std"] = [std]
            
            if self.test_speed:
                data.update({'time': [crop_img_time, feature_extract_time, transform_feat_time, matching_time]})