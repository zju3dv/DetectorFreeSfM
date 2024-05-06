import torch
from torch.nn import functional as F
from kornia.utils import create_meshgrid
from einops import repeat
from loguru import logger
from .geometry import warp_source_views


@torch.no_grad()
def mask_grid_pts_at_padded_regions(grid_pt, mask, scale=8):
    """
    For megadepth dataset, zero-padding exists in images
    mask: B * N * H * W (in resized solution)
    grid_pt: B * N * n_grid * 2
    """
    if scale != 1:
        mask = F.interpolate(
            mask, scale_factor=1 / scale, mode="nearest", recompute_scale_factor=False
        )
    mask = repeat(mask, "b n h w -> b n (h w) c", c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt, mask


@torch.no_grad()
def dense_grid_spv(data, config):
    """
    Input:
    "images": B * N * 1 * H * W
    Output:
    "points": b_id, n_id, x, y
    "tracks": N_track * track_length, indices of points, where the first is the reference view

    Then input to fineprocess:
    N_track: track_length * WW * C features,
    attention is performed:
    N_track: 1 * WW * C <-> (track_length - 1) * WW * C
    """
    grid_scale = config["grid_scale"]  # 8
    fine_scale = config["fine_scale"] # 2
    fine_window_size = config['window_size']
    sample_n_tracks_per_instance = config["sample_n_tracks_per_instance"]  # 400
    track_length_tolerance = config["track_length_tolerance"]  # 0
    reference_points_pertubation = config["reference_points_pertubation"]  # 0
    scale_pertubation = config["scale_pertubation"]

    device = data["images"].device

    # Generate grid coordinates:
    B, N, _, H, W = data["images"].shape
    scale = (
        grid_scale * data["scales"][..., None, [1, 0]]
        if "scales" in data
        else grid_scale
    )  # B * N_view * 1 * 2
    h_coarse, w_coarse = map(lambda x: x // grid_scale, [H, W])
    grid_coord_c = create_meshgrid(
        h_coarse, w_coarse, normalized_coordinates=False, device=device
    ).reshape(1, 1, h_coarse * w_coarse, 2)
    grid_coord_c = grid_coord_c * scale  # B * N * n_points * 2, in original image scale

    # Mask padded regions:
    if "masks" in data:
        grid_coord_c, pad_mask = mask_grid_pts_at_padded_regions(
            grid_coord_c, data["masks"], scale=grid_scale
        ) 
        pad_mask= pad_mask[:, 0, :, 0] # Mask: [B, n_pts], for first image
    else:
        pad_mask = None

    # Warp reference view to query views:
    # NOTE: no need mutual nearest neighbour
    query_coords = grid_coord_c[:, 0]  # [B, n_pts, 2]
    valid_mask, warpped_pts, world_points, scales_absolute, view_point_vector = warp_source_views(
        src_points=query_coords,
        src_depth_map=data["depth"][:, 0],
        src_intrinsic=data["intrinsics"][:, 0],
        src_extrin=data["extrinsics"][:, 0],
        src_origin_img_size=data['original_hw'][:, 0],
        dst_intrinsic=data["intrinsics"][:, 1:],
        dst_extrin=data["extrinsics"][:, 1:],
        dst_depth_maps=data["depth"][:, 1:],
        dst_origin_imgs_sizes=data['original_hw'][:, 1:],
        depth_consistency_thres=0.005,
        cycle_reproj_distance_thres=1
    )  # valid_mask: [B, n_dst, n_pts], warpped_pts: [B, n_dst, n_pts, 2], world_pts: [B, n_pts, 3], scales_absolute: [B, n_view, n_pts], view_point_vector: [B, n_view, n_pts, 3]
    valid_mask = valid_mask * pad_mask[:, None] if pad_mask is not None else valid_mask

    # Find valid GT tracks and sample(or pad)
    n_query_view, n_pts = valid_mask.shape[1:]
    track_valid_mask = torch.sum(valid_mask, dim=1) >= (
        n_query_view - track_length_tolerance
    )  # [B, 1, n_pts]
    n_pts_idxs = torch.arange(0, n_pts, device=device)

    valid_pt_idxs_list = []
    padded_mask_list = []
    for b_id in range(B):
        valid_pt_idxs = n_pts_idxs[track_valid_mask[b_id]]  # N_valid
        n_valid = valid_pt_idxs.shape[0]
        if n_valid == 0:
            logger.error(f"No valid gt track for scene:{data['scene_name'][b_id]}, fill the wrong gt to avoid dead lock")
            valid_pt_idxs = torch.tensor([0], device=device).long()
            valid_mask[b_id, :, valid_pt_idxs] = 1
            n_valid = 1

        if n_valid < sample_n_tracks_per_instance:
            if n_valid < 40:
                pass
                # logger.warning(f"Very few gt track for scene:{data['scene_name'][b_id]}, only have: {n_valid}")
            ratio = sample_n_tracks_per_instance // n_valid
            n_sample = sample_n_tracks_per_instance % n_valid

            # Repeat ratio times:
            valid_pt_idxs = valid_pt_idxs.repeat(ratio)
            # Sample a part of idxs:
            sample_idxs = torch.randint(0, n_valid, (n_sample,), device=device)

            valid_pt_idxs = torch.cat(
                [valid_pt_idxs, valid_pt_idxs[sample_idxs]], dim=0
            )  # (sample_n_tracks_per_instance)

            padded_mask = torch.ones_like(valid_pt_idxs).bool()
            padded_mask[:n_valid] = False
        else:
            # Permutate and sample top n:
            perm_idxs = torch.randperm(n_valid, device=device)
            valid_pt_idxs = valid_pt_idxs[perm_idxs][:sample_n_tracks_per_instance]
            padded_mask = torch.zeros_like(valid_pt_idxs).bool()

        valid_pt_idxs_list.append(valid_pt_idxs)
        padded_mask_list.append(padded_mask)

    # GT:
    padded_mask = torch.stack(padded_mask_list, dim=0)[:, None].expand(-1, n_query_view, -1) # B * n_dst * n_selected
    valid_mask = torch.stack(
        [valid_mask[i, :, valid_pt_idxs_list[i]] for i in range(B)]
    )  # [B, n_dst, n_selected]
    valid_mask[padded_mask] = 0

    reference_points = torch.stack(
        [warpped_pts[i, :, valid_pt_idxs_list[i]] for i in range(B)]
    )  # [B, n_dst, n_selected, 2]
    reference_img_idxs = torch.arange(1, N, device=device)[None, :, None].expand(B, -1, reference_points.shape[-2]) # [B, n_dst, n_selected]

    world_points = torch.stack(
        [world_points[i, valid_pt_idxs_list[i]] for i in range(B)]
    )  # [B, n_selected, 3]

    query_points = torch.stack(
        [query_coords[i, valid_pt_idxs_list[i]] for i in range(B)]
    )  # [B, n_selected, 2]
    query_img_idxs = torch.zeros((query_points.shape[0], query_points.shape[1]), device=device).long() # B * n_selected

    scales_absolute = torch.stack([scales_absolute[i, :, valid_pt_idxs_list[i]] for i in range(B)]) # B * n_view * n_selected
    view_point_vector = torch.stack([view_point_vector[i, :, valid_pt_idxs_list[i]] for i in range(B)]) # # B * n_view * n_selected * 3
    if 'scales' in data:
        scales_absolute /= data['scales'][..., [0]]
    scales_relative = (scales_absolute / (scales_absolute[:, [0]] + 1e-4)) # B * n_view * n_selected
    # scale pertubation:
    scales_relative += torch.randn_like(scales_relative) * scale_pertubation * 2 - scale_pertubation
    scales_relative[:, 0] = 1

    # Make coarse query points:
    coarse_pertubation = torch.rand_like(reference_points) * 0.5 * 2 - 0.5 # [-0.5, 0.5]
    reference_points_coarse = ((reference_points / scale[:, 1:]).round() + coarse_pertubation) * scale[:, 1:]

    fine_scales = (
        fine_scale * data["scales"][..., 1:, None, [1, 0]]
        if "scales" in data
        else fine_scale
    )  # B * n_dst * 1 * 2

    reference_points_coarse += (
        torch.rand_like(reference_points_coarse) * reference_points_pertubation * 2
        - reference_points_pertubation
    ) * fine_scales  # Add pertubation [-pertubation, +pertubation], from fine scale to original scale

    # Make fine heatmap gt:
    offset = (reference_points - reference_points_coarse) / fine_scales
    window_radius = (fine_window_size // 2) * scales_relative[:, 1:].clamp(1/(fine_window_size // 2), fine_window_size // 2)
    offset_normalize = ((offset / window_radius[...,None]).clamp(-1,1) + 1) / 2 #[0, 1]
    coord = (offset_normalize * (fine_window_size - 1)) # range: [0, window_size -1], [x, y]
    fine_local_heatmap_gt = torch.zeros((B, n_query_view, sample_n_tracks_per_instance, fine_window_size, fine_window_size), device=device)

    b_ids = repeat(torch.arange(B, device=device), 'b -> (b n m)', n=n_query_view, m=sample_n_tracks_per_instance)
    query_view_ids = repeat(torch.arange(n_query_view, device=device), 'n -> (b n m)', b=B, m=sample_n_tracks_per_instance)
    sampled_ids = repeat(torch.arange(sample_n_tracks_per_instance, device=device), 'm -> (b n m)', b=B, n=n_query_view)
    coord_quantized = (coord.round()).to(torch.long)
    coord_quantized = coord_quantized.view(-1, 2)
    fine_local_heatmap_gt[b_ids, query_view_ids, sampled_ids, coord_quantized[:, 1], coord_quantized[:, 0]] = 1

    # NOTE: all points are in original scale
    data.update(
        {
            "track_valid_mask": valid_mask.to(torch.bool),  # [B, n_dst, n_selected]
            "reference_points_gt": reference_points,  # [B, n_dst, n_selected, 2]
            "reference_points_coarse": reference_points_coarse,  # [B, n_dst, n_selected, 2]
            "reference_img_idxs": reference_img_idxs, # [B, n_dst, n_selected]
            "query_points": query_points,  # [B, n_selected, 2]
            "query_img_idxs": query_img_idxs, # [B, n_selected]
            "world_points": world_points,  # [B, n_selected, 3]
            "scales_relative": scales_relative, # [B, n_view, n_selected]
            "view_point_vector": view_point_vector, # [B, n_view, n_selected, 3]
            "fine_local_heatmap_gt_focal": fine_local_heatmap_gt, # [B, n_dst, n_selected, W, W]
        }
    )


@torch.no_grad()
def compute_supervision(data, config):
    supervision_type = config["type"]
    if supervision_type == "dense_grid":
        # Use grid points as tracks:
        dense_grid_spv(data, config)
    else:
        raise NotImplementedError
