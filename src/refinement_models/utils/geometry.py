import torch
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, repeat

def plot(src_pts, vector):
    """
    Only for debug visualization.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(src_pts[:, 0],src_pts[:, 1], src_pts[:,2], vector[:,0], vector[:,1], vector[:,2], normalize=False, color='r', arrow_length_ratio=0.15)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # ax.set_title('b-vectors on unit sphere')

    plt.savefig('test.png')


@torch.no_grad()
def warp_source_views(
    src_points,
    src_depth_map,
    src_intrinsic,
    src_extrin,
    src_origin_img_size,
    dst_intrinsic,
    dst_extrin,
    dst_depth_maps,
    dst_origin_imgs_sizes,
    depth_consistency_thres=0.2,
    cycle_reproj_distance_thres=5, # pixel
    border_thres=8,
):
    """
    Warp source view points to multiple query views and check depth consistency.

    Args:
        src_view_points: B * n_pts * 2 - <x, y>
        src_view_depth_map: B * H * W
        src_intrinsic: B * 3 * 3
        src_extrin: B * 4 * 4, from world to cam
        src_origin_img_size: B * 2, - <h, w>
        dst_intrinsic: B * n_dst * 3 * 3
        dst_depth_maps: B * n_dst * H * W
        dst_mask: B * n_ds * H * W
        dst_imgs_sizes: B * n_dst * 2
        dst_extrin: B * n_dst * 4 * 4, from world to cam

    Return:
        valid_mask: B * n_dst * n_pts
        dst_pts: B * n_dst * n_pts * 2
        world_points: B * n_pts * 3
    """
    src_points = src_points.round().long()
    device = src_points.device
    B, N_pts = src_points.shape[:2]
    _, N_dst = dst_intrinsic.shape[:2]

    # Sample depth, get calculable_mask on depth != 0
    src_points_depth = torch.stack(
        [
            src_depth_map[i, src_points[i, :, 1], src_points[i, :, 0]]
            for i in range(src_points.shape[0])
        ],
        dim=0,
    )  # B * N_pts
    nonzero_mask = src_points_depth != 0  # B * N_pts
    src_border_mask = (
        (src_points[:, :, 0] > border_thres)
        * (src_points[:, :, 1] > border_thres)
        * (src_points[:, :, 0] < (src_origin_img_size[:, None, 1] - border_thres))
        * (src_points[:, :, 1] < (src_origin_img_size[:, None, 0] - border_thres))
    ) # B * N_pts

    # Unproject:
    src_points_h = (
        torch.cat([src_points, torch.ones((B, N_pts, 1), device=device)], dim=-1)
        * src_points_depth[..., None]
    )  # B * N_pts * 3
    src_points_cam = src_intrinsic.inverse() @ src_points_h.transpose(
        2, 1
    )  # B * 3 * N_pts

    # From source cam to world:
    src_pose = src_extrin.inverse()
    world_points = (
        src_pose[:, :3, :3] @ src_points_cam + src_pose[:, :3, [3]]
    )  # B * 3 * N_pts

    # Transform to dst views:
    dst_pts_cam = (
        dst_extrin[..., :3, :3] @ world_points[:, None] + dst_extrin[..., :3, [3]]
    )  # B * N_dst * 3 * N_pts

    # Project:
    dst_pts_h = (dst_intrinsic @ dst_pts_cam).transpose(3, 2)  # B * N_dst * N_pts * 3
    proj_depth = dst_pts_h[..., 2]  # B * N_dst * N_pts * 1
    dst_pts = dst_pts_h[..., :2] / (dst_pts_h[..., [2]] + 1e-4)  # B * N_dst * N_pts * 2

    # Covis check:
    h, w = dst_depth_maps.shape[-2:]
    covisible_mask = (
        (dst_pts[..., 0] > border_thres)
        * (dst_pts[..., 0] < dst_origin_imgs_sizes[..., None, 1] - border_thres)
        * (dst_pts[..., 1] > border_thres)
        * (dst_pts[..., 1] < dst_origin_imgs_sizes[..., None, 0] - border_thres)
    )  # B * N_dst * N_pts
    dst_pts[~covisible_mask, :] = 0  # B * N_dst * N_pts * 2

    # Depth consistency check:
    dst_pts_long = (dst_pts.long()).view(B * N_dst, N_pts, 2)
    dst_depth_maps = rearrange(dst_depth_maps, "b n h w -> (b n) h w")
    sampled_depth = torch.stack(
        [
            dst_depth_maps[i, dst_pts_long[i, :, 1], dst_pts_long[i, :, 0]]
            for i in range(dst_pts_long.shape[0])
        ],
        dim=0,
    )
    sampled_depth = rearrange(
        sampled_depth, "(b n) m -> b n m", b=B
    )  # B * N_dst * N_pts
    consistency_mask = (
        (sampled_depth - proj_depth) / (sampled_depth + 1e-4)
    ).abs() < depth_consistency_thres

    # Back project to src view for cycle check:
    dst_pts_h = torch.cat([dst_pts, torch.ones((B, N_dst, N_pts, 1), device=device)], dim=-1) * sampled_depth[..., None] # B * N_dst * N_pts * 3
    dst_pts_cam = dst_intrinsic.inverse() @ dst_pts_h.transpose(2, 3)
    dst_pose = dst_extrin.inverse()
    world_points_cycle_back = dst_pose[:, :, :3, :3] @ dst_pts_cam + dst_pose[:, :, :3, [3]]
    src_warp_back_cam = src_extrin[:, None, :3, :3] @ world_points_cycle_back + src_extrin[:, None, :3, [3]]
    src_warp_back_h = (src_intrinsic @ src_warp_back_cam).transpose(2, 3)
    src_back_proj_depth = src_warp_back_h[..., 2]
    src_back_proj_pts = src_warp_back_h[..., :2] / (src_warp_back_h[..., [2]] + 1e-4)
    cycle_reproj_distance_mask = (torch.linalg.norm(src_back_proj_pts - src_points[:, None], dim=-1)) < cycle_reproj_distance_thres
    cycle_depth_distance_mask = ((src_back_proj_depth - src_points_depth[:, None]).abs() / (src_points_depth[:, None] + 1e-4)) < depth_consistency_thres

    valid_mask = (
        nonzero_mask[:, None] * src_border_mask[:, None] * covisible_mask * consistency_mask * cycle_reproj_distance_mask * cycle_depth_distance_mask
    )  # B * N_dst * N_pts
    
    # Get absolute scale of each points:
    dst_scales_absolute = dst_intrinsic[:, :, 0, 0][..., None] / (proj_depth + 1e-4) # B * N_dst * N_pts
    src_scales_absolute = src_intrinsic[:, 0, 0][..., None] / (src_points_depth + 1e-4) # B * N_pts
    scale_absolute = torch.cat([src_scales_absolute[:, None], dst_scales_absolute], dim=1) # B * N_view * N_pts

    # Get relative view points infos:
    relative_pose = src_extrin[:, None] @ dst_extrin.inverse()# B * N_dst * 4 * 4
    t = relative_pose[..., :3, 3] # B * N_dst * 3, from src camera to dst camera
    """
             /             \
            /               \
         f /                 \ a
          /                   \
         / alpha         beta  \
        /_______________________\
    src_view        t         dst_view;  view_point_vector is encoded by the t_norm and the gamma
    """
    # plot(src_pts=np.zeros((t.shape[1], 3)), vector=t[0].cpu().numpy())
    f = repeat(src_points_cam.transpose(1,2), 'b n_track c -> b n_view n_track c', n_view=N_dst) # B * N_dst * N_track * 3
    t = repeat(t, 'b n_view c -> b n_view n_track c', n_track=N_pts) # B * N_dst * N_track * 3
    a = f - t # B * N_track * N_dst * 3
    f_norm, t_norm, a_norm = map(lambda x: torch.linalg.norm(x, dim=-1, keepdim=True), [f, t, a])
    alpha = torch.arccos(torch.einsum('bntc,bntc->bnt', f, t)[..., None] / (f_norm * t_norm + 1e-4))
    beta = torch.arccos(torch.einsum('bntc,bntc->bnt', a, -1 * t)[..., None] / (a_norm * t_norm + 1e-4))
    gamma = torch.arccos(torch.zeros(1, device=a.device)).item() * 2  - alpha - beta
    view_point_vector = (t / (t_norm + 1e-4)) * gamma # B * N_dst * N_track * 3
    view_point_vector = torch.cat([torch.zeros((B, 1, N_pts, 3), device=view_point_vector.device), view_point_vector], dim=1) # B * N_view * N_track * 3

    return valid_mask, dst_pts, world_points.transpose(2, 1), scale_absolute, view_point_vector
