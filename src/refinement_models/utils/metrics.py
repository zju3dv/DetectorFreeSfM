import torch
import numpy as np
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
from src.utils.utils import estimate_pose, estimate_pose_degensac, estimate_pose_magsac, \
                            compute_pose_error, pose_auc, epidist_prec


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


@torch.no_grad()
def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """

    query_points = data['query_points_refined'] # [B, n_track, 2]
    reference_points_refined = data['reference_points_refined'][-1].clone().detach() # [B, n_ref_view, n_track, 2]
    B, n_ref_view = reference_points_refined.shape[:2]
    intrinsics = data['intrinsics'] # [B, n_view, 3, 3]
    relative_extrinsics = data['relative_poses'] # [B, n_view, 4, 4]
    relative_extrinsics_flatten = relative_extrinsics.view(-1, 4, 4) # [B*n_view, 4, 4]
    Tx = numeric.cross_product_matrix(relative_extrinsics_flatten[..., :3, 3])
    E_mat = (Tx @ relative_extrinsics_flatten[..., :3, :3]).view(B, -1, 3, 3) #[B, n_view, 3, 3]

    # [B, n_ref_view, n_track]
    epi_errs = torch.stack([torch.stack([symmetric_epipolar_distance(query_points[bs], reference_points_refined[bs, view_id], E_mat[bs, 1 + view_id], intrinsics[bs, 0], intrinsics[bs, 1+view_id]) for view_id in range(n_ref_view)], dim=0) for bs in range(B)], dim=0)
    data.update({'epi_errs': epi_errs})


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    method = config['pose_estimation_method'] # RANSAC
    pixel_thr = config['ransac_pixel_thr']  # 1.0
    conf = config['ransac_conf']  # 0.99999
    max_iters = config['ransac_max_iters']  # 1000
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    query_points = data['query_points_refined'].cpu().numpy() # [B, n_track, 2]
    reference_points_refined = data['reference_points_refined'][-1].clone().detach().cpu().numpy() # [B, n_ref_view, n_track, 2]
    intrinsics = data['intrinsics'].cpu().numpy() # [B, n_view, 3, 3]
    relative_poses = data['relative_poses'].cpu().numpy() # [B, n_view, 4, 4]
    track_valid_mask = data['track_valid_mask'].cpu().numpy() # [B, n_ref_view, n_track]
    # reference_points_gt = data['reference_points_gt'].cpu().numpy()

    B, n_ref_view = reference_points_refined.shape[:2]

    for bs in range(B):
        for view_id in range(n_ref_view):
            mask = track_valid_mask[bs, view_id]
            if method == 'RANSAC':
                ret = estimate_pose(query_points[bs][mask], reference_points_refined[bs, view_id][mask], intrinsics[bs, 0], intrinsics[bs, 1+view_id], pixel_thr, conf=conf)
            elif method == 'DEGENSAC':
                ret = estimate_pose_degensac(query_points[bs][mask], reference_points_refined[bs, view_id][mask], intrinsics[bs, 0], intrinsics[bs, 1+view_id], pixel_thr, conf=conf, max_iters=max_iters)
            elif method == 'MAGSAC':
                raise NotImplementedError
                ret = estimate_pose_magsac(pts0[mask], pts1[mask], K0[bs], K1[bs], config.TRAINER.USE_MAGSACPP, conf=conf, max_iters=max_iters)
            else:
                raise NotImplementedError

            if ret is None:
                data['R_errs'].append(np.inf)
                data['t_errs'].append(np.inf)
                data['inliers'].append(np.array([]).astype(np.bool))
            else:
                # Compute pose gt:
                R, t, inliers = ret
                t_errs, R_errs = compute_pose_error(relative_poses[bs, 1 + view_id], R, t, ignore_gt_t_thr=0.0)
                data['R_errs'].append(R_errs)
                data['t_errs'].append(t_errs)
                data['inliers'].append(inliers)


def aggregate_metrics(metrics, configs):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4
    """
    epi_err_thr = configs['epi_err_thr']

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)
    aucs = pose_auc(pose_errors, angular_thresholds, True)  # (auc@5, auc@10, auc@20)

    # # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object), dist_thresholds, True)  # (prec@epi_err_thr)

    # return aucs
    return {**aucs, **precs}
