import numpy as np
from src.utils.utils import make_matching_plot, error_colormap


def draw_single_evaluation(data, bs, ref_view_id, thr=3e-4):

    image0 = (data['images'][bs][0].cpu().numpy() * 255).round().astype(np.int32)[0] # H*W
    image1 = (data['images'][bs][ref_view_id + 1].cpu().numpy() * 255).round().astype(np.int32)[0]

    query_points = data['query_points'][bs].cpu().numpy() # [n_track, 2]
    reference_points = data['reference_points_refined'][-1][bs][ref_view_id].clone().detach().cpu().numpy() # [n_track, 2]
    mask = data['track_valid_mask'][bs][ref_view_id].cpu().numpy() # [n_track]
    epi_errs = data['epi_errs'][bs][ref_view_id].cpu().numpy()

    query_points = query_points[mask]
    reference_points = reference_points[mask]
    epi_errs = epi_errs[mask]
    
    # for megadepth, we visualize matches on the resized image
    if 'scales' in data:
        query_points = query_points / data['scales'][bs, [0], [1,0]].cpu().numpy()
        reference_points = reference_points / data['scales'][bs, [ref_view_id + 1], [1,0]].cpu().numpy()

    correct_mask = epi_errs < thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = len(query_points)
    recall = n_correct / n_gt_matches

    # Display matching info
    color = np.clip(epi_errs / (thr*2), 0, 1)
    color = error_colormap(1 - color, alpha=0.3)  # 0.1 is too vague
    text = [
        f"Scene {data['scene_name'][bs]}",
        f'Matches {len(query_points)}',
        f"relative scale {data['scales_relative_reference'].cpu().numpy()[bs][ref_view_id].mean()}",
        f'Precision({thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(query_points)}',
        f'Recall({thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    figure = make_matching_plot(image0, image1, query_points, reference_points, query_points, reference_points, color, text)
    return figure


def draw_all_figures(data, configs):
    """
    Args:
        data (dict)
        config (dict)
    Returns:
        figures (dict{plt.figure})
    """
    figures = {'evaluation': []}

    B, n_ref_view = data['reference_points_refined'][-1].shape[:2]
    for bs in range(B):
        for ref_view_id in range(n_ref_view):
            figures['evaluation'].append(draw_single_evaluation(data, bs, ref_view_id, thr=configs["epi_err_thr"]))

    return figures
