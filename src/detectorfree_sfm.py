import os
import os.path as osp
from loguru import logger
import natsort
from src.evaluator import Evaluator
from src.construct_pairs import construct_img_pairs
from src.utils.vis_utils import save_colmap_ws_to_vis3d
from src.utils.colmap.eval_helper import get_best_colmap_index

def DetectorFreeSfM(
    args,
    method,
    work_dir,
    gt_pose_dir=None,
    prior_intrin_dir=None,
    prior_pose_dir=None,
    prior_colmap_dir=None,
    colmap_configs=None,
    use_ray=False,
    ray_cfg=None,
    verbose=True,
    visualize=True,
):
    # Prepare data structure
    img_resize = args.img_resize
    img_preload = args.img_preload
    n_images = args.n_images
    image_pth = osp.join(work_dir, "images")
    assert osp.exists(image_pth), f"{image_pth} is not exist!"
    img_names = natsort.natsorted(os.listdir(image_pth))
    img_list = [
        osp.join(image_pth, img_name) for img_name in img_names if ("._" not in img_name) and ('.DS_Store' not in img_name)
    ]

    # Used for debugging:
    if n_images is not None:
        img_list = img_list[:n_images]

    selected_img_list = []
    if args.down_sample_ratio is not None:
        assert args.down_sample_ratio > 0
        for id, img_path in enumerate(img_list):
            if id % args.down_sample_ratio == 0:
                selected_img_list.append(img_path)
        logger.info(f"total: {len(selected_img_list)} images")
        img_list = selected_img_list

    logger.info(f"Total {len(img_list)} images") if verbose else None

    img_pairs = construct_img_pairs(
        img_list, args, strategy=args.img_pair_strategy, pair_path=osp.join(work_dir, 'pairs.txt'), verbose=verbose
    )

    # Make evaluator:
    evaluator = (
        Evaluator(img_list, gt_pose_dir, triangulate_mode=args.triangulation_mode, verbose=verbose)
        if not args.close_eval and gt_pose_dir is not None
        else None
    )

    if method == "DetectorFreeSfM":
        from .coarse_match.coarse_match import detector_free_coarse_matching
        from .sfm_runner import coarse_SfM_runner

        # Parse configs
        triangulation_mode = args.triangulation_mode
        enable_post_optimization = args.NEUSFM_enable_post_optimization
        coarse_match_type = args.NEUSFM_coarse_match_type
        coarse_matcher = args.NEUSFM_coarse_matcher
        coarse_match_thr = args.NEUSFM_coarse_match_thr
        coarse_match_round_ratio = args.NEUSFM_coarse_match_round
        suffix = args.suffix if 'suffix' in args else ''

        method_name = "_".join(
            [
                "_".join(
                    [
                        method,
                        coarse_matcher,
                        coarse_match_type,
                        f"round{coarse_match_round_ratio}"
                        if coarse_match_round_ratio is not None
                        else "",
                    ]
                ),
                "pri_pose" if triangulation_mode else "scratch",
            ]
        )
        if prior_intrin_dir is None:
            method_name += '_no_intrin'

        if suffix != "":
            method_name += f'_{suffix}'
        feature_out = osp.join(work_dir, method_name, "keypoints.h5")
        match_out = osp.join(work_dir, method_name, "matches.h5")  # Coarse match
        colmap_coarse_dir = osp.join(work_dir, method_name, "colmap_coarse")
        colmap_refined_dir = osp.join(work_dir, method_name, "colmap_refined")
        vis_dir = osp.join(work_dir, "vis3d", method_name)

        if osp.exists(osp.join(work_dir, method_name)) and args.redo_all:
            os.system(f"rm -rf {osp.join(work_dir, method_name)}")
        os.makedirs(osp.join(work_dir, method_name), exist_ok=True)

        if not osp.exists(match_out) or args.redo_matching:
            # Coarse-Level Matching:
            logger.info("Detector-free coarse matching begin...")
            detector_free_coarse_matching(
                img_list,
                img_pairs,
                feature_out=feature_out,
                match_out=match_out,
                img_resize=img_resize,
                img_preload=img_preload,
                matcher=coarse_matcher,
                match_type=coarse_match_type,
                match_round_ratio=coarse_match_round_ratio,
                match_thr=coarse_match_thr,
                use_ray=use_ray,
                ray_cfg=ray_cfg,
                verbose=verbose,
            )

        if not osp.exists(colmap_coarse_dir) or args.redo_sfm:
            # Coarse Mapping:
            logger.info("Coarse mapping begin...")
            coarse_SfM_runner(
                img_list,
                img_pairs,
                osp.join(work_dir, method_name),
                feature_out=feature_out,
                match_out=match_out,
                colmap_coarse_dir=colmap_coarse_dir,
                colmap_configs=colmap_configs,
                triangulation_mode=triangulation_mode,
                prior_intrin_path=prior_intrin_dir,
                prior_pose_path=prior_pose_dir if triangulation_mode else None,
                prior_model_path=prior_colmap_dir if triangulation_mode else None,
                verbose=verbose,
            )

        if not triangulation_mode:
            best_model_id = get_best_colmap_index(colmap_coarse_dir)
        else:
            best_model_id = '0'
        save_colmap_ws_to_vis3d(
            osp.join(colmap_coarse_dir, best_model_id),
            vis_dir,
            name_prefix="coarse",
        ) if visualize else None

        if evaluator is not None and not enable_post_optimization:
            error_dict, metrics_dict = evaluator.eval_metric(
                osp.join(colmap_coarse_dir, best_model_id)
            )
            return metrics_dict

        # Post Optimization
        if enable_post_optimization:
            from .post_optimization.post_optimization import post_optimization

            if (
                not osp.exists(osp.join(colmap_refined_dir, "images.bin"))
                or args.redo_refine
            ):
                post_optimization(
                    img_list,
                    img_pairs,
                    match_out_pth=match_out,
                    chunk_size=args.NEUSFM_refinement_chunk_size,
                    matcher_model_path=args.NEUSFM_fine_match_model_path,
                    matcher_cfg_path=args.NEUSFM_fine_match_cfg_path,
                    img_resize=img_resize,
                    img_preload=img_preload,
                    colmap_coarse_dir=osp.join(colmap_coarse_dir, best_model_id),
                    refined_model_save_dir=colmap_refined_dir,
                    only_basename_in_colmap=True,
                    fine_match_use_ray=use_ray,
                    ray_cfg=ray_cfg,
                    colmap_configs=colmap_configs,
                    refine_iter_n_times=args.refine_iter_n_times,
                    refine_3D_pts_only=triangulation_mode and not args.tri_refine_pose_and_points,
                    verbose=verbose,
                )

            save_colmap_ws_to_vis3d(
                colmap_refined_dir, vis_dir, name_prefix="after_refine"
            ) if visualize else None

            # Evaluation:
            if evaluator is not None:
                logger.info(
                    f"Metric of: Coarse reconstruction"
                ) if verbose else None
                error_dict, metrics_dict = evaluator.eval_metric(
                    osp.join(colmap_coarse_dir, best_model_id)
                )
                temp_refined_dirs = [
                    osp.join(osp.dirname(colmap_refined_dir), f"model_refined_{id}")
                    for id in range(args.refine_iter_n_times - 1)
                ]
                for temp_dir in temp_refined_dirs:
                    logger.info(f"Metric of: {temp_dir}") if verbose else None
                    error_dict, metrics_dict = evaluator.eval_metric(
                        osp.join(osp.dirname(colmap_refined_dir), temp_dir)
                    )

                logger.info(f"Metric of: Final") if verbose else None
                error_dict, metrics_dict = evaluator.eval_metric(colmap_refined_dir)

                metrics_dict = evaluator.prepare_output_from_buffer()
                return metrics_dict
    else:
        raise NotImplementedError
