import os
import os.path as osp

from loguru import logger
from tqdm import tqdm
from shutil import rmtree
from pathlib import Path

from src.sfm_runner import sfm_model_geometry_refiner
from src.utils.data_io import load_obj
from src.sfm_runner import reregistration
from ..dataset.coarse_sfm_refinement_dataset import CoarseColmapDataset
from .matcher_model import *
from .utils.write_fixed_images import fix_farest_images, fix_all_images

cfgs = {
    "coarse_colmap_data": {
        "img_resize": 1200,
        "df": None,
        "feature_track_assignment_strategy": "midium_scale",
        "img_preload": False,
    },
    "fine_match_debug": True,
    "multiview_matcher_data": {
        "max_track_length": 16,
        "chunk": 6000 
    },
    "fine_matcher": {
        "model": {
            "cfg_path": [''],
            "weight_path": [''],
            "seed": 666,
        },
        "visualize": False,
        "extract_feature_method": "fine_match_backbone",
        "ray": {
            "slurm": False,
            "n_workers": 1,
            "n_cpus_per_worker": 1,
            "n_gpus_per_worker": 1,
            "local_mode": False,
        },
    },
    "visualize": False,
    "evaluation": False,
    "refine_iter_n_times": 2,
    "model_refiner_no_filter_pts": False,
    "first_iter_resize_img_to_half": False,
    "enable_update_reproj_kpts_to_model": False,
    "enable_adaptive_downscale_window": True, # Down scale searching window size after each iteration, e.g., 15->11->7
    "incremental_refiner_filter_thresholds": [3, 2, 1.5],
    "incremental_refiner_use_pba": False, # NOTE: pba does not allow share intrins or fix extrinsics, and only allow simple_radial camer model
    "enable_multiple_models": False,
}

def post_optimization(
    image_lists,
    covis_pairs_pth,
    colmap_coarse_dir,
    refined_model_save_dir,
    match_out_pth,
    chunk_size=6000,
    matcher_model_path=None,
    matcher_cfg_path=None,
    img_resize=None,
    img_preload=False,
    fine_match_use_ray=False,  # Use ray for fine match
    ray_cfg=None,
    colmap_configs=None,
    only_basename_in_colmap=False,
    visualize_dir=None,
    vis3d_pth=None,
    refine_iter_n_times=2,
    refine_3D_pts_only=False,
    verbose=True
):
    """
    Iterative n times:
        Reproject current 3D model to update keypoints;
        Refine Keypoints;
        Itertative m times:
            BA(optimize 3D points and poses);
            Adjust scene structure:
                merge feature track;
                complete feature track;
                filter feature track;
        Reregistration[Optional];
    """
    # Cfg overwrite:
    cfgs['refine_iter_n_times'] = refine_iter_n_times
    if matcher_cfg_path is not None:
        cfgs['fine_matcher']['model']['cfg_path'] = matcher_cfg_path
    if matcher_model_path is not None:
        cfgs['fine_matcher']['model']['weight_path'] = matcher_model_path
    
    cfgs['coarse_colmap_data']['img_preload'] = img_preload
    cfgs['incremental_refiner_use_pba'] = colmap_configs["use_pba"]
    cfgs['multiview_matcher_data']['chunk'] = chunk_size

    # Link images to temp directory for later extract colors.
    temp_image_path = osp.join(osp.dirname(refined_model_save_dir), f'temp_images')
    if osp.exists(temp_image_path):
        os.system(f"rm -rf {temp_image_path}")
    os.makedirs(temp_image_path)
    for img_path in image_lists:
        os.system(f"ln -s {img_path} {osp.join(temp_image_path, osp.basename(img_path))}")

    # Clear all previous results:
    temp_refined_dirs = [dir_name for dir_name in os.listdir(osp.dirname(refined_model_save_dir)) if 'model_refined' in dir_name or osp.basename(refined_model_save_dir) == dir_name]
    for temp_result_name in temp_refined_dirs:
        rmtree(osp.join(osp.dirname(refined_model_save_dir),  temp_result_name))

    iter_n_times = cfgs['refine_iter_n_times']
    iter_id = tqdm(range(iter_n_times)) if verbose else range(iter_n_times)

    for i in iter_id:
        if cfgs['first_iter_resize_img_to_half'] and i == 0:
            cfgs["coarse_colmap_data"]['img_resize'] = img_resize // 2
        else:
            cfgs["coarse_colmap_data"]['img_resize'] = img_resize

        # Construct scene data
        colmap_image_dataset = CoarseColmapDataset(
            cfgs["coarse_colmap_data"],
            image_lists,
            covis_pairs_pth,
            colmap_coarse_dir if i == 0 else last_model_dir,
            refined_model_save_dir,
            only_basename_in_colmap=only_basename_in_colmap,
            vis_path=vis3d_pth if vis3d_pth is not None else None,
            verbose=verbose
        )
        logger.info("Scene data construct finish!") if verbose else None

        if cfgs['enable_update_reproj_kpts_to_model']:
            if i != 0:
                # Leverage current model to update keypoints
                colmap_image_dataset.update_kpts_by_current_model_projection(fix_ref_node=True)

        state = colmap_image_dataset.state
        if state == False:
            logger.warning(
                f"Build colmap coarse dataset fail! colmap point3D or images or cameras is empty!"
            )
            return state, None, None

        # Fine level match
        save_path = osp.join(match_out_pth.rsplit("/", 2)[0], "fine_matches.pkl")
        if not osp.exists(save_path) or cfgs["fine_match_debug"]:
            logger.info(f"Multi-view refinement matching begin!")
            model_idx = 0 if i == 0 else 1
            rewindow_size_factor = i * 2
            fine_match_results = multiview_matcher(
                cfgs["fine_matcher"],
                cfgs["multiview_matcher_data"],
                colmap_image_dataset,
                rewindow_size_factor=rewindow_size_factor if cfgs["enable_adaptive_downscale_window"] else None,
                model_idx=model_idx if cfgs['enable_multiple_models'] else None,
                visualize_dir=visualize_dir,
                use_ray=fine_match_use_ray,
                ray_cfg=ray_cfg,
                verbose=verbose
            )
        else:
            logger.info(f"Fine matches exists! Load from {save_path}")
            fine_match_results = load_obj(save_path)
        
        if i != iter_n_times -1:
            current_model_dir = osp.join(osp.dirname(refined_model_save_dir), f'model_refined_{i}')
        else:
            current_model_dir = refined_model_save_dir

        last_model_dir = current_model_dir

        colmap_refined_kpts_dir = osp.join(osp.dirname(refined_model_save_dir), 'temp_refined_kpts')
        Path(colmap_refined_kpts_dir).mkdir(parents=True, exist_ok=True)
        colmap_image_dataset.update_refined_kpts_to_colmap_multiview(fine_match_results)

        if i == 0:
            if osp.exists(osp.join(colmap_refined_kpts_dir, 'database.db')):
                os.system(f"rm -rf {osp.join(colmap_refined_kpts_dir, 'database.db')}")
            os.system(f"cp {osp.join(osp.dirname(colmap_coarse_dir), 'database.db')} {osp.join(colmap_refined_kpts_dir, 'database.db')}")
            if refine_3D_pts_only:
                # Triangulation mode
                fix_all_images(reconstructed_model_dir=colmap_coarse_dir, output_path=osp.join(colmap_refined_kpts_dir, 'fixed_images.txt'))
            else:
                fix_farest_images(reconstructed_model_dir=colmap_coarse_dir, output_path=osp.join(colmap_refined_kpts_dir, 'fixed_images.txt'))

        colmap_image_dataset.save_colmap_model(osp.join(colmap_refined_kpts_dir, 'model'))

        # Refinement:
        filter_threshold = cfgs['incremental_refiner_filter_thresholds'][i] if i < len(cfgs['incremental_refiner_filter_thresholds'])-1 else cfgs['incremental_refiner_filter_thresholds'][-1]
        success = sfm_model_geometry_refiner.main(colmap_refined_kpts_dir, current_model_dir, no_filter_pts=cfgs["model_refiner_no_filter_pts"], colmap_configs=colmap_configs, image_path=temp_image_path, verbose=verbose, refine_3D_pts_only=refine_3D_pts_only, filter_threshold=filter_threshold, use_pba=cfgs["incremental_refiner_use_pba"])

        if not success:
            # Refine failed scenario, use the coarse model instead.
            os.system(f"cp {osp.join(colmap_refined_kpts_dir, 'model') + '/*'} {current_model_dir}")

        os.system(f"rm -rf {osp.join(colmap_refined_kpts_dir, 'model')}")
        os.makedirs(osp.join(colmap_refined_kpts_dir, 'model'), exist_ok=True)
        os.system(f"cp {current_model_dir+'/*'} {osp.join(colmap_refined_kpts_dir, 'model')}")

        # Re-registration:
        if i % 2 == 0 and not refine_3D_pts_only:
            reregistration.main(colmap_refined_kpts_dir, current_model_dir, colmap_configs=colmap_configs, verbose=verbose)

    return state