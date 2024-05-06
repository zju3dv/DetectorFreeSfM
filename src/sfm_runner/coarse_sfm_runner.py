import os
import os.path as osp
from loguru import logger
from shutil import rmtree
from pathlib import Path

from . import generate_empty
from third_party.Hierarchical_Localization.hloc import reconstruction, triangulation


def coarse_SfM_runner(
    img_list,
    img_pairs,
    work_dir,
    feature_out,
    match_out,
    colmap_coarse_dir,
    colmap_debug=True,
    colmap_configs=None,
    triangulation_mode=False,
    prior_intrin_path=None,
    prior_pose_path=None,
    prior_model_path=None,
    verbose=True
):
    base_path = work_dir
    os.makedirs(base_path, exist_ok=True)
    colmap_temp_path = osp.join(base_path, "temp_output")
    colmap_output_path = colmap_coarse_dir
    # create temp directory
    if osp.exists(colmap_temp_path):
        logger.info(" -- temp path exists - cleaning up from crash")
        rmtree(colmap_temp_path)
        if os.path.exists(colmap_output_path):
            rmtree(colmap_output_path)

    # create output directory
    if osp.exists(colmap_output_path):
        if not colmap_debug:
            logger.info("colmap results already exists, don't need to run colmap")
            return
        else:
            rmtree(colmap_output_path)

    os.makedirs(colmap_temp_path)
    os.makedirs(colmap_output_path)

    # Create colmap-friendy structures
    os.makedirs(os.path.join(colmap_temp_path, "images"))
    img_paths = img_list

    # Link images
    temp_images_path = osp.join(colmap_temp_path, 'images')
    for _src in img_paths:
        _dst = osp.join(temp_images_path, osp.basename(_src))
        # os.system(f"ln -s {_src} {_dst}")
        os.system(f"cp {_src} {_dst}") # NOTE: copy image is more robust but slower
    
    # Ouput match pairs:
    pair_path = osp.join(colmap_temp_path, 'pairs.txt')
    with open(pair_path, "w") as f:
        for img_pair in img_pairs:
            img0_path, img1_path = img_pair.split(" ")
            img0_name = osp.basename(img0_path)
            img1_name = osp.basename(img1_path)

            # Load matches
            f.write(img0_name + " " + img1_name + "\n")

    if not triangulation_mode:
        reconstruction.main(Path(colmap_output_path), Path(temp_images_path), Path(pair_path), Path(feature_out), Path(match_out), Path(prior_intrin_path) if prior_intrin_path is not None else None, verbose=verbose, colmap_configs=colmap_configs)
    else:
        # Prepare reference SfM model
        reference_sfm_model = osp.join(colmap_temp_path, 'sfm_empty')
        generate_empty.generate_model(
            img_list,
            reference_sfm_model,
            prior_colmap_model_path=prior_model_path,
            prior_pose_path=prior_pose_path,
            prior_intrin_path=prior_intrin_path,
            single_camera=colmap_configs["ImageReader_single_camera"],
        )

        triangulation.main(Path(colmap_output_path), Path(reference_sfm_model), Path(temp_images_path), Path(pair_path), Path(feature_out), Path(match_out), colmap_configs=colmap_configs, verbose=verbose)
    rmtree(colmap_temp_path)