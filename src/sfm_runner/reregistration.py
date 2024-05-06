import os
import logging
import subprocess
import os.path as osp

from pathlib import Path


def run_image_reregistration(
    deep_sfm_dir, after_refine_dir, colmap_path, image_path="/", colmap_configs=None, verbose=True
):
    logging.info("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, "model")
    database_path = osp.join(deep_sfm_dir, "database.db")
    cmd = [
        str(colmap_path),
        "image_registrator",
        "--database_path",
        str(database_path),
        "--input_path",
        str(deep_sfm_model_dir),
        "--output_path",
        str(after_refine_dir),
    ]

    if colmap_configs is not None and colmap_configs["no_refine_intrinsics"] is True:
        cmd += [
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ]
    
    if 'reregistration' in colmap_configs:
        # Set to lower threshold to registrate more images
        cmd += [
            "--Mapper.abs_pose_max_error",
            str(colmap_configs['reregistration']['abs_pose_max_error']),
            "--Mapper.abs_pose_min_num_inliers",
            str(colmap_configs['reregistration']['abs_pose_min_num_inliers']),
            "--Mapper.abs_pose_min_inlier_ratio",
            str(colmap_configs['reregistration']['abs_pose_min_inlier_ratio']),
            "--Mapper.filter_max_reproj_error",
            str(colmap_configs['reregistration']['filter_max_reproj_error'])
        ]

    if verbose:
        logging.info(' '.join(cmd))
        ret = subprocess.call(cmd)
    else:
        ret_all = subprocess.run(cmd, capture_output=True)
        with open(osp.join(after_refine_dir, 'reregistration_output.txt'), 'w') as f:
            f.write(ret_all.stdout.decode())
        ret = ret_all.returncode

    if ret != 0:
        logging.warning("Problem with image registration, existing.")
        exit(ret)


def main(
    deep_sfm_dir,
    after_refine_dir,
    colmap_path="colmap",
    image_path="/",
    colmap_configs=None,
    verbose=True
):
    assert Path(deep_sfm_dir).exists(), deep_sfm_dir

    Path(after_refine_dir).mkdir(parents=True, exist_ok=True)
    run_image_reregistration(
        deep_sfm_dir,
        after_refine_dir,
        colmap_path,
        image_path,
        colmap_configs=colmap_configs,
        verbose=verbose
    )
