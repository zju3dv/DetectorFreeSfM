import os
import logging
import subprocess
import os.path as osp
from shutil import rmtree
import multiprocessing

from pathlib import Path

COLMAP_PATH = os.environ.get("COLMAP_PATH", 'colmap') # 'colmap is default value
def run_incremental_model_refiner(
    deep_sfm_dir, after_refine_dir, no_filter_pts=False, image_path="/", colmap_configs=None, verbose=True, refine_3D_pts_only=False, filter_threshold=2, use_pba=False
):
    logging.info("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, "model")
    database_path = osp.join(deep_sfm_dir, "database.db")
    threshold = filter_threshold
    cmd = [
        COLMAP_PATH,
        "incremental_model_refiner_no_filter_pts" if no_filter_pts else "incremental_model_refiner",
        "--input_path",
        str(deep_sfm_model_dir),
        "--output_path",
        str(after_refine_dir),
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--Mapper.filter_max_reproj_error",
        str(threshold),
        "--Mapper.tri_merge_max_reproj_error",
        str(threshold),
        "--Mapper.tri_complete_max_reproj_error",
        str(threshold),
        "--Mapper.extract_colors",
        str('1')
    ]

    if use_pba:
        # NOTE: PBA does not allow share intrinsics or fix extrinsics, and only allow SIMPLE_RADIAL camera model
        cmd += [
            "--Mapper.ba_global_use_pba",
            "1"
        ]
    else:
        cmd += [
            "--image_list_path",
            str(osp.join(deep_sfm_dir, 'fixed_images.txt')),
        ]
        pass

    if (colmap_configs is not None and colmap_configs["no_refine_intrinsics"] is True) or refine_3D_pts_only:
        cmd += [
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_extra_params", # Distortion params
            "0",
        ]

    if colmap_configs is not None and 'n_threads' in colmap_configs:
        cmd += ["--Mapper.num_threads", str(min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))]
    
    if refine_3D_pts_only:
        if "--image_list_path" not in cmd:
            cmd += [
                "--image_list_path",
                str(osp.join(deep_sfm_dir, 'fixed_images.txt')),
            ] # For triangulation, must fix!

        cmd += [
            "--Mapper.fix_existing_images",
            "1",
        ]

    if verbose:
        logging.info(' '.join(cmd))
        ret = subprocess.call(cmd)
    else:
        ret_all = subprocess.run(cmd, capture_output=True)
        with open(osp.join(after_refine_dir, 'incremental_model_refiner_output.txt'), 'w') as f:
            f.write(ret_all.stdout.decode())
        ret = ret_all.returncode

    if ret != 0:
        logging.warning(f"Problem with run_incremental_model_refiner for {deep_sfm_model_dir}, existing.")
        return False
    else:
        return True


def main(
    deep_sfm_dir,
    after_refine_dir,
    no_filter_pts=False,
    image_path="/",
    colmap_configs=None,
    refine_3D_pts_only=False,
    filter_threshold=2,
    use_pba=False,
    verbose=True,
):
    assert Path(deep_sfm_dir).exists(), deep_sfm_dir

    if osp.exists(after_refine_dir):
        rmtree(after_refine_dir)
    Path(after_refine_dir).mkdir(parents=True, exist_ok=True)
    success = run_incremental_model_refiner(
            deep_sfm_dir,
            after_refine_dir,
            no_filter_pts,
            image_path,
            colmap_configs=colmap_configs,
            refine_3D_pts_only=refine_3D_pts_only,
            filter_threshold=filter_threshold,
            use_pba=use_pba,
            verbose=verbose
        )
    return success