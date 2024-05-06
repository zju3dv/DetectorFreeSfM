import os
import os.path as osp
from src.utils.data_io import save_h5, load_calib
from src.utils.metric_utils import compute_stereo_metrics_from_colmap, pose_auc
from tqdm import tqdm
import numpy as np

# Metrics utils
def scene_pose_auc(error_dict, angular_thresholds=[5, 10, 20], save_path=None):
    """
    Calculate and save pose auc of all bags in a scene
    Parameters:
    -------------
    error_dict: Dict
        {bag_name: {pair_name : [err_q : np.array, err_t : np.array]}}
    """
    R_error, t_error = [], []

    # combine all bags
    for sub_bag_err in error_dict.values():
        for pair_err in sub_bag_err.values():
            R_error.append(pair_err[0])
            t_error.append(pair_err[1])

    # pose auc
    pose_errors = np.max(np.stack([R_error, t_error]), axis=0)
    aucs = pose_auc(pose_errors, angular_thresholds, True)

    if save_path is not None:
        assert osp.exists(save_path.rsplit("/", 1)[0])
        with open(save_path, "w") as f:
            for key in aucs.keys():
                f.write(key + ":\n")
                f.write(f"{aucs[key]}\n")
                f.write("--------------\n")
        print(f"{(osp.splitext(save_path)[0]).rsplit('/',1)[1]} aucs:", aucs)
    return aucs


def imc_bag_pose_auc(
    error_dict,
    save_dir,
    base_save_name="scene_pose_auc.txt",
    thresholds=[1, 2, 3, 4, 5, 10, 20],
):
    err_dict_bag = {"5bag": {}, "10bag": {}, "25bag": {}}
    for bag_name, bag_error in error_dict.items():
        if "5bag" in bag_name and "25bag" not in bag_name:
            err_dict_bag["5bag"][bag_name] = bag_error
        elif "10bag" in bag_name:
            err_dict_bag["10bag"][bag_name] = bag_error
        elif "25bag" in bag_name:
            err_dict_bag["25bag"][bag_name] = bag_error

    scene_pose_auc(
        error_dict,
        angular_thresholds=thresholds,
        save_path=osp.join(save_dir, base_save_name),
    )

    for name, err_dict_sub_bag in err_dict_bag.items():
        scene_pose_auc(
            err_dict_sub_bag,
            angular_thresholds=thresholds,
            save_path=osp.join(
                save_dir, osp.splitext(base_save_name)[0] + name + osp.splitext(base_save_name)[1]
            ),
        ) if len(err_dict_sub_bag) != 0 else print(f"No {name}!")

def eval_colmap_results(
    dataset, colmap_output_path, best_index, discard_penalty=False, save_error_path=None
):
    """
    Computes the error using quaternions and translation vector for COLMAP
    """
    assert osp.exists(colmap_output_path)
    # Load visiblity and images
    image_path_list = dataset.img_paths
    f_names_list = dataset.f_names
    calib_list = dataset.calib_paths
    pair_ids = dataset.pair_ids

    # Load camera information
    calib_dict = load_calib(calib_list)

    # Check if colmap results exist. Otherwise, this whole bag is a fail.
    is_colmap_valid = os.path.exists(osp.join(colmap_output_path, str(best_index)))

    if is_colmap_valid:

        # Find the best colmap reconstruction
        # best_index = get_best_colmap_index(cfg)

        print("Computing pose errors")

        """
        num_cores = int(multiprocessing.cpu_count() * 0.9)
        # num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
        result = Parallel(n_jobs=num_cores)(
            delayed(compute_stereo_metrics_from_colmap)(
                image_path_list[pair[0]],
                image_path_list[pair[1]],
                calib_dict[f_names_list[pair[0]]],
                calib_dict[f_names_list[pair[1]]],
                best_index,
                colmap_output_path,
            )
            for pair in tqdm(pair_ids)
        )
        """

        result = [
            compute_stereo_metrics_from_colmap(
                image_path_list[pair[0]],
                image_path_list[pair[1]],
                calib_dict[f_names_list[pair[0]]],
                calib_dict[f_names_list[pair[1]]],
                best_index,
                colmap_output_path,
                use_imc_pose_error_method=False,
            )
            for pair in tqdm(pair_ids)
        ]

    # Collect err_q, err_t from results
    err_dict = {}
    R_error, t_error = [], []

    if discard_penalty:
        for _i in range(len(pair_ids)):
            pair = pair_ids[_i]
            if is_colmap_valid:
                err_q = result[_i][0]
                err_t = result[_i][1]
                if err_q != np.inf and err_t != np.inf:
                    err_dict[f_names_list[pair[0]] + "-" + f_names_list[pair[1]]] = [
                        err_q,
                        err_t,
                    ]
                    R_error.append(err_q)
                    t_error.append(err_t)

    else:
        for _i in range(len(pair_ids)):
            pair = pair_ids[_i]
            if is_colmap_valid:
                err_q = result[_i][0]
                err_t = result[_i][1]
            else:
                err_q = np.inf
                err_t = np.inf
            err_dict[f_names_list[pair[0]] + "-" + f_names_list[pair[1]]] = [
                err_q,
                err_t,
            ]
            R_error.append(err_q)
            t_error.append(err_t)

    # pose auc
    angular_thresholds = [1, 2, 3, 4, 5, 10, 20]
    pose_errors = np.max(np.stack([R_error, t_error]), axis=0)
    aucs = pose_auc(pose_errors, angular_thresholds, True)
    # print(aucs)

    if save_error_path is not None:
        os.makedirs(save_error_path, exist_ok=True)
        file_name_part = ["colmap_pose_error_auc"]
        file_name_part.append("_colmap_err")
        file_name_part.append("_discard_penalty") if discard_penalty else None
        file_name_part.append(".txt")
        file_name = "".join(file_name_part)
        with open(osp.join(save_error_path, file_name), "w",) as f:
            for key in aucs.keys():
                f.write(key + ":\n")
                f.write(f"{aucs[key]}\n")
                f.write("--------------\n")

        # Finally, save packed errors
        save_h5(err_dict, osp.join(save_error_path, "colmap_pose_error.h5"))

    return err_dict, aucs