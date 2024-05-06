import numpy as np
import os
from loguru import logger
from .colmap.read_write_model import qvec2rotmat, read_images_binary
from .colmap.eval_helper import quaternion_from_matrix

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython

        IPython.embed()

    return err_q, err_t

def calc_trans_error(model, data):
    alignment_error = model - data
    sqrt_val = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error),
                              0))
    return sqrt_val

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def compute_stereo_metrics_from_colmap(
    img1,
    img2,
    calib1,
    calib2,
    best_index,
    colmap_output_path,
    use_imc_pose_error_method="False",
):
    """Computes (pairwise) error metrics from Colmap results."""

    # Load COLMAP dR and dt

    # First read images.bin for the best reconstruction
    images_bin = read_images_binary(
        os.path.join(colmap_output_path, str(best_index), "images.bin")
    )

    # For each key check if images_bin[key].name = image_name
    R_1_actual, t_1_actual = None, None
    R_2_actual, t_2_actual = None, None
    for key in images_bin.keys():
        if images_bin[key].name == os.path.basename(img1):
            R_1_actual = qvec2rotmat(images_bin[key].qvec)
            t_1_actual = images_bin[key].tvec
        if images_bin[key].name == os.path.basename(img2):
            R_2_actual = qvec2rotmat(images_bin[key].qvec)
            t_2_actual = images_bin[key].tvec

    # Compute err_q and err_t only when R, t are not None
    err_q, err_t = np.inf, np.inf
    if (
        (R_1_actual is not None)
        and (R_2_actual is not None)
        and (t_1_actual is not None)
        and (t_2_actual is not None)
    ):
        # Compute dR, dt (actual)
        dR_act = np.dot(R_2_actual, R_1_actual.T)
        dt_act = t_2_actual - np.dot(dR_act, t_1_actual)

        # Get R, t from calibration information
        R_1, t_1 = calib1["R"], calib1["T"].reshape((3, 1))
        R_2, t_2 = calib2["R"], calib2["T"].reshape((3, 1))

        # Compute ground truth dR, dt
        dR = np.dot(R_2, R_1.T)
        dt = t_2 - np.dot(dR, t_1)  # (3,)

        # Save err_, err_t
        if use_imc_pose_error_method:
            err_q, err_t = evaluate_R_t(dR, dt, dR_act, dt_act)  # rad!
        else:
            err_q = angle_error_mat(dR_act, dR)  # err_R actually
            dt = dt.flatten()
            dt_act = dt_act.flatten()
            err_t = angle_error_vec(dt_act, dt)  # degree!
    return err_q, err_t


def pose_auc(errors, thresholds, ret_dict=False):
    if len(errors) == 0:
        aucs = [0 for i in thresholds]
    else:
        sort_idx = np.argsort(errors)
        errors = np.array(errors.copy())[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0.0, errors]
        recall = np.r_[0.0, recall]
        aucs = []
        for t in thresholds:
            last_index = np.searchsorted(errors, t)
            r = np.r_[recall[:last_index], recall[last_index - 1]]
            e = np.r_[errors[:last_index], t]
            aucs.append(np.trapz(r, x=e) / t)
    if ret_dict:
        return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}
    else:
        return aucs

# Evaluate query pose errors
def query_pose_error(pose_pred, pose_gt):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance

def aggregate_metrics(metrics, thres=[1, 3, 5]):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4
    """
    R_errs = metrics["R_errs"]
    t_errs = metrics["t_errs"]

    degree_distance_metric = {}
    for threshold in thres:
        degree_distance_metric[f"{threshold}cm@{threshold}degree"] = np.mean(
            (np.array(R_errs) < threshold) & (np.array(t_errs) < threshold)
        )
    return degree_distance_metric

def add_metric(metric_all_dict, scene_metric):
    for metric_name, metric in scene_metric.items():
        if metric_name in metric_all_dict:
            metric_all_dict[metric_name].append(metric)
        else:
            metric_all_dict[metric_name] = [metric]
        
    return metric_all_dict

def average_metric(metric_dict, total_num_constrain=None):
    """
    metric_all_dict: {metric_name: [metrics]}
    total_num_constrin: used to prevent the unfair average, e.g., some metric have fewer number
    """
    metric_avg = {}
    for metric_name, metric_list in metric_dict.items():
        if len(metric_list) == 0:
            pass
        elif isinstance(metric_list[0], dict):
            sub_metrics = {}
            for metric in metric_list:
                for sub_metric_name, metric_value in metric.items():
                    if sub_metric_name in sub_metrics:
                        sub_metrics[sub_metric_name].append(metric_value)
                    else:
                        sub_metrics[sub_metric_name] = [metric_value]
            metric_avg[metric_name] = average_metric(sub_metrics, total_num_constrain=total_num_constrain)
        else:
            metric_avg[metric_name] = np.mean(np.array(metric_list))
            if total_num_constrain is not None:
                if len(metric_list) != total_num_constrain:
                    logger.warning(f"metric average number constrain violated, constrain: {total_num_constrain}, however metric: {metric_name} only have: {len(metric_list)} metrics")

    return metric_avg

def output_metric(metric_dict, bag_name=None):
    output_str = []
    if bag_name is not None:
        assert isinstance(bag_name, str)
        output_str += ["==================================="]
        print("===================================")
        output_str += [f'Metrics of: {bag_name}']
        print(f'Metrics of: {bag_name}')

    for metric_name, value in metric_dict.items():
        if isinstance(value, dict):
            output_str += [f"*******************"]
            print(f"*******************")

            output_str += [f"{metric_name:}"]
            print(f"{metric_name:}")
            
            sub_str = output_metric(value)
            output_str += sub_str

            output_str += [f"*******************"]
            print(f"*******************")
        else:
            output_str += [f"{metric_name}: {value}"]
            print(f"{metric_name}: {value}")
    
    return output_str

def aggregate_multi_scene_metrics(metric_dict, dataset_name='IMC', verbose=True, output_path=None):
    import ipdb; ipdb.set_trace()
    metric_all = {} # metric_name: [metrics]

    if ("IMC" in dataset_name) or ('onepose_bag_dataset' in dataset_name):
       scene_bag = {"3bag": {}, "5bag": {}, "10bag": {}, "25bag": {}}
    elif dataset_name in ['onepose_dataset', 'onepose_sparse_dataset', 'scannet_dataset', 'eth3d_dataset'] or 'eth3d' in dataset_name or 'dtu' in dataset_name or 'tnt' in dataset_name:
        scene_bag = {obj_name : {} for obj_name in metric_dict.keys()}
    else:
        scene_bag = {obj_name : {} for obj_name in metric_dict.keys()}

    # Aggregate metric
    for scene_name, scene_metric in metric_dict.items():
        metric_all = add_metric(metric_all, scene_metric)

        if ("IMC" in dataset_name) or ('onepose_bag_dataset' in dataset_name):
            if "3bag" in scene_name:
                scene_bag["3bag"] = add_metric(scene_bag["3bag"], scene_metric)
            if "5bag" in scene_name and "25bag" not in scene_name:
                scene_bag["5bag"] = add_metric(scene_bag["5bag"], scene_metric)
            elif "10bag" in scene_name:
                scene_bag["10bag"] = add_metric(scene_bag["10bag"], scene_metric)
            elif "25bag" in scene_name:
                scene_bag["25bag"] = add_metric(scene_bag["25bag"], scene_metric)
        elif dataset_name in ['onepose_dataset', 'onepose_sparse_dataset', 'eth3d_dataset', 'scannet_dataset'] or 'eth3d' in dataset_name or 'dtu' in dataset_name:
            scene_bag[scene_name] = add_metric(scene_bag[scene_name], scene_metric)
        else:
            scene_bag[scene_name] = add_metric(scene_bag[scene_name], scene_metric)
    
    # Average metric:
    metric_all_avg = average_metric(metric_all, total_num_constrain=len(metric_dict))
    scene_bag_metric_avg = {}
    for bag_name, bag_metric in scene_bag.items():
        scene_bag_metric_avg[bag_name] = average_metric(bag_metric)

    if verbose:
        output_str = []
        output_str += output_metric(metric_all_avg, bag_name="all scene")

        for bag_name, bag_metric in scene_bag_metric_avg.items():
            output_str += output_metric(bag_metric, bag_name=bag_name)
        
        if output_path is not None:
            with open(output_path, 'w') as f:
                for line in output_str:
                    f.write(line)
                    f.write('\n')

    return metric_all_avg

def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall

def compute_auc(errors, thresholds, min_error=None):
    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e)/t
        aucs.append(auc*100)
    return aucs