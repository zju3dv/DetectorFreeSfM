import os.path as osp
import numpy as np
from itertools import combinations
from loguru import logger
import pycolmap
from typing import List
from src.utils.colmap.read_write_model import read_model
from src.utils.metric_utils import (
    pose_auc,
    qvec2rotmat,
    evaluate_R_t,
    angle_error_mat,
    angle_error_vec,
)
from tqdm import tqdm
from pathlib import Path
import subprocess

def eval_multiview(tool_path: Path, ply_path: Path, scan_path: Path,
                   tolerances: List[float]):
    if not tool_path.exists():
        raise FileNotFoundError(
            f"Cannot find the evaluation executable at {tool_path}; "
            "Please install it from "
            "https://github.com/ETH3D/multi-view-evaluation")

    cmd = [
        str(tool_path),
        '--reconstruction_ply_path', str(ply_path),
        '--ground_truth_mlp_path', str(scan_path),
        '--tolerances', ",".join(map(str, tolerances))
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, err = p.communicate()
    lines = out.decode().split("\n")
    accuracy = None
    completeness = None
    for line in lines:
        if line.startswith("Accuracies"):
            accuracy = list(map(
                float, line.replace("Accuracies: ", "").split(" ")))
        if line.startswith("Completenesses"):
            completeness = list(map(
                float, line.replace("Completenesses: ", "").split(" ")))
    assert accuracy is not None and len(accuracy) == len(tolerances)
    assert completeness is not None and len(completeness) == len(tolerances)
    accuracy_dict = {}
    for i, tolerance in enumerate(tolerances):
        accuracy_dict[f'accuracy@{tolerance}'] = accuracy[i]
    completeness_dict = {}
    for i, tolerance in enumerate(tolerances):
        completeness_dict[f'completeness@{tolerance}'] = completeness[i]
    return {"accuracy": accuracy_dict, "completeness": completeness_dict}

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
    thresholds=[1, 3, 5, 10, 20],
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
                save_dir,
                osp.splitext(base_save_name)[0]
                + name
                + osp.splitext(base_save_name)[1],
            ),
        ) if len(err_dict_sub_bag) != 0 else print(f"No {name}!")

class Evaluator:
    def __init__(
        self,
        image_list,
        pose_gt_path,
        parallel_eval=False,
        discard_nonrig_penality=False,
        ignore_failed_scenes=False,
        # ignore_failed_scenes=True,
        triangulate_mode=False,
        verbose=False
    ) -> None:
        self.parallel_eval = parallel_eval
        self.discard_nonrig_penality = discard_nonrig_penality
        self.verbose = verbose
        self.image_list = image_list
        self.ignore_failed_scenes = ignore_failed_scenes
        self.angular_thresholds = [1, 3, 5, 10, 20]

        # Evaluate point cloud accuracy
        self.triangulate_mode = triangulate_mode
        self.multiview_eval_tool_path = "third_party/multi-view-evaluation/build/ETH3DMultiViewEvaluation"
        self.gt_scan_path = osp.join(osp.dirname(pose_gt_path), 'dslr_scan_eval', 'scan_alignment.mlp')
        self.tolerances = [0.01, 0.02, 0.05]
        self.max_align_error = 1.0

        # Construct eval stereo pair:
        self.eval_pairs = self.construct_eval_pairs(
            image_list
        )  # [[img_name0 img_name1]]

        # Metric buffer: (used to compare metric after each iterative)
        self.metric_buffer = []

        # Load gt poses:
        assert osp.exists(pose_gt_path), f"Pose gt:{pose_gt_path} not exists!"
        self.pose_gt = {}  # img_name: pose_gt
        self.pose_gt_base_name = {} # img_base_name: pose_gt
        self.camera_center = {}
        for image_path in image_list:
            img_base = osp.splitext(osp.basename(image_path))[0]
            pose_path = osp.join(pose_gt_path, img_base + ".txt")
            pose = np.loadtxt(pose_path)
            self.pose_gt[image_path] = pose  # w2c pose
            self.pose_gt_base_name[osp.basename(image_path)] = pose
            self.camera_center[osp.basename(image_path)] = np.linalg.inv(pose)[:3, 3] # [3]

    def construct_eval_pairs(self, img_list):
        pair_ids = list(combinations(range(len(img_list)), 2))
        img_pairs = []
        for pair_id in pair_ids:
            img_pairs.append([img_list[pair_id[0]], img_list[pair_id[1]]])

        return img_pairs

    def compute_stereo_metrics_from_colmap(
        self,
        img_name1,
        img_name2,
        all_images_dict,
        use_imc_pose_error_method=False,
    ):
        """Computes (pairwise) error metrics from Colmap results."""

        # Load COLMAP dR and dt
        if osp.basename(img_name1) in all_images_dict:
            image1 = all_images_dict[osp.basename(img_name1)]
            R_1_actual = qvec2rotmat(image1.qvec)
            t_1_actual = image1.tvec
        else:
            R_1_actual, t_1_actual = None, None
        if osp.basename(img_name2) in all_images_dict:
            image2 = all_images_dict[osp.basename(img_name2)]
            R_2_actual = qvec2rotmat(image2.qvec)
            t_2_actual = image2.tvec
        else:
            R_2_actual, t_2_actual = None, None

        # Load gt pose:
        gt_pose1 = self.pose_gt[img_name1] # 4*4
        gt_pose2 = self.pose_gt[img_name2]

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
            R_1, t_1 = gt_pose1[:3,:3], gt_pose1[:3,[3]]
            R_2, t_2 = gt_pose2[:3,:3], gt_pose2[:3,[3]]

            # Compute ground truth dR, dt
            dR = np.dot(R_2, R_1.T)
            dt = t_2 - np.dot(dR, t_1)  # (3,1)

            # Save err_, err_t
            if use_imc_pose_error_method:
                err_q, err_t = evaluate_R_t(dR, dt, dR_act, dt_act)  # rad!
            else:
                err_q = angle_error_mat(dR_act, dR)  # err_R actually
                dt = dt.flatten()
                dt_act = dt_act.flatten()
                err_t = angle_error_vec(dt_act, dt)  # degree!
        return err_q, err_t
    
    def load_model(self, colmap_model):
        state = True
        if colmap_model is None:
            state = False
        elif isinstance(colmap_model, str):
            if not osp.exists(colmap_model):
                state = False
            else:
                cameras, images, points = read_model(colmap_model)
        elif isinstance(colmap_model, list):
            assert len(colmap_model) == 3
            cameras, images, points = colmap_model
        else:
            raise NotImplementedError

        if state == True:
            return [cameras, images, points]
        else:
            return [state]
    
    def prepare_output_from_buffer(self):
        assert len(self.metric_buffer) != 0, "Empty metric buffer! You are supposed to eval metric firstly"
        if len(self.metric_buffer) == 1:
            return self.metric_buffer[0]
        else:
            metric_total_dict = {}
            for id, metric_dict in enumerate(self.metric_buffer):
                metric_total_dict.update({metric_name+f"_{id}": metric for metric_name, metric in metric_dict.items()})
            return metric_total_dict

    def eval_metric(self, colmap_model=None):
        if self.triangulate_mode:
            return self.eval_pointcloud_metric(colmap_model)
        else:
            return self.eval_pose_metric(colmap_model)
    
    def eval_pointcloud_metric(self, colmap_model=None):
        if osp.exists(colmap_model):
            loaded_model = pycolmap.Reconstruction(colmap_model)
            ply_path = Path(colmap_model) / 'reconstruction.ply'
            loaded_model.export_PLY(ply_path)
            results = eval_multiview(
                Path(self.multiview_eval_tool_path), Path(ply_path),
                Path(self.gt_scan_path),
                tolerances=self.tolerances)

            self.metric_buffer.append(results)

            if self.verbose:
                print(results)
            return None, results
        else:
            return None, None

    def eval_pose_metric(self, colmap_model=None):
        """
        colmap_model: path(str) or list[images, cameras, points]
        """
        failed_recon = False
        loaded_model = self.load_model(colmap_model)
        if len(loaded_model) == 3:
            cameras, images, points = loaded_model
        elif len(loaded_model) == 1:
            failed_recon = True
            images = []
        else:
            raise NotImplementedError
        

        if not failed_recon:
            # Convert images to {img_name: image}
            img_name2img = {}
            for image in images.values():
                img_name2img[image.name] = image

            if self.parallel_eval:
                raise NotImplementedError
            else:
                iterator = tqdm(self.eval_pairs) if self.verbose else self.eval_pairs
                result = [
                    self.compute_stereo_metrics_from_colmap(
                        pair[0],
                        pair[1],
                        img_name2img,
                        use_imc_pose_error_method=False,
                    )
                    for pair in iterator
                ]

            # Collect err_q, err_t from results
            err_dict = {}
            R_error, t_error = [], []

            for id, pair in enumerate(self.eval_pairs):
                err_q, err_t = result[id]
                if err_q != np.inf and err_t != np.inf:
                    err_dict["-".join(pair)] = [
                        err_q,
                        err_t,
                    ]
                    R_error.append(err_q)
                    t_error.append(err_t)
                else:
                    if not self.discard_nonrig_penality:
                        err_dict["-".join(pair)] = [
                            err_q,
                            err_t,
                        ]
                        R_error.append(err_q)
                        t_error.append(err_t)

            pose_errors = np.max(np.stack([R_error, t_error]), axis=0)

        else:
            if self.ignore_failed_scenes:
                logger.warning(f"Failed bags ignored!")
                self.metric_buffer.append({})
                return None, {}
            else:
                pose_errors = []
                err_dict = []

        # pose auc:
        aucs = pose_auc(pose_errors, self.angular_thresholds, ret_dict=True)

        metric_dict = {
            "aucs": aucs,
        }

        self.metric_buffer.append(metric_dict)

        return err_dict, metric_dict
