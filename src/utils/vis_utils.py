import numpy as np
import os.path as osp
from loguru import logger
from src.colmap.read_write_model import qvec2rotmat
from src.utils.colmap.read_write_model import read_model
from src.post_optimization.utils.geometry_utils import convert_pose2T

def save_colmap_ws_to_vis3d(colmap_dir, save_path, name_prefix=''):
    from wis3d.wis3d import Wis3D
    if not osp.exists(colmap_dir):
        logger.warning(f"{colmap_dir} not exists!")
        return
    cameras, images, points3D = read_model(colmap_dir)
    save_path, name = save_path.rsplit('/',1)
    wis3d = Wis3D(save_path, name)

    # Point cloud:
    coord3D = []
    color = []
    for point3D in points3D.values():
        coord3D.append(point3D.xyz)
        color.append(point3D.rgb)
    if len(coord3D) == 0:
        logger.warning(f"Empty point cloud in {colmap_dir}")
    else:
        coord3D = np.stack(coord3D)
        color = np.stack(color)
        wis3d.add_point_cloud(coord3D, color, name= f'point_cloud_{name_prefix}')

    # Camera tragetory:
    for id, image in images.items():
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        T = convert_pose2T([R, t])
        T = pose_cvt(T)
        T_inv = np.linalg.inv(T)
        wis3d.add_camera_trajectory(T_inv[None], name='poses{:0>3d}_'.format(id) + name_prefix)

def pose_cvt(T_cv, scale=1):
    """
    Convert transformation from CV representation
    Input: 4x4 Transformation matrix
    Output: 4x4 Converted transformation matrix
    """
    R = T_cv[:3, :3]
    t = T_cv[:3, 3]

    R_rot = np.eye(3)
    R_rot[1, 1] = -1
    R_rot[2, 2] = -1

    R = np.matmul(R_rot, R)
    t = np.matmul(R_rot, t)

    t *= scale

    T_cg = np.eye(4)
    T_cg[:3, :3] = R
    T_cg[:3, 3] = t
    return T_cg