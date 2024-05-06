import os.path as osp
import numpy as np
import cv2
import random
from src.utils.colmap.database import COLMAPDatabase

def import_features_and_matches(output_db_path, image_list, image_pairs, keypoints_dict, matches_dict, intrin_prior_path, colmap_cfgs=None, shuffle_image_list=True):
    """
    Make database and import keypoints and intrinsics to db
    """
    db = COLMAPDatabase.connect(output_db_path)
    db.create_tables()

    assert osp.exists(intrin_prior_path)
    single_camera = False
    if colmap_cfgs is not None and "ImageReader_single_camera" in colmap_cfgs:
        if colmap_cfgs["ImageReader_single_camera"]:
            single_camera = True
    
    if shuffle_image_list:
        random.shuffle(image_list)
    
    imgname2id = {}
    for id, image_path in enumerate(image_list):
        imgname2id[image_path] = id + 1
        img_name = osp.basename(image_path)
        img_base_name = osp.splitext(img_name)[0]

        if single_camera:
            if id == 0:
                assert osp.isfile(intrin_prior_path), f"single_camera is switched, however given a intrin directory"
                K = np.loadtxt(intrin_prior_path) # 3*3

                fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]
                image = cv2.imread(image_path)
                h, w, _ = image.shape

                db.add_camera(1, w, h, np.array((fx, fy, cx, cy)), camera_id=1)
        else:
            assert osp.isdir(intrin_prior_path), f"Provided intrinsics path is not a directory! You need to switch single_camera for providing only one intrinsic file "
            intrin_prior_file_path = osp.join(intrin_prior_path, img_base_name+'.txt')
            K = np.loadtxt(intrin_prior_file_path)

            fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]
            image = cv2.imread(image_path)
            h, w, _ = image.shape

            db.add_camera(1, w, h, np.array((fx, fy, cx, cy)), camera_id=id+1)
        
        # Import keypoints
        keypoints = keypoints_dict[img_name] # N*2
        db.add_image(name=img_name, camera_id=1 if single_camera else id+1, image_id=id+1)

        keypoints += 0.5
        db.add_keypoints(image_id=id+1, keypoints=keypoints)
    
    for i, img_pair in enumerate(image_pairs):
        img0_path, img1_path = img_pair.split(" ")
        img0_id, img1_id = imgname2id[img0_path], imgname2id[img1_path]
        img0_name = osp.basename(img0_path)
        img1_name = osp.basename(img1_path)

        # Load matches
        key = " ".join([img0_name, img1_name])
        matches = np.squeeze(matches_dict[key]).T # N*2
        if matches.ndim != 2:
            # No match scenario
            matches = np.empty((0,2))

        db.add_matches(img0_id, img1_id, matches)

    db.commit()
    db.close()

def set_single_camera(database_path):
    db = COLMAPDatabase.connect(database_path)

    rows = db.execute("SELECT camera_id FROM cameras")
    camera_ids = [id[0] for id in rows]
    if len(camera_ids) == 1:
        pass
    else:
        for camera_id in camera_ids:
            if camera_id == 1:
                row = db.execute(f"SELECT * FROM cameras WHERE camera_id = {camera_ids[0]}")
                info = next(row)
            else:
                db.execute(f"DELETE FROM cameras WHERE camera_id = {camera_id}")


    # Update camera ids in images
    for (image_id,) in db.execute("SELECT image_id FROM images;"):
        db.execute(f"UPDATE IMAGES SET camera_id = {1} WHERE image_id = {image_id}")

    db.commit()
    db.close()

def load_intrin_to_database(output_db_path, intrin_prior_path, colmap_cfgs=None):
    assert osp.exists(intrin_prior_path)
    single_camera = False
    if colmap_cfgs is not None and "ImageReader_single_camera" in colmap_cfgs:
        if colmap_cfgs["ImageReader_single_camera"]:
            single_camera = True

    db = COLMAPDatabase.connect(output_db_path)
    # Check num of camera:
    rows = db.execute("SELECT camera_id FROM cameras")
    camera_ids = [id[0] for id in rows]
    if len(camera_ids) == 1:
        assert osp.isfile(intrin_prior_path) and single_camera, f"single_camera is switched, however given a intrin directory"

        row = db.execute(f"SELECT width, height FROM cameras WHERE camera_id = {camera_ids[0]}")
        w, h = next(row)
        db.execute(f"DELETE FROM cameras WHERE camera_id = {camera_ids[0]}")

        K = np.loadtxt(intrin_prior_path) # 3*3
        fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]

        db.add_camera(1, w, h, np.array((fx, fy, cx, cy)), camera_id=camera_ids[0])

    else:
        # Load image name, camera id from images:
        for image_name, camera_id in db.execute("SELECT name, camera_id FROM images"):
            # Delete camera:
            row = db.execute(f"SELECT width, height FROM cameras WHERE camera_id = {camera_id}")
            w, h = next(row)
            db.execute(f"DELETE FROM cameras WHERE camera_id = {camera_id}")

            # Then add new camera:
            img_base_name = osp.splitext(osp.basename(image_name))[0]
            assert osp.isdir(intrin_prior_path), f"Provided intrinsics path is not a directory! You need to switch single_camera for providing only one intrinsic file "

            intrin_prior_file_path = osp.join(intrin_prior_path, img_base_name+'.txt')
            K = np.loadtxt(intrin_prior_file_path)
            fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]

            db.add_camera(1, w, h, np.array((fx, fy, cx, cy)), camera_id=camera_id)

    db.commit()
    db.close()