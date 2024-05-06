import h5py
import os.path as osp


def feature_load(feature_path, img_paths):
    assert osp.exists(feature_path), f"feature path :{feature_path} not exists!"
    feature_dict = {}
    with h5py.File(feature_path, "r") as f:
        for img_path in img_paths:
            feature_dict[img_path] = {key: value.__array__() for key, value in f[img_path].items()}
    return feature_dict


def feature_save(feature_dict, feature_path):
    with h5py.File(feature_path, "w") as f:
        for key, value in feature_dict.items():
            grp = f.create_group(key)
            for key_sub, value_sub in value.items():
                assert "/" not in key_sub
                grp.create_dataset(key_sub, data=value_sub)
