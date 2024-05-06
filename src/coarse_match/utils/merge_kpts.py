import numpy as np
from loguru import logger

def agg_groupby_2d(keys, vals, agg='avg'):
    """
    Args:
        keys: (N, 2) 2d keys
        vals: (N,) values to average over
        agg: aggregation method
    Returns:
        dict: {key: agg_val}
    """
    assert agg in ['avg', 'sum']
    unique_keys, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group_sums = np.bincount(group, weights=vals)
    values = group_sums if agg == 'sum' else group_sums / counts
    return dict(zip(map(tuple, unique_keys), values))

class Match2Kpts(object):
    """extract all possible keypoints for each image from all image-pair matches"""
    def __init__(self, matches, names, name_split='-', cov_threshold=0):
        self.names = names
        self.matches = matches
        self.cov_threshold = cov_threshold
        self.name2matches = {name: [] for name in names}
        for k in matches.keys():
            try:
                name0, name1 = k.split(name_split)
            except ValueError as _:
                name0, name1 = k.split('-')
            self.name2matches[name0].append((k, 0))
            self.name2matches[name1].append((k, 1))
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            name = self.names[idx]
            kpts = np.concatenate([self.matches[k][:, [2*id, 2*id+1, 4]]
                        for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold], 0)
            return name, kpts
        elif isinstance(idx, slice):
            names = self.names[idx]
            try:
                kpts = [np.concatenate([self.matches[k][:, [2*id, 2*id+1, 4]]
                            for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold], 0) for name in names]
            except:
                kpts = []
                for name in names:
                    kpt = [self.matches[k][:, [2*id, 2*id+1, 4]]
                            for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold]
                    if len(kpt) != 0:
                        kpts.append(np.concatenate(kpt,0))
                    else:
                        kpts.append(np.empty((0,3)))
                        logger.warning(f"no keypoints in image:{name}")
            return list(zip(names, kpts))
        else:
            raise TypeError(f'{type(self).__name__} indices must be integers')