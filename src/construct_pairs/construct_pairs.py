from loguru import logger
import ray

from .pairs_exhaustive import exhaustive_all_pairs
from .pairs_from_img_index import pairs_from_index

@ray.remote(num_cpus=2, num_gpus=0.1, max_calls=1)
def construct_img_pairs_ray_wrapper(*args, **kwargs):
    return construct_img_pairs(*args, **kwargs)

def construct_img_pairs(img_list, args, strategy='exhaustive', pair_path=None, verbose=True):
    # Construct image pairs:
    logger.info(f'Using {strategy} matching build pairs') if verbose else None
    if strategy == 'exhaustive':
        img_pairs = exhaustive_all_pairs(img_list)
    elif strategy == 'pair_from_index':
        img_pairs = pairs_from_index(img_list, args.INDEX_num_of_pair)
    else:
        raise NotImplementedError
    
    return img_pairs
