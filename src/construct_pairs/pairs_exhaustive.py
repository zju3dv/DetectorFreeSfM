import numpy as np
from loguru import logger
from itertools import combinations

def exhaustive_all_pairs(img_list):
    pair_ids = list(combinations(range(len(img_list)), 2))
    img_pairs = []
    for pair_id in pair_ids:
        img_pairs.append(" ".join([img_list[pair_id[0]], img_list[pair_id[1]]]))

    logger.info(f"Total:{len(img_pairs)} pairs")
    return img_pairs