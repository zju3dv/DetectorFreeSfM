import natsort
from loguru import logger

def compare(img_name):
    key = img_name.split('/')[-1].split('.')[0]
    return int(key)


def pairs_from_index(img_lists, num_match=5, gap=1):
    """Get covis images by image id."""
    pairs = []
    img_lists = natsort.natsorted(img_lists)
    img_ids = range(len(img_lists))

    for i in img_ids:
        count = 0
        j = i + 1
        
        while j < len(img_ids) and count < num_match:
            if (j - i) % gap == 0:
                count += 1
                pairs.append(" ".join([img_lists[i], img_lists[j]])) 
            j += 1
    
    logger.info(f"Total:{len(pairs)} pairs")
    return pairs