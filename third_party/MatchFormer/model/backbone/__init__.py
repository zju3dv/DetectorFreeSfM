from .match_LA_lite import Matchformer_LA_lite
from .match_LA_large import Matchformer_LA_large
from .match_SEA_lite import Matchformer_SEA_lite
from .match_SEA_large import Matchformer_SEA_large


def build_backbone(config):
    if config['backbone_type'] == 'litela':
        return Matchformer_LA_lite()
    elif config['backbone_type'] == 'largela':
        return Matchformer_LA_large()
    elif config['backbone_type'] == 'litesea':
        return Matchformer_SEA_lite()
    elif config['backbone_type'] == 'largesea':
        return Matchformer_SEA_large()    
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")