from .resnet import ResNetFPN_8_2, ResNetFPN_8_1, ResNetFPN_4_1, ResNetFPN_2_1, ResNetFPN_16_4
from .S2DNet import s2dnet
from math import log
from functools import lru_cache
from einops.einops import rearrange


def build_backbone(config):
    if config['type'] == 'ResNetFPN':
        if config['resolution'] == [8, 2] or config['resolution'] == [4, 2]:
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == [8, 1]:
            return ResNetFPN_8_1(config['resnetfpn'])
        elif config['resolution'] == [4, 1]:
            return ResNetFPN_4_1(config['resnetfpn'])
        elif config['resolution'] == [2, 1]:
            return ResNetFPN_2_1(config['resnetfpn'])
        elif config['resolution'] == [16, 4]:
            return ResNetFPN_16_4(config['resnetfpn'])
    elif config['type'] == 'S2DNet':
        return s2dnet.S2DNet(config['s2dnet'])
    else:
        raise ValueError("reaching this line! LOFTR_BACKBONE.TYEP and RESOLUTION are not correct")


@lru_cache(maxsize=128)
def _res2ind(resolutions, output_layers):
    """resolutions to indices of feats returned by resfpn"""
    lid2ind = {lid: ind for ind, lid in enumerate(output_layers)}
    inds = [lid2ind[int(log(r, 2))] for r in resolutions]
    return inds


def _get_win_rel_scale(config):
    try:
        min_layer_id = min(config['resnetfpn']['output_layers'])
        rel_scale = int(log(config['resolution'][1], 2)) - min_layer_id
    except KeyError as _:
        min_layer_id = min(config['RESNETFPN']['OUTPUT_LAYERS'])
        rel_scale = int(log(config['RESOLUTION'][1], 2)) - min_layer_id
    return 2**rel_scale


def _get_feat_dims(config):
    layer_dims = [1, *config['resnetfpn']['block_dims']]
    output_layers = config['resnetfpn']['output_layers']
    return [layer_dims[i] for i in output_layers]


def _split_backbone_feats(feats, bs):
    split_feats = [feat.split(bs, dim=0) for feat in feats]
    feats0 = [f[0] for f in split_feats]
    feats1 = [f[1] for f in split_feats]
    return feats0, feats1

def _reshape_backbone_feats(feats, bs):
    feats_reshape = [rearrange(feat, '(b n) c h w -> b n c h w', b=bs) for feat in feats]
    return feats_reshape


def _extract_backbone_feats(feats, config):
    """For backwrad compatibility temporarily."""
    if config['type'] == 'ResNetFPN':
        return feats
        _output_layers = tuple(config['resnetfpn']['output_layers'])
        if len(_output_layers) == 2:
            for r, l in zip(config['resolution'], _output_layers):
                assert 2 ** l == r
            return feats
        else:
            assert NotImplementedError
            return [feats[i] for i in _res2ind(config['resolution'], _output_layers)]
    else:
        return feats
