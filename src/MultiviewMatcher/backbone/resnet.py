import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes//4)
        self.conv2 = conv3x3(planes//4, planes//4, stride=stride)
        self.conv3 = conv1x1(planes//4, planes)
        self.bn1 = nn.BatchNorm2d(planes//4)
        self.bn2 = nn.BatchNorm2d(planes//4)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


block_type = {
    'BasicBlock': BasicBlock,
    'BottleneckBlock': BottleneckBlock
}


class ResNetFPN_8_2(nn.Module):
    """ResNet Feature + FPN, in 8->2 style"""

    def __init__(self, config):
        super().__init__()
        # Config
        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        self.block_dims = block_dims
        self.output_layers = config['output_layers']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # TODO: add an additional layer for `x0_out`? (with small width)
        # TODO: Return dict with stage as key directly.
        feats = [x, x1_out, x2_out, x3_out]
        output_feats = [feats[i] for i in self.output_layers]

        return output_feats

class ResNetFPN_8_1(nn.Module):
    """ResNet Feature + FPN, in 8->1 style"""

    def __init__(self, config):
        super().__init__()
        # Config
        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        self.block_dims = block_dims

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/1
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/2
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/4
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        self.layer1_outconv = conv1x1(initial_dim, block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], initial_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/1
        x2 = self.layer2(x1)  # 1/2
        x3 = self.layer3(x2)  # 1/4
        x4 = self.layer4(x3)  # 1/8

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return [x4_out, x1_out]


class ResNetFPN_4_1(nn.Module):
    """ResNet Feature + FPN, in 4->1 style"""

    def __init__(self, config):
        super().__init__()
        # Config
        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        self.block_dims = block_dims
        self.output_layers = config['output_layers']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/1
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/2
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/4

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/1
        x2 = self.layer2(x1)  # 1/2
        x3 = self.layer3(x2)  # 1/4

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # TODO: add an additional layer for `x0_out`? (with small width)
        # TODO: Return dict with stage as key directly.
        feats = [x, x1_out, x2_out, x3_out]
        output_feats = [feats[i] for i in self.output_layers]

        return output_feats

    def _local_patch_zoomin(self, features, scales=None, sparse=True, sample_points=None, sample_points_bids=None):
        if sparse:
            crop_size = features.shape[-1]
            window_size = self.window_size
            assert window_size <= crop_size

            bids = torch.arange(features.shape[0], device=features.device)
            center = torch.full((features.shape[0], 2), crop_size // 2, device=features.device)
            redius = window_size // 2
            if scales is not None and self.enable_zoomin:
                redius *= scales[:, None]
            boxes = torch.cat([center - redius, center + redius], dim=-1).to(torch.float32) # L*5
            unfold_features = self.roi_align_custom(features, boxes, bids.to(torch.int32))

            # unfold_features = roi_align(features, boxes, output_size=(self.W, self.W), sampling_ratio=1, spatial_scale=scale, aligned=aligned)
        else:
            if sample_points_bids is None:
                bids = torch.zeros((sample_points.shape[0],), device=sample_points.device).long()
            else:
                bids = sample_points_bids

            redius = self.window_size // 2
            if scales is not None:
                redius *= scales[:, None]
            boxes = torch.cat([sample_points - redius, sample_points + redius], dim=-1).to(torch.float32) # L*5
            unfold_features = self.roi_align_custom(features, boxes, bids.to(torch.int32))
        return unfold_features

class ResNetFPN_2_1(nn.Module):
    """ResNet Feature + FPN, in 2->1 style"""

    def __init__(self, config):
        super().__init__()
        # Config
        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        self.block_dims = block_dims
        self.output_layers = config['output_layers']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/1
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/2

        # 3. FPN upsample
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[1])
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/1
        x2 = self.layer2(x1)  # 1/2

        # FPN
        x2_out = self.layer2_outconv(x2)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # TODO: add an additional layer for `x0_out`? (with small width)
        # TODO: Return dict with stage as key directly.
        feats = [x, x1_out, x2_out]
        output_feats = [feats[i] for i in self.output_layers]

        return output_feats

class ResNetFPN_16_4(nn.Module):
    """ResNet Feature + FPN, in 16->4 style"""

    def __init__(self, config):
        super().__init__()
        # Config
        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        self.block_dims = block_dims

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        return [x4_out, x2_out]


# FIXME: REPEAT
class ResNet18C2(nn.Module):
    default_cfg = {
        'block_type': 'BasicBlock',
        'initial_dim': 64,
        'block_dims': [64]
    }
    
    def __init__(self, config):
        super().__init__()
        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        self.block_dims = block_dims
        
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        # NOTE: No downsample between c1 and c2, so the pretrained weight might not function properly.
        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        return x1
    
    
class ResNet18C3(nn.Module):
    default_cfg = {
        'block_type': 'BasicBlock',
        'initial_dim': 64,
        'block_dims': [64, 128]
    }
    
    def __init__(self, config):
        super().__init__()
        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        self.block_dims = block_dims
        
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        # NOTE: No downsample between c1 and c2, so the pretrained weight might not function properly.
        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        return x2
    

@register_model
def resnet18_c2(pretrained=False, in_chans=1, config=ResNet18C2.default_cfg, **kwargs):
    assert in_chans == 1
    model = ResNet18C2(config=config)
    if pretrained:
        assert config == ResNet18C2.default_cfg
        cur_dir = osp.dirname(osp.realpath(__file__))
        checkpoint = torch.load(osp.join(cur_dir, 'resnet18-5c106cde-c2.pth'))
        model.load_state_dict(checkpoint)
    return model


@register_model
def resnet18_c3(pretrained=False, in_chans=1, config=ResNet18C3.default_cfg, **kwargs):
    assert in_chans == 1
    model = ResNet18C3(config=config)
    if pretrained:
        assert config == ResNet18C3.default_cfg
        cur_dir = osp.dirname(osp.realpath(__file__))
        checkpoint = torch.load(osp.join(cur_dir, 'resnet18-5c106cde-c3.pth'))
        model.load_state_dict(checkpoint)
    return model