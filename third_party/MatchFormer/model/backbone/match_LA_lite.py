import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,H,W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, eps=1e-6 , cross = False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.cross = cross
        self.feature_map = elu_feature_map
        self.eps = eps
        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x): 
        x_q, x_kv = x, x
        B, N, C = x_q.shape
        MiniB = B // 2
        query = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 1, 2, 3)
        kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)

        if self.cross == True:
            k1, k2 = kv[0].split(MiniB)
            v1, v2 = kv[1].split(MiniB)
            key = torch.cat([k2, k1], dim=0)
            value = torch.cat([v2, v1], dim=0)
        else:
            key, value = kv[0], kv[1]  
            
        Q = self.feature_map(query)
        K = self.feature_map(key)
        v_length = value.size(1)
        value = value / v_length
        KV = torch.einsum("nshd,nshv->nhdv", K, value)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        x = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        x = x.contiguous().view(B, -1, C)
        
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,cross = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, cross= cross)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x,H,W):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm(x),H,W))
        
        return x

class Positional(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, with_pos=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = Positional(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        _, _, H, W = x.shape       
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W

class AttentionBlock(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dims=128, patch_size=7, num_heads=1, mlp_ratios=4,
                 qkv_bias=True, drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),stride=2, depths=1, cross=[False,False,True]):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride = stride, in_chans=in_chans,
                                              embed_dim=embed_dims)
        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            drop=drop_rate, drop_path=0, norm_layer=norm_layer, cross=cross[i])
            for i in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, H, W  = self.patch_embed(x)
        for i, blk in enumerate(self.block):
            x = blk(x,H,W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

class Matchformer_LA_lite(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dims=[128, 192, 256, 512], num_heads=[8, 8, 8, 8],
                stage1_cross = [False,False,True],stage2_cross = [False,False,True],stage3_cross = [False,True,True],stage4_cross = [False,True,True]):
        super().__init__()
        #Attention
        self.AttentionBlock1 =  AttentionBlock(img_size=img_size // 2, patch_size=7, num_heads= num_heads[0], mlp_ratios=4,  in_chans=in_chans,
                                              embed_dims=embed_dims[0],stride=4,depths=3, cross=stage1_cross)
        self.AttentionBlock2 =  AttentionBlock(img_size=img_size // 4, patch_size=3, num_heads= num_heads[1], mlp_ratios=4,  in_chans=embed_dims[0],
                                              embed_dims=embed_dims[1],stride=2,depths=3, cross=stage2_cross)
        self.AttentionBlock3 =  AttentionBlock(img_size=img_size // 16,patch_size=3, num_heads= num_heads[2], mlp_ratios=4,  in_chans=embed_dims[1],
                                              embed_dims=embed_dims[2],stride=2,depths=3, cross=stage3_cross)
        self.AttentionBlock4 =  AttentionBlock(img_size=img_size // 32,patch_size=3, num_heads= num_heads[3], mlp_ratios=4,  in_chans=embed_dims[2],
                                              embed_dims=embed_dims[3],stride=2,depths=3, cross=stage4_cross)
                                            
        #FPN
        self.layer4_outconv = conv1x1(embed_dims[3], embed_dims[3])
        self.layer3_outconv = conv1x1(embed_dims[2], embed_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(embed_dims[3], embed_dims[3]),
            nn.BatchNorm2d(embed_dims[3]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[3], embed_dims[2]),
        )

        self.layer2_outconv = conv1x1(embed_dims[1], embed_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(embed_dims[2], embed_dims[2]),
            nn.BatchNorm2d(embed_dims[2]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[2], embed_dims[1]),
        )
        self.layer1_outconv = conv1x1(embed_dims[0], embed_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(embed_dims[1], embed_dims[1]),
            nn.BatchNorm2d(embed_dims[1]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[1], embed_dims[0]),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # stage 1 # 1/4        
        x = self.AttentionBlock1(x)
        out1 = x
        # stage 2 # 1/8
        x = self.AttentionBlock2(x)     
        out2 = x                    
        # stage 3 # 1/16
        x = self.AttentionBlock3(x)                          
        out3 = x
        # stage 3 # 1/32
        x = self.AttentionBlock4(x)                          
        out4 = x 
        
        #FPN
        c4_out = self.layer4_outconv(out4)
        _,_,H,W = out3.shape
        c4_out_2x = F.interpolate(c4_out, size =(H,W), mode='bilinear', align_corners=True)
        c3_out = self.layer3_outconv(out3)
        _,_,H,W = out2.shape
        c3_out = self.layer3_outconv2(c3_out +c4_out_2x)
        c3_out_2x = F.interpolate(c3_out, size =(H,W), mode='bilinear', align_corners=True)
        c2_out = self.layer2_outconv(out2)
        _,_,H,W = out1.shape
        c2_out = self.layer2_outconv2(c2_out +c3_out_2x)
        c2_out_2x = F.interpolate(c2_out, size =(H,W), mode='bilinear', align_corners=True)
        c1_out = self.layer1_outconv(out1)
        c1_out = self.layer1_outconv2(c1_out+c2_out_2x)
        
        return c2_out,c1_out