import copy
from einops import rearrange, repeat
import torch
import torch.nn as nn

from .linear_attention import LinearAttention, FullAttention

class LoFTREncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        attention="linear",
        kernel_fn="elu + 1",
        redraw_interval=1,
        d_kernel=None,
        rezero=None,
        norm_method="layernorm",
    ):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = (
            LinearAttention(
                self.dim,
                kernel_fn=kernel_fn,
                redraw_interval=redraw_interval,
                d_kernel=d_kernel,
            )
            if attention == "linear"
            else FullAttention()
        )
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        if norm_method == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_method == "instancenorm":
            self.norm1 = nn.InstanceNorm1d(d_model)
            self.norm2 = nn.InstanceNorm1d(d_model)
        else:
            raise NotImplementedError

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if rezero is not None:
            self.res_weight = nn.Parameter(torch.Tensor([rezero]), requires_grad=True)
        self.rezero = True if rezero is not None else False

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.dropout1(
            message
        )  # dropout before merging multi-head queried outputs
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.dropout2(message)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message if not self.rezero else x + self.res_weight * message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config["d_model"]  # Feature of query image
        self.nhead = config["nhead"]
        self.layer_names = list(config["layer_names"]) * config["layer_iter_n"]
        self.norm_method = config["norm_method"]
        self.attention_type = config['attention_type']
        if config["redraw_interval"] is not None:
            assert (
                config["redraw_interval"] % 2 == 0
            ), "redraw_interval must be divisible by 2 since each attetnion layer is repeatedly called twice."

        encoder_layer = build_encoder_layer(config)

        module_list = []
        for layer_name in self.layer_names:
            module_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(module_list)

        if config["final_proj"]:
            self.final_proj = nn.Linear(config["d_model"], config["d_model"], bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features_reference_crop, features_query_crop, data, reference_mask=None, query_mask=None):
        """
        Args:
            features_reference_crop: B * n_track * WW * C
            features_query_crop: B * n_track * (n_view - 1) * WW * C
            reference_mask: None
            query_mask: B * n_track * (n_view - 1), feature track mask

        Return:
            features_reference_crop: B * n_track * WW * C
            features_query_crop: B * n_track * (n_view - 1) * WW * C
        """
        self.device = features_reference_crop.device
        B, n_track, n_query, WW = features_query_crop.shape[:4]

        # Track-wise parallel:
        if self.attention_type == 'multiview':
            features_reference_crop = rearrange(features_reference_crop, "b t w c -> (b t) w c")
            features_query_crop = rearrange(features_query_crop, 'b t n w c -> (b t) (n w) c')
            query_mask = repeat(query_mask, 'b t n -> (b t) (n w)', w=WW) if query_mask is not None else None
        else:
            raise NotImplementedError

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == "self":
                src0, src1 = features_reference_crop, features_query_crop
                features_reference_crop, features_query_crop = (
                    layer(features_reference_crop, src0, reference_mask, reference_mask),
                    layer(features_query_crop, src1, query_mask, query_mask),
                )
            elif name == "cross":
                src0, src1 = features_reference_crop, features_query_crop  # [N, L, C], [N, P, C]
                features_query_crop, features_reference_crop = (
                    layer(features_query_crop, src0, query_mask, reference_mask),
                    layer(features_reference_crop, src1, reference_mask, query_mask),
                )
            else:
                raise NotImplementedError

        # Rearrange:
        if self.attention_type == 'multiview':
            features_reference_crop = rearrange(features_reference_crop, "(b t) w c -> b t w c", b=B)
            features_query_crop = rearrange(features_query_crop, '(b t) (n w) c -> b t n w c', b=B, n=n_query)
        else:
            raise NotImplementedError

        return features_reference_crop, features_query_crop

def build_encoder_layer(config):
    if config["type"] == "LoFTR":
        layer = LoFTREncoderLayer(
            config["d_model"],
            config["nhead"],
            config["dropout"],
            config["attention"],
            config["kernel_fn"],
            config["redraw_interval"],
            config["d_kernel"],
            rezero=config["rezero"],
            norm_method=config["norm_method"],
        )
    else:
        raise NotImplementedError
    return layer
