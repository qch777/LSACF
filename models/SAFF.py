"""
# File       : SAFF.py
# Time       ：2023/7/18 18:43
# Author     ：qch
# version    ：python 3.8
# Description：
"""

from models.ops.modules.ms_deform_attn import MSDeformAttn
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SAFF(nn.Module):
    def __init__(self, n_levels=5, n_heads=8, n_points=4, n_layer=1):
        super().__init__()
        self.d_model = 256
        self.dropout = 0.5
        self.d_ffn = 1024
        self.activation = "relu"
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.reference_points = nn.Linear(self.d_model, self.n_levels * 2)
        self.attn = MSDeformAttn(self.d_model, n_levels, n_heads, n_points)
        self.layer = n_layer
        self.gn = nn.GroupNorm(32, 256)
        self.conv1 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(2048, 256, kernel_size=3)
        self.conv11 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv22 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv33 = nn.Conv2d(256, 1024, kernel_size=1)

        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # ffn
        self.linear1 = nn.Linear(self.d_model, self.d_ffn)
        self.activation = _get_activation_fn(self.activation)
        self.dropout2 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.d_ffn, self.d_model)
        self.dropout3 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, x):
        bs, _, h1, w1 = x[0].shape
        bs, _, h2, w2 = x[1].shape
        bs, _, h3, w3 = x[2].shape
        bs, _, h4, w4 = x[3].shape
        srcs5 = self.gn(self.conv4(x[3]))
        bs, _, h5, w5 = srcs5.shape
        # print(srcs5.shape)

        srcs = []
        src_flatten = []
        spatial_shapes = []
        srcs.append(self.gn(self.conv1(x[0])))
        srcs.append(x[1])
        srcs.append(self.gn(self.conv2(x[2])))
        srcs.append(self.gn(self.conv3(x[3])))
        srcs.append(self.gn(self.conv4(x[3])))
        # for i in range(5):
        #     print(srcs[i].shape)
        for i in range(self.n_levels):
            bs, c, h, w = srcs[i].shape
            srcs[i] = srcs[i].flatten(2).transpose(1, 2)
            src_flatten.append(srcs[i])
            spatial_shape = (h, w)  # 特征图shape
            spatial_shapes.append(spatial_shape)

        src_flatten = torch.cat(src_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        bs_src_flatten, hxw_src_flatten, c_src_flatten = src_flatten.shape
        reference_points = self.reference_points(src_flatten).sigmoid().view(bs_src_flatten, hxw_src_flatten,
                                                                             self.n_levels, 2)
        input_mask = None
        # out = self.attn(src_flatten, reference_points, src_flatten, spatial_shapes, level_start_index, input_mask)
        out_attn = self.attn(src_flatten, reference_points, src_flatten, spatial_shapes, level_start_index, input_mask)
        for i in range(self.layer - 1):
            out_attn = self.attn(out_attn, reference_points, out_attn, spatial_shapes, level_start_index, input_mask)

        # print(out.shape)
        out = src_flatten + self.dropout1(out_attn)
        out = self.norm1(out)
        # ffn   feed forward + add + norm
        out = self.forward_ffn(out)
        # print(out.shape)
        h = [h1, h2, h3, h4, h5]
        w = [w1, w2, w3, w4, w5]
        # for i in range(5):
        #     print(h[i])
        #     print(w[i])
        f = torch.split(out, [h1 * w1, h2 * w2, h3 * w3, h4 * w4, h5 * w5], 1)
        feature = []
        for i in range(self.n_levels):
            feature.append(f[i].transpose(1, 2))
            bs, ts, hxw = feature[i].shape
            hi = int(hxw / w[i])
            wi = int(hxw / h[i])
            feature[i] = feature[i].reshape([bs, ts, hi, wi])
        src_out = []
        src_out.append(self.conv11(feature[0]))
        src_out.append(feature[1])
        src_out.append(self.conv22(feature[2]))
        src_out.append(self.conv33(feature[3]))
        # for i in range(4):
        #     print(src_out[i].shape)

        return src_out


class SAFF_256(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0][1]


class SAFF_512(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0][2]


class SAFF_1024(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0][3]
