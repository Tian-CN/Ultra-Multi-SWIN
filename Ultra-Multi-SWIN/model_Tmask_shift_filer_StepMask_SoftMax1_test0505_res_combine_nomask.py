""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)  # TZK_MASK  进embeding前把无效部分给直接mask掉。   1型给0
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.norm_shift = norm_layer(dim * 3)  # TZK_shift
        self.norm_shift_CB = norm_layer(dim)  # TZK_combine
        # self.norm_shift = norm_layer(dim * 2)  # TZK_TEST 去掉斜向

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        mlp_hidden_dim_shift = int(dim)  # TZK_shift
        mlp_in_features_shift = int(dim * 3) # TZK_TEST
        mlp_in_features_shift_CB = int(dim)  # TZK_combine
        # mlp_in_features_shift = int(dim * 2)  # TZK_TEST 去掉斜向
        self.mlp_shift = Mlp(in_features=mlp_in_features_shift, hidden_features=mlp_hidden_dim, out_features=mlp_hidden_dim_shift, act_layer=act_layer,
                             drop=drop)  # TZK_shift
        self.mlp_shift_CB = Mlp(in_features=mlp_in_features_shift_CB, hidden_features=mlp_hidden_dim,
                             out_features=mlp_hidden_dim_shift, act_layer=act_layer,
                             drop=drop)  # TZK_combine
        self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1) # TZK_RES

    def forward(self, x, attn_mask, attn_mask_row, attn_mask_column, count_laye):  # TZK_MASK
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        conbine_flag = False    # TZK_combine
        if count_laye > 1:      # TZK_combine
            conbine_flag = True # TZK_combine


        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_row = torch.roll(x, shifts=(0, -self.shift_size), dims=(1, 2))  # TZK_shift
            shifted_x_column = torch.roll(x, shifts=(-self.shift_size, 0), dims=(1, 2))  # TZK_shift
        else:
            shifted_x = x
            shifted_x_row = x  # TZK_shift
            shifted_x_column = x  # TZK_shift

            attn_mask = None
            attn_mask_row = None
            attn_mask_column = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows_row = window_partition(shifted_x_row, self.window_size)  # TZK_shift_cut
        x_windows_column = window_partition(shifted_x_column, self.window_size)  # TZK_shift_cut

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]
        x_windows_row = x_windows_row.view(-1, self.window_size * self.window_size, C)  # TZK_shift_cut
        x_windows_column = x_windows_column.view(-1, self.window_size * self.window_size, C)  # TZK_shift_cut

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]
        attn_windows_row = self.attn(x_windows_row, mask=attn_mask_row)  # TZK_MASK
        attn_windows_column = self.attn(x_windows_column, mask=attn_mask_column)  # TZK_MASK

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]
        attn_windows_row = attn_windows_row.view(-1, self.window_size, self.window_size,
                                                 C)  # [nW*B, Mh, Mw, C] # TZK_MASK
        shifted_x_row = window_reverse(attn_windows_row, self.window_size, Hp, Wp)  # [B, H', W', C]
        attn_windows_column = attn_windows_column.view(-1, self.window_size, self.window_size,
                                                       C)  # [nW*B, Mh, Mw, C] # TZK_MASK
        shifted_x_column = window_reverse(attn_windows_column, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_row = torch.roll(shifted_x_row, shifts=(0, self.shift_size), dims=(1, 2))  # TZK_MASK
            x_column = torch.roll(shifted_x_column, shifts=(self.shift_size, 0), dims=(1, 2))  # TZK_MASK
        else:
            x = shifted_x
            x_row = shifted_x_row  # TZK_MASK
            x_column = shifted_x_column  # TZK_MASK

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()
            x_row = x_row[:, :H, :W, :].contiguous()  # TZK_MASK
            x_column = x_column[:, :H, :W, :].contiguous()  # TZK_MASK

        x = x.view(B, H * W, C)
        x_row = x_row.view(B, H * W, C)  # TZK_MASK
        x_column = x_column.view(B, H * W, C)  # TZK_MASK

        if conbine_flag == False:   # TZK_combine
            x = torch.cat((x, x_row, x_column), dim=2)  # TZK_MASK
            # x = torch.cat((x, x_row), dim=2)  # TZK_TEST 去掉斜向

            # FFN
            # x = x + self.drop_path(self.mlp_shift(self.norm_shift(self.drop_path(x))))  # TZK_shift
            x = self.drop_path(self.mlp_shift(self.norm_shift(self.drop_path(x))))  # TZK_shift

            x = shortcut + self.drop_path(x)
            # x = self.drop_path(shortcut) + self.drop_path(x)  #  TZK_dropout
            shortcut = self.conv1d(shortcut.permute(0, 2, 1))  # TZK_RES
            # x = self.drop_path(x) + shortcut.permute(0, 2, 1)  # TZK_RES  TZK_dorpout
            x = x + shortcut.permute(0, 2, 1)  # TZK_RES

            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # x = torch.cat((x, x_row, x_column), dim=2)  # TZK_MASK
            # x = torch.cat((x, x_row), dim=2)  # TZK_TEST 去掉斜向

            # FFN
            # x = x + self.drop_path(self.mlp_shift(self.norm_shift(self.drop_path(x))))  # TZK_shift
            x = self.drop_path(self.mlp_shift_CB(self.norm_shift_CB(self.drop_path(x))))  # TZK_shift

            x = shortcut + self.drop_path(x)
            # x = self.drop_path(shortcut) + self.drop_path(x)  #  TZK_dropout
            shortcut = self.conv1d(shortcut.permute(0, 2, 1))  # TZK_RES
            # x = self.drop_path(x) + shortcut.permute(0, 2, 1)  # TZK_RES  TZK_dorpout
            x = x + shortcut.permute(0, 2, 1)  # TZK_RES

            x = x + self.drop_path(self.mlp(self.norm2(x)))


        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, i_layer=0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2
        self.i_laye = i_layer

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #                      dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # self.softmax = nn.Softmax2d()  # TZK_MASK 创建MASK模版
        self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1) #TZK_RES
        H_ = int(56 / (2 ** self.i_laye))
        self.mask_layer = [0, 3]
        self.attn_ultra_mask_Parameter = nn.Parameter(
            (self.create_ultrasound_mask_Parameter(H=H_, W=H_)).requires_grad_(requires_grad=True))
        self.pos_drop = nn.Dropout(p=attn_drop)  # TZK_dorpout

    def create_ultrasound_mask(self, x, H, W):  # TZK_MASK
        # calculate ultrasound attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1), device=x.device)  # [1, H, W, 1]
        B, nW, C = x.shape
        img_mask_B = torch.zeros((B, 1, 1, C), device=x.device)
        cnt = 1
        # y=a1 x + a2
        a1_left, a2_left = -2.671, 1
        a1_right, a2_right = 2.674, -1.674
        for h in range(H):
            for w in range(W):
                h1 = h / H
                w1 = w / W
                y1 = w1 * a1_left + a2_left
                y2 = w1 * a1_right + a2_right
                if (h1 < y1) or (h1 < y2):
                    img_mask[:, h, w, :] = cnt

        attn_mask = img_mask + img_mask_B  # 广播机制，扩充为[B, H, W, C],待调试检查
        # attn_mask = img_mask
        # transpose: [B, H, W, C] -> [B, W, H, C] -> [B, C, H, W]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        attn_mask = attn_mask.transpose(1, 2).transpose(1, 3).flatten(2).transpose(1, 2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def create_ultrasound_filer(self, x, H, W):  # TZK_filer
        # calculate ultrasound attention mask for SW-MSA
        img_filer = torch.zeros((1, H, W, 1), device=x.device)  # [1, H, W, 1]
        B, nW, C = x.shape
        img_filer_B = torch.zeros((B, 1, 1, C), device=x.device)
        cnt = 1
        filer_max, filer_min = 1, 0
        for h in range(H):
            if h < H/3:
                img_filer[:, h, :, :] = 1
            else:
                img_filer[:, h, :, :] = ((H-h)/H)+0.1

        attn_filer = img_filer + img_filer_B  # 广播机制，扩充为[B, H, W, C],待调试检查
        # attn_mask = img_mask
        # transpose: [B, H, W, C] -> [B, W, H, C] -> [B, C, H, W]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        attn_filer = attn_filer.transpose(1, 2).transpose(1, 3).flatten(2).transpose(1, 2)
        return attn_filer

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def create_mask_row(self, x, H, W):  # TZK_MASK
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        # h_slices = (slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0

        for w in w_slices:
            img_mask[:, :, w, :] = cnt
            cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def create_mask_column(self, x, H, W):  # TZK_MASK
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # w_slices = (slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            img_mask[:, h, :, :] = cnt
            cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) #filer -100
        return attn_mask
    def create_ultrasound_mask_Parameter(self, H, W):  # TZK_MASK
        # calculate ultrasound attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # [1, H, W, 1]
        # img_mask = torch.ones((1, H, W, 1), device=x.device)  # [1, H, W, 1]
        # B, nW, C = x.shape
        img_mask_B = torch.zeros((1, 1, 1, 1))
        cnt = 1
        # y=a1 x + a2
        a1_left, a2_left = -2.671, 1
        a1_right, a2_right = 2.674, -1.674
        for h in range(H):
            for w in range(W):
                h1 = h / H
                w1 = w / W
                y1 = w1 * a1_left + a2_left
                y2 = w1 * a1_right + a2_right
                if (h1 < y1) or (h1 < y2):
                    img_mask[:, h, w, :] = cnt

        attn_mask = img_mask + img_mask_B  # 广播机制，扩充为[B, H, W, C],待调试检查
        # attn_mask = img_mask
        # transpose: [B, H, W, C] -> [B, W, H, C] -> [B, C, H, W]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        attn_mask = attn_mask.transpose(1, 2).transpose(1, 3).flatten(2).transpose(1, 2)
        # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(1.0)).masked_fill(attn_mask == 0, float(0.1))
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-10.0)).masked_fill(attn_mask == 0, float(0.0)) # TZK_MASK
        return attn_mask

    def forward(self, x, H, W, count_laye):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        attn_mask_row = self.create_mask_row(x, H, W)  # TZK_MASK_row
        attn_mask_column = self.create_mask_column(x, H, W)  # TZK_MASK

        # if self.i_laye<2: #TZK MASK
        #     attn_ultra_mask = self.create_ultrasound_mask(x, H, W)  # TZK_MASK
        #     x = x + attn_ultra_mask  # 广播机制，MASK[1, nW, 1]扩充为[B, nW, C],待调试检查
        #     # x = self.softmax(attn)
        # attn_ultra_filer = self.create_ultrasound_filer(x, H, W)  # TZK_filer
        # x = x * attn_ultra_filer# TZK_filer
        # if self.i_laye in self.mask_layer:  # nomask
            # x = x + self.attn_ultra_mask_Parameter  # TZK_MASK nomask
            # x = self.pos_drop(self.pos_drop(x) + self.pos_drop(self.attn_ultra_mask_Parameter)) # TZK_MASK  TZK_dropout
            # x = self.pos_drop(x + self.attn_ultra_mask_Parameter)  # TZK_MASK  TZK_dropout

        for blk in self.blocks:
            blk.H, blk.W = H, W
            # attn_ultra_mask = self.create_ultrasound_mask(x, blk.H, blk.W)  # TZK_MASK
            # x = x + attn_ultra_mask  # 广播机制，MASK[1, nW, 1]扩充为[B, nW, C],待调试检查
            # x = self.softmax(x)
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, attn_mask_row, attn_mask_column)  # TZK_MASK
            else:
                # x_shortcut = x  #TZK_RES
                x = blk(x, attn_mask, attn_mask_row, attn_mask_column, count_laye)  # TZK_MASK                x_shortcut = self.conv1d(x_shortcut.permute(0, 2, 1))
                # x_shortcut = self.conv1d(x_shortcut.permute(0, 2, 1)) #TZK_RES
                # x = x + x_shortcut.permute(0, 2, 1)  #TZK_RES


        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

            # TZK MASK

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                i_layer=i_layer)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.softmax = nn.Softmax(dim=1)  # TZK_MASK 创建MASK模版
        self.mask_drop = nn.Dropout(p=drop_rate)  # TZK_MASK 创建MASK模版
        # self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)  # TZK_RES
        self.attn_ultra_mask_Parameter = nn.Parameter((self.create_ultrasound_mask_Parameter(B=128, C=192, H=28, W=28)).requires_grad_(requires_grad=True))
        self.attn_ultra_filer_Parameter = nn.Parameter(
            (self.create_ultrasound_filer_Parameter(C=96, H=56, W=56)).requires_grad_(requires_grad=True))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_ultrasound_filer(self, x, H, W):  # TZK_filer
        # calculate ultrasound attention mask for SW-MSA
        img_filer = torch.zeros((1, H, W, 1), device=x.device)  # [1, H, W, 1]
        B, nW, C = x.shape
        img_filer_B = torch.zeros((B, 1, 1, C), device=x.device)
        cnt = 1
        filer_max, filer_min = 1, 0
        filter_value = (H)/4

        for h in range(H):
            if h < filter_value:
                img_filer[:, h, :, :] = 1
            else:
                img_filer[:, h, :, :] = ((H-h)/(H-filter_value))+0.1

        attn_filer = img_filer + img_filer_B  # 广播机制，扩充为[B, H, W, C],待调试检查
        # attn_mask = img_mask
        # transpose: [B, H, W, C] -> [B, W, H, C] -> [B, C, H, W]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        attn_filer = attn_filer.transpose(1, 2).transpose(1, 3).flatten(2).transpose(1, 2)
        return attn_filer
    def create_ultrasound_filer_Parameter(self, C, H, W):  # TZK_filer
        # calculate ultrasound attention mask for SW-MSA
        img_filer = torch.zeros((1, H, W, 1))  # [1, H, W, 1]
        # B, nW, C = x.shape
        img_filer_B = torch.zeros((1, 1, 1, 1))
        cnt = 1
        filer_max, filer_min = 1, 0
        filter_value = (H)/4

        for h in range(H):
            if h < filter_value:
                img_filer[:, h, :, :] = 1
            else:
                img_filer[:, h, :, :] = ((H-h)/(H-filter_value))+0.1

        attn_filer = img_filer + img_filer_B  # 广播机制，扩充为[B, H, W, C],待调试检查
        # attn_mask = img_mask
        # transpose: [B, H, W, C] -> [B, W, H, C] -> [B, C, H, W]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        attn_filer = attn_filer.transpose(1, 2).transpose(1, 3).flatten(2).transpose(1, 2)
        return attn_filer
    def create_ultrasound_mask(self, x, H, W):  # TZK_MASK
        # calculate ultrasound attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1), device=x.device)  # [1, H, W, 1]
        # img_mask = torch.ones((1, H, W, 1), device=x.device)  # [1, H, W, 1]
        B, nW, C = x.shape
        img_mask_B = torch.zeros((B, 1, 1, C), device=x.device)
        cnt = 1
        # y=a1 x + a2
        a1_left, a2_left = -2.671, 1
        a1_right, a2_right = 2.674, -1.674
        for h in range(H):
            for w in range(W):
                h1 = h / H
                w1 = w / W
                y1 = w1 * a1_left + a2_left
                y2 = w1 * a1_right + a2_right
                if (h1 < y1) or (h1 < y2):
                    img_mask[:, h, w, :] = cnt

        attn_mask = img_mask + img_mask_B  # 广播机制，扩充为[B, H, W, C],待调试检查
        # attn_mask = img_mask
        # transpose: [B, H, W, C] -> [B, W, H, C] -> [B, C, H, W]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        attn_mask = attn_mask.transpose(1, 2).transpose(1, 3).flatten(2).transpose(1, 2)
        # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(1.0)).masked_fill(attn_mask == 0, float(0.1))
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-10.0)).masked_fill(attn_mask == 0, float(0.0)) # TZK_MASK
        return attn_mask
    def create_ultrasound_mask_Parameter(self, B, C, H, W):  # TZK_MASK
        # calculate ultrasound attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # [1, H, W, 1]
        # img_mask = torch.ones((1, H, W, 1), device=x.device)  # [1, H, W, 1]
        # B, nW, C = x.shape
        img_mask_B = torch.zeros((1, 1, 1, 1))
        cnt = 1
        # y=a1 x + a2
        a1_left, a2_left = -2.671, 1
        a1_right, a2_right = 2.674, -1.674
        for h in range(H):
            for w in range(W):
                h1 = h / H
                w1 = w / W
                y1 = w1 * a1_left + a2_left
                y2 = w1 * a1_right + a2_right
                if (h1 < y1) or (h1 < y2):
                    img_mask[:, h, w, :] = cnt

        attn_mask = img_mask + img_mask_B  # 广播机制，扩充为[B, H, W, C],待调试检查
        # attn_mask = img_mask
        # transpose: [B, H, W, C] -> [B, W, H, C] -> [B, C, H, W]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        attn_mask = attn_mask.transpose(1, 2).transpose(1, 3).flatten(2).transpose(1, 2)
        # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(1.0)).masked_fill(attn_mask == 0, float(0.1))
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-10.0)).masked_fill(attn_mask == 0, float(0.0)) # TZK_MASK
        return attn_mask

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # attn_ultra_filer = self.create_ultrasound_filer(x, H, W)  # TZK_filer
        # x = x * attn_ultra_filer  # TZK_filer

        # x = x * self.attn_ultra_filer_Parameter  # TZK_filer nomask
        # x = self.pos_drop(x * self.attn_ultra_filer_Parameter)  # TZK_filer TZK_dropout

        count_laye = 0
        for layer in self.layers:
            # x_shortcut = x  # TZK_RES

            # if (0 < count_laye) and (count_laye < 2):  # TZK MASK 1, H = 28, W = 28 2, 14 14 3, 7 7 4,7 7
            # if (1 < count_laye) and (count_laye < 3):  # TZK MASK
            #     attn_ultra_mask = self.create_ultrasound_mask(x, H, W)  # TZK_MASK
                # x = x + attn_ultra_mask  # 广播机制，MASK[1, nW, 1]扩充为[B, nW, C],待调试检查
                # x = x + self.attn_ultra_mask_Parameter  # TZK 可训练过程MASK
                # x = torch.mul(x, attn_ultra_mask)
                # x = self.softmax(x)
                # x = self.mask_drop(x) # TZK_test
                ## x = x + x_Shortcut  # TZK_RES
            x, H, W = layer(x, H, W, count_laye)
            count_laye = count_laye + 1

            # x_shortcut = self.conv1d(x_shortcut.permute(0, 2, 1))  # TZK_RES
            # x = x + x_shortcut.permute(0, 2, 1)  # TZK_RES
        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth        #没下下来
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model
