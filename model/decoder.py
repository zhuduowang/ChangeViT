import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model.utils import weight_init


def drop_path(x, drop_prob: float = 0., training: bool = False):
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
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape

        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B2, N2, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim1)        
        self.norm2 = norm_layer(dim2)
        self.attn = CrossAttention(dim1, dim2, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class FeatureInjector(nn.Module):
    def __init__(self, dim1=384, dim2=[64, 128, 256], num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.c2_c5 = Block(dim1, dim2[0], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c3_c5 = Block(dim1, dim2[1], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c4_c5 = Block(dim1, dim2[2], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)

        self.fuse = nn.Conv2d(dim1*3, dim1, 1, bias=False)

        weight_init(self)


    def base_forward(self, c2, c3, c4, c5):
        H, W = c5.shape[2:]

        c2 = rearrange(c2, 'b c h w -> b (h w) c')
        c3 = rearrange(c3, 'b c h w -> b (h w) c')
        c4 = rearrange(c4, 'b c h w -> b (h w) c')
        c5 = rearrange(c5, 'b c h w -> b (h w) c')

        _c2 = self.c2_c5(c5, c2)
        _c2 = rearrange(_c2, 'b (h w) c -> b c h w', h=H, w=W)

        _c3 = self.c3_c5(c5, c3)
        _c3 = rearrange(_c3, 'b (h w) c -> b c h w', h=H, w=W)

        _c4 = self.c4_c5(c5, c4)
        _c4 = rearrange(_c4, 'b (h w) c -> b c h w', h=H, w=W)

        _c5 = self.fuse(torch.cat([_c2, _c3, _c4], dim=1))

        return _c5

    def forward(self, fx, fy):
        _c5x = self.base_forward(fx[0], fx[1], fx[2], fx[3])
        _c5y = self.base_forward(fy[0], fy[1], fy[2], fy[3])

        return _c5x, _c5y


class Decoder(nn.Module):
    def __init__(self, in_dim=[64, 128, 256, 384], decay=4, num_class=1):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.up_c5 = nn.Sequential(
            nn.Conv2d(c5_channel, c4_channel, 1, bias=False),
            nn.ConvTranspose2d(c4_channel, c4_channel, kernel_size=4, stride=2, padding=1)
        )

        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, c3_channel, 1, bias=False),
            nn.ConvTranspose2d(c3_channel, c3_channel, kernel_size=4, stride=2, padding=1)
        )

        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, c2_channel, 1, bias=False),
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1)
        )

        self.classfier = nn.Sequential(
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(c2_channel, num_class, 3, 1, padding=1, bias=False)
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim*3, dim//decay, 1, bias=False),
                nn.BatchNorm2d(dim//decay),
                nn.ReLU(),
                nn.Conv2d(dim//decay, dim//decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim//decay, dim//decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim//decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim 
        ])

    def difference_modeling(self, x, y, block):
        f = torch.cat([x, y, torch.abs(x-y)], dim=1)
        f = block(f)

        return f

    def forward(self, fx, fy):
        c2x, c3x, c4x = fx[:-1]
        c2y, c3y, c4y = fy[:-1]

        c5x, c5y = self.structure_enhance(fx, fy)

        c2 = self.difference_modeling(c2x, c2y, self.mlp[0])
        c3 = self.difference_modeling(c3x, c3y, self.mlp[1])
        c4 = self.difference_modeling(c4x, c4y, self.mlp[2])
        c5 = self.difference_modeling(c5x, c5y, self.mlp[3])

        c4f = c4 + self.up_c5(c5)
        c3f = c3 + self.up_c4(c4f)
        c2f = c2 + self.up_c3(c3f)

        pred = self.classfier(c2f)
        pred_mask = torch.sigmoid(pred)

        return pred_mask
