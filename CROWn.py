import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from typing import Optional, Sequence, Union
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import math
from math import sqrt


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat :int = 96,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    

class DWT2D_Haar(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        l = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)])
        h = torch.tensor([1/math.sqrt(2),-1/math.sqrt(2)])
        ll = torch.outer(l, l); lh = torch.outer(l, h)
        hl = torch.outer(h, l); hh = torch.outer(h, h)
        k = torch.stack([ll, lh, hl, hh], dim=0)  # (4,2,2)
        weight = torch.zeros((channels*4, 1, 2, 2))
        for c in range(channels):
            weight[c*4+0,0] = ll
            weight[c*4+1,0] = lh
            weight[c*4+2,0] = hl
            weight[c*4+3,0] = hh
        self.register_buffer('weight', weight)
        self.groups = channels

    def forward(self, x):  # x: (N,C,D,H)
        x = F.pad(x, (0, x.shape[-1]%2, 0, x.shape[-2]%2), mode='reflect')
        y = F.conv2d(x, self.weight, stride=2, groups=self.groups)  # (N,4C,D/2,H/2)
        C = x.shape[1]
        LL,LH,HL,HH = torch.chunk(y, 4, dim=1)  # (N,C,D/2,H/2)
        return LL, LH, HL, HH


def gn(c):  # GroupNorm
    return nn.GroupNorm(num_groups=min(32, max(1, c//4)), num_channels=c)

class ChannelMLP(nn.Module):
    def __init__(self, c, expansion=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c*expansion, 1)
        self.fc2 = nn.Conv2d(c*expansion, c, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class SR(nn.Module):
    """Spatial Reduction for K/V"""
    def __init__(self, c, sr_ratio=2):
        super().__init__()
        self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio>1 else None
        self.norm = gn(c)
    def forward(self, x):
        if self.sr is None: return x
        return self.norm(self.sr(x))

class CrossSourceMHA(nn.Module):
    
    def __init__(self, c_q, c_k, c_v, c_out, heads=4, sr_ratio=2):
        super().__init__()
        self.h = heads
        self.q = nn.Conv2d(c_q, c_out, 1)
        self.k = nn.Conv2d(c_k, c_out, 1)
        self.v = nn.Conv2d(c_v, c_out, 1)
        self.norm_q = gn(c_q); self.norm_k = gn(c_k); self.norm_v = gn(c_v)
        self.proj = nn.Conv2d(c_out, c_out, 1)
        self.sr_k = SR(c_k, sr_ratio); self.sr_v = SR(c_v, sr_ratio)
        self.scale = (c_out // heads) ** -0.5

    def _reshape(self, x):  # (N,C,H,W)->(N,heads,HW,dim)
        N,C,H,W = x.shape
        x = x.view(N, self.h, C//self.h, H*W)          # (N,h,dim,HW)
        return x.permute(0,1,3,2).contiguous()         # (N,h,HW,dim)

    def forward(self, q_src, k_src, v_src):
        q = self.q(self.norm_q(q_src))
        k = self.k(self.norm_k(self.sr_k(k_src)))
        v = self.v(self.norm_v(self.sr_v(v_src)))

        N, Cq, Hq, Wq = q.shape
        q = self._reshape(q); k = self._reshape(k); v = self._reshape(v)
        attn = (q * self.scale) @ k.transpose(-2, -1)   # (N,h,HWq,HWk)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                  # (N,h,HWq,dim)
        out = out.permute(0,1,3,2).contiguous().view(N, Cq, Hq, Wq)  # (N,Cq,Hq,Wq)
        return self.proj(out)

class μPCAD(nn.Module):
    """
    Microlocal Polyphase Co-Attentive Decimator 
    """
    def __init__(self, C_in, C_out, heads=4, sr_ratio=2, mlp_ratio=4, drop=0.0):
        super().__init__()
        C_mid = 2*C_in  

        self.proj_pool = nn.Conv2d(C_in, 2*C_in, 1)   # -> split for Max/Avg
        self.proj_wav  = nn.Conv2d(C_in, C_in, 1)     # -> for DWT
        self.dwt = DWT2D_Haar(C_in)

        self.maxpool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AvgPool2d(2,2)

        # [Max, Avg, LL, LH, HL, HH] -> 3x3 -> MLP
        self.fuse_local = nn.Sequential(
            nn.Conv2d(6*C_in, C_mid, 3, padding=1),
            gn(C_mid), nn.GELU(),
            ChannelMLP(C_mid, expansion=mlp_ratio, drop=drop)
        )

        # Q=Max, K=Avg, V=Wavelet
        self.attn = CrossSourceMHA(c_q=C_in, c_k=C_in, c_v=4*C_in,
                                   c_out=C_mid, heads=heads, sr_ratio=sr_ratio)
        self.attn_mlp = ChannelMLP(C_mid, expansion=mlp_ratio, drop=drop)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))  
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, C_mid//4, 1), nn.GELU(),
            nn.Conv2d(C_mid//4, C_mid, 1), nn.Sigmoid()
        )

        self.blur_w = nn.Conv3d(C_mid, C_mid, kernel_size=(1,1,3),
                                padding=(0,0,1), groups=C_mid, bias=False)
        with torch.no_grad():
            k = torch.tensor([1.,2.,1.]).view(1,1,1,1,3)/4.0
            self.blur_w.weight.copy_(k.repeat(C_mid,1,1,1,1))
        for p in self.blur_w.parameters(): p.requires_grad = False

        self.down_w = nn.Conv3d(C_mid, C_out, kernel_size=(1,1,3),
                                stride=(1,1,2), padding=(0,0,1), bias=True)

    def forward(self, x):  # x: (B,C,D,H,W)
        B,C,D,H,W = x.shape

        x4 = x.permute(0,4,1,2,3).contiguous().view(B*W, C, D, H)  # (BW,C,D,H)

        pool_in = self.proj_pool(x4)                        # (BW,2C,D,H)
        q_in, k_in = torch.chunk(pool_in, 2, dim=1)         # for Max / Avg
        v_in  = self.proj_wav(x4)                           # (BW,C,D,H)

        x_max = self.maxpool(q_in)                          # (BW,C,D/2,H/2)
        x_avg = self.avgpool(k_in)                          # (BW,C,D/2,H/2)
        LL,LH,HL,HH = self.dwt(v_in)                        # (BW,C,D/2,H/2)*4

        local = torch.cat([x_max, x_avg, LL, LH, HL, HH], dim=1)  # (BW,6C,·,·)
        local = self.fuse_local(local)                      # (BW,C_mid,D/2,H/2)

        wav_cat = torch.cat([LL,LH,HL,HH], dim=1)           # (BW,4C,·,·)
        attn = self.attn(q_src=x_max, k_src=x_avg, v_src=wav_cat)  # (BW,C_mid,·,·)
        attn = self.attn_mlp(attn)

        fused = torch.sigmoid(self.alpha)*local + torch.sigmoid(self.beta)*attn \
                + self.gamma * nn.Conv2d(LL.shape[1], local.shape[1], 1, bias=False).to(x.device)(LL)
        fused = fused * self.se(fused)                      # SE

        y = fused.view(B, W, -1, D//2, H//2).permute(0,2,3,4,1).contiguous()  # (B,C_mid,D/2,H/2,W)
        y = self.blur_w(y)                                  
        y = self.down_w(y)                                  # (B,C_out,D/2,H/2,W/2)
        return y



def space_to_depth_3d(x: torch.Tensor, block_size: int = 2):
    # x: (B,C,D,H,W) -> (B, C*block^3, D/block, H/block, W/block)
    B, C, D, H, W = x.shape
    assert D % block_size == 0 and H % block_size == 0 and W % block_size == 0
    b = block_size
    x = x.view(B, C, D//b, b, H//b, b, W//b, b)       # B,C,D',b,H',b,W',b
    x = x.permute(0,1,3,5,7,2,4,6).contiguous()       # B,C,b,b,b,D',H',W'
    x = x.view(B, C*(b**3), D//b, H//b, W//b)
    return x

def gn3d(c):  
    return nn.GroupNorm(num_groups=min(32, max(1, c//4)), num_channels=c)

class FixedBlur3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv3d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        with torch.no_grad():
            k1 = torch.tensor([1., 2., 1.]) / 4.0  
            k = torch.einsum('i,j,k->ijk', k1, k1, k1) / 2.0  
            w = k.view(1,1,3,3,3).repeat(c,1,1,1,1)
            self.dw.weight.copy_(w)
        for p in self.dw.parameters(): p.requires_grad = False
    def forward(self, x):
        return self.dw(x)

class Sobel3D(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([-1., 0., 1.])
        ga = torch.tensor([1., 2., 1.])
        kx = torch.einsum('i,j,k->ijk', gx, ga, ga)
        ky = torch.einsum('i,j,k->ijk', ga, gx, ga)
        kz = torch.einsum('i,j,k->ijk', ga, ga, gx)
        K = torch.stack([kx, ky, kz], 0) / 8.0
        self.register_buffer('K', K.view(3,1,3,3,3))
    def forward(self, x):  # x: (B,1,D,H,W)
        dx = F.conv3d(x, self.K[0:1], padding=1)
        dy = F.conv3d(x, self.K[1:2], padding=1)
        dz = F.conv3d(x, self.K[2:3], padding=1)
        return torch.sqrt(dx*dx + dy*dy + dz*dz + 1e-6)

class DepthwiseSepBlock3D(nn.Module):
    """DWConv3D -> GN -> GELU -> PWConv"""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.dw = nn.Conv3d(c_in, c_in, k, padding=k//2, groups=c_in, bias=False)
        self.pw = nn.Conv3d(c_in, c_out, 1, bias=True)
        self.norm = gn3d(c_in)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pw(x)
        return x

class PhaseAttention3D(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.c_in = c_in
        self.red = max(1, c_in // reduction)
        self.ctx = nn.Sequential(
            nn.Conv3d(c_in*8, self.red, 1, bias=True),
            nn.GELU(),
            nn.Conv3d(self.red, 8, 1, bias=True)
        )
    def forward(self, x):
        B, C8, D, H, W = x.shape
        C = C8 // 8
        w = self.ctx(x).view(B, 8, D, H, W)             # (B,8,D,H,W)
        w = w.softmax(dim=1)                          
        xp = x.view(B, C, 8, D, H, W)                    # (B,C,8,D,H,W)
        out = (xp * w.unsqueeze(1)).sum(dim=2)
        return out

class OCF(nn.Module):
    """
    Octaphase Coset Fibration 
    """
    def __init__(self, C_in, C_out, block=2, mlp_ratio=2):
        super().__init__()
        assert block == 2, ""
        self.blur = FixedBlur3D(C_in)
        self.phase_attn = PhaseAttention3D(C_in, reduction=4)
        self.edge = Sobel3D()
        C_mid = max(C_in, C_out)
        self.fuse = DepthwiseSepBlock3D(C_in, C_mid, k=3)
        self.proj = nn.Sequential(
            nn.Conv3d(C_mid, C_mid*mlp_ratio, 1),
            nn.GELU(),
            nn.Conv3d(C_mid*mlp_ratio, C_out, 1),
        )
        self.edge_gain = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, x):  # x: (B,C,D,H,W)
        B,C,D,H,W = x.shape
        x = self.blur(x)
        x_sd = space_to_depth_3d(x, block_size=2)        # (B, C*8, D/2,H/2,W/2)
        x_pa = self.phase_attn(x_sd)                     # (B, C,   D/2,H/2,W/2)
        with torch.no_grad():
            ed = x.mean(1, keepdim=True)                
        ed = self.edge(ed)                               # (B,1,D-2,H-2,W-2)≈(B,1,D,H,W)
        ed = F.avg_pool3d(ed, kernel_size=2, stride=2)   # D/2,H/2,W/2
        ed = ed / (ed.mean(dim=[2,3,4], keepdim=True) + 1e-6)
        gate = 1.0 + torch.tanh(self.edge_gain) * ed     # (B,1,·,·,·)
        x_pa = x_pa * gate
        y = self.fuse(x_pa)
        y = self.proj(y)                                 # (B,C_out,D/2,H/2,W/2)
        return y

    
class CROWn(nn.Module):
    
    'Coset-fibRated micrO-local co-attention Network'
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        
        features: Sequence[int] = (24, 48, 96, 192, 384, 24),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)


        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout,feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout,feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout,feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout,feat=12)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        self.μPCAD_1 = μPCAD(C_in=fea[0], C_out=fea[1], heads=1)
        self.μPCAD_2 = μPCAD(C_in=fea[1], C_out=fea[2], heads=2)
        self.μPCAD_3 = μPCAD(C_in=fea[2], C_out=fea[3], heads=4)
        self.μPCAD_4 = μPCAD(C_in=fea[3], C_out=fea[4], heads=8)

        self.OCF_0 = OCF(C_in=fea[0], C_out=fea[1]) 
        self.OCF_1 = OCF(C_in=fea[1], C_out=fea[2])
        self.OCF_2 = OCF(C_in=fea[2], C_out=fea[3])
        self.OCF_3 = OCF(C_in=fea[3], C_out=fea[4])



    def forward(self, x: torch.Tensor):
                
        x0 = self.conv_0(x)
        x1 = self.down_1(x0) + self.μPCAD_1(x0)       
        x2 = self.down_2(x1) + self.μPCAD_2(x1)   
        x3 = self.down_3(x2) + self.μPCAD_3(x2)
        x4 = self.down_4(x3) + self.μPCAD_4(x3)
        
        x4 = x4 + self.OCF_3(x3)
        x3 = x3 + self.OCF_2(x2)
        x2 = x2 + self.OCF_1(x1)
        x1 = x1 + self.OCF_0(x0)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        
        logits = self.final_conv(u1)

        return logits