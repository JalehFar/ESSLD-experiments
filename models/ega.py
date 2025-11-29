import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from .other_blocks import ConvBlock, ECA, DenseFeatureStack

class MultiScaleGaussian(nn.Module):
    def __init__(self, dim, sizes=[3, 5], sigmas=[0.8, 1.2]):
        super().__init__()
        self.filters = nn.ModuleList()
        for size, sigma in zip(sizes, sigmas):
            kernel = self._build_kernel(size, sigma)
            conv = nn.Conv2d(dim, dim, kernel_size=size, padding=size // 2, groups=dim, bias=False)
            conv.weight.data.copy_(kernel.repeat(dim, 1, 1, 1))
            conv.weight.requires_grad = False
            self.filters.append(conv)

    def forward(self, x):
        return sum(f(x) for f in self.filters) / len(self.filters)

    def _build_kernel(self, size: int, sigma: float):
        kernel_range = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        xx, yy = torch.meshgrid(kernel_range, kernel_range, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

class ScharrEdge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        
        self.conv_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        
        self.conv_x.weight.data.copy_(scharr_x.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))
        self.conv_y.weight.data.copy_(scharr_y.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

class EGACore(nn.Module):
    def __init__(self, dim, norm_layer=dict(type='BN'), act_layer=nn.GELU):
        super().__init__()
        self.gaussian = MultiScaleGaussian(dim)
        self.scharr = ScharrEdge(dim)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        self.eca = ECA(dim)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            build_norm_layer(norm_layer, dim)[1],
            act_layer()
        )

    def forward(self, x):
        g = self.gaussian(x)
        e = self.scharr(x)
        fused = self.norm(self.act(x + g + e))
        att = self.eca(fused)
        return self.fuse_conv(att)

class EGA(nn.Module):
    def __init__(self, input_channels, num_classes, norm_layer=dict(type='BN'), act_layer=nn.GELU):
        super().__init__()
        self.channel_proj = nn.Conv2d(input_channels, num_classes, kernel_size=1)
        self.enhancer = EGACore(dim=num_classes, norm_layer=norm_layer, act_layer=act_layer)
        self.shortcut = nn.Conv2d(input_channels, num_classes, kernel_size=1) if input_channels != num_classes else nn.Identity()

    def forward(self, x):
        sc = self.shortcut(x)
        x = self.channel_proj(x)
        x = self.enhancer(x)
        return x + sc

class DownSampleEGA(nn.Module):
    def __init__(self, in_channel, base_channel, kernel_size, unit, growth_rate, 
                 skip_channel=None, downsample=True, skip=True):
        super().__init__()
        self.skip = skip
        stride = 2 if downsample else 1
        
        self.downsampler = ConvBlock(in_channel, in_channel, 3, stride=stride, 
                                   batch_norm=True, preactivation=True)
        
        self.ega_enhancer = EGA(in_channel, base_channel)
        
        self.dense_stack = DenseFeatureStack(base_channel, 3, unit, growth_rate)

        if skip:
            self.skip_conv = ConvBlock(
                in_channel=base_channel + unit * growth_rate,
                out_channel=skip_channel,
                kernel_size=3, stride=1, batch_norm=True, preactivation=True
            )

    def forward(self, x):
        x = self.downsampler(x)
        x = self.ega_enhancer(x)
        x = self.dense_stack(x)

        if self.skip:
            x_skip = self.skip_conv(x)
            return x, x_skip
        else:
            return x