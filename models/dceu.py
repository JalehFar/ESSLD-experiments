import torch
import torch.nn as nn
from .other_blocks import ECA

class DCEU(nn.Module):
    
    def __init__(self, inp, oup, groups=32, reduction=4, use_residual=True):
        super(DCEU, self).__init__()
        self.use_residual = use_residual

        # Pooling operations
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.shared_conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.shared_bn = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

        # Learnable Gating Mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // reduction, 2, 1),
            nn.Softmax(dim=1)
        )

        self.channel_att = ECA(channel=oup)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Path 1: Mean Pooling
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.relu(self.shared_bn(self.shared_conv1(y_mean)))
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)
        attn_mean = self.conv_h(x_h_mean).sigmoid() * self.conv_w(x_w_mean).sigmoid()

        # Path 2: Max Pooling
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.relu(self.shared_bn(self.shared_conv1(y_max)))
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)
        attn_max = self.conv_h(x_h_max).sigmoid() * self.conv_w(x_w_max).sigmoid()
        
        # Adaptive Fusion
        gate_weights = self.gate(identity)
        attn = attn_mean * gate_weights[:, 0:1] + attn_max * gate_weights[:, 1:2]
        
        out = identity * attn
        out = self.channel_att(out)

        if self.use_residual:
            out = out + identity
            
        return out