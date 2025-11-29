import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, 
                 batch_norm=True, preactivation=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        layers = []
        if preactivation:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))
            if batch_norm:
                layers = [nn.BatchNorm2d(in_channel)] + layers
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, 
                 batch_norm=True, preactivation=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        layers = []
        layers.append(nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False))
        layers.append(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=not batch_norm))
        
        if preactivation:
            dw = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False)
            pw = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)
            layers = [nn.ReLU(), dw, pw]
            if batch_norm:
                layers = [nn.BatchNorm2d(in_channel)] + layers
        else:
            ops = []
            ops.append(nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False))
            ops.append(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
            if batch_norm:
                ops.append(nn.BatchNorm2d(out_channel))
            ops.append(nn.ReLU(inplace=True))
            layers = ops

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class DenseFeatureStack(nn.Module):
    def __init__(self, in_channel, kernel_size, unit, growth_rate):
        super(DenseFeatureStack, self).__init__()

        self.conv_units = nn.ModuleList()
        current_in = in_channel
        for i in range(unit):
            self.conv_units.append(
                ConvBlock(
                    in_channel=current_in,
                    out_channel=growth_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    batch_norm=True,
                    preactivation=True
                )
            )
            current_in += growth_rate

    def forward(self, x):
        stack_feature = None

        for conv in self.conv_units:
            if stack_feature is None:
                inputs = x
            else:
                inputs = torch.cat([x, stack_feature], dim=1)
            
            out = conv(inputs)
            
            if stack_feature is None:
                stack_feature = out
            else:
                stack_feature = torch.cat([stack_feature, out], dim=1)

        return torch.cat([x, stack_feature], dim=1)

# --- Global Fusion ---
class GlobalContextFusion(nn.Module):
    def __init__(self, in_channels, max_pool_kernels, ch, ch_k, ch_v, br):
        super(GlobalContextFusion, self).__init__()
        self.ch_bottle = in_channels[-1]
        self.ch_in = ch * br
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br

        # Multi-scale adaptation layers
        self.ch_convs = nn.ModuleList([
            DepthWiseSeparateConvBlock(inc, ch, 3, 1, batch_norm=True, preactivation=True)
            for inc in in_channels
        ])

        self.max_pool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=k)
            for k in max_pool_kernels
        ])

        # Channel Attention Projections
        self.ch_Wq = DepthWiseSeparateConvBlock(self.ch_in, self.ch_in, 1, 1, batch_norm=True, preactivation=True)
        self.ch_Wk = DepthWiseSeparateConvBlock(self.ch_in, 1, 1, 1, batch_norm=True, preactivation=True)
        self.ch_Wv = DepthWiseSeparateConvBlock(self.ch_in, self.ch_in, 1, 1, batch_norm=True, preactivation=True)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_score_conv = nn.Conv2d(self.ch_in, self.ch_in, 1)
        self.ch_layer_norm = nn.LayerNorm((self.ch_in, 1, 1))
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention Projections
        self.sp_Wq = DepthWiseSeparateConvBlock(self.ch_in, br * ch_k, 1, 1, batch_norm=True, preactivation=True)
        self.sp_Wk = DepthWiseSeparateConvBlock(self.ch_in, br * ch_k, 1, 1, batch_norm=True, preactivation=True)
        self.sp_Wv = DepthWiseSeparateConvBlock(self.ch_in, br * ch_v, 1, 1, batch_norm=True, preactivation=True)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_output_conv = DepthWiseSeparateConvBlock(br * ch_v, self.ch_in, 1, 1, batch_norm=True, preactivation=True)

        self.output_conv = DepthWiseSeparateConvBlock(self.ch_in, self.ch_bottle, 3, 1, batch_norm=True, preactivation=True)

    def forward(self, feature_maps):
        max_pool_maps = [pool(f) for pool, f in zip(self.max_pool_layers, feature_maps)]
        ch_outs = [conv(m) for conv, m in zip(self.ch_convs, max_pool_maps)]
        x = torch.cat(ch_outs, dim=1) 

        bs, c, h, w = x.size()

        ch_Q = self.ch_Wq(x).reshape(bs, -1, h * w)       
        ch_K = self.ch_Wk(x).reshape(bs, -1, 1)           
        ch_K = self.ch_softmax(ch_K)                      
        
        Z_ch = torch.matmul(ch_Q, ch_K).unsqueeze(-1)     
        ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z_ch)))
        ch_out = self.ch_Wv(x) * ch_score

        sp_Q = self.sp_Wq(ch_out).reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).reshape(bs, self.ch_k, -1) 
        
        sp_K = self.sp_Wk(ch_out).reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1)
        sp_K = sp_K.mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k) 
        sp_K = self.sp_softmax(sp_K)
        
        Z_sp = torch.matmul(sp_K, sp_Q).reshape(bs, 1, h, w, self.br)
        sp_score = self.sigmoid(Z_sp)
        
        sp_V = self.sp_Wv(ch_out).reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1) 
        sp_out = sp_V * sp_score
        
        sp_out = sp_out.permute(0, 4, 1, 2, 3).reshape(bs, self.br * self.ch_v, h, w)
        sp_out = self.sp_output_conv(sp_out)

        return self.output_conv(sp_out)