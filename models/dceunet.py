import torch
import torch.nn as nn

from .other_blocks import ConvBlock, GlobalContextFusion
from .ega import DownSampleEGA
from .dceu import DCEU

class DCEUNet(nn.Module):
    
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 ):
        super(DCEUNet, self).__init__()

        base_channels = [24, 24, 24]
        skip_channels = [12, 24, 24]
        units = [3, 5, 5]
        pmfs_ch = 48
        upsample_mode = 'bilinear'

        # --- Encoder ---
        kernel_sizes = [5, 3, 3]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))]

        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                DownSampleEGA(
                    in_channel=(input_channels if i == 0 else downsample_channels[i - 1]),
                    base_channel=base_channels[i],
                    kernel_size=kernel_sizes[i],
                    skip_channel=skip_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i],
                    downsample=True,
                    skip=True
                )
            )

        # --- Bottleneck ---
        self.global_fusion = GlobalContextFusion(
            in_channels=downsample_channels,
            max_pool_kernels=[4, 2, 1],
            ch=pmfs_ch,
            ch_k=pmfs_ch,
            ch_v=pmfs_ch,
            br=3
        )
        
        global_out_channels = downsample_channels[-1]
        self.dceu_enhancer = DCEU(inp=global_out_channels, oup=global_out_channels)

        # --- Decoder ---
        self.bottle_conv = ConvBlock(
            in_channel=downsample_channels[2] + skip_channels[2],
            out_channel=skip_channels[2],
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True
        )
        self.upsample_1 = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.upsample_2 = nn.Upsample(scale_factor=4, mode=upsample_mode)

        # --- Output ---
        self.out_conv = ConvBlock(
            in_channel=sum(skip_channels),
            out_channel=num_classes,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True
        )
        self.upsample_out = nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, x):
        # Encoder
        x1, skip1 = self.down_convs[0](x)
        x2, skip2 = self.down_convs[1](x1)
        x3, skip3 = self.down_convs[2](x2)

        # Bottleneck (Global Fusion + DCEU)
        x3 = self.global_fusion([x1, x2, x3])
        x3 = self.dceu_enhancer(x3)
        skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))

        # Decoder
        skip2 = self.upsample_1(skip2)
        skip3 = self.upsample_2(skip3)

        # Output
        out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))
        out = self.upsample_out(out)

        return out