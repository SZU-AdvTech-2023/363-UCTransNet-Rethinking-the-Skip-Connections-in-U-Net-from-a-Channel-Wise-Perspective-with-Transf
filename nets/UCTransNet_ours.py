# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans import ChannelTransformer
from UCTransNet import ConvBatchNorm, DownBlock, UpBlock_attention, get_activation, _make_nConv

class DilatedConvBatchNorm(nn.Module):
    """(dilated convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, activation='ReLU', dilation=2):
        super(DilatedConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class ModifiedDownBlock(nn.Module):
    """Downscaling with maxpool convolution and dilated convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(ModifiedDownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs_standard = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.nConvs_dilated = _make_nConv(in_channels, out_channels, nb_Conv, activation)

        # Replace standard convolutions with dilated convolutions in nConvs_dilated
        for i in range(nb_Conv):
            self.nConvs_dilated[i] = DilatedConvBatchNorm(out_channels, out_channels, activation)

    def forward(self, x):
        x = self.maxpool(x)
        conv_standard = self.nConvs_standard(x)
        conv_dilated = self.nConvs_dilated(x)
        return torch.cat([conv_standard, conv_dilated], dim=1)  # Concatenate along the channel dimension


class U2DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U2DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x_pooled = self.maxpool(x)
        return x, x_pooled


class U2UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U2UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip_input):
        x = self.up(x)
        x = torch.cat([x, skip_input], dim=1)  # Concatenate along channel dimension
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


# Update UCTransNet class
class UCTransNet2(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super(UCTransNet2, self).__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)

        # Modified down blocks with combined U2-Net strategy
        self.down1 = ModifiedDownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = ModifiedDownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down3 = ModifiedDownBlock(in_channels*8, in_channels*16, nb_Conv=2)
        self.down4 = ModifiedDownBlock(in_channels*16, in_channels*32, nb_Conv=2) # 注意这里的通道数倍增

        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels*4, in_channels*16, in_channels*32, in_channels*64],
                                     patchSize=config.patch_sizes)

        # Up-sampling blocks
        self.up4 = U2UpBlock(in_channels*32, in_channels*8)
        self.up3 = U2UpBlock(in_channels*16, in_channels*4)
        self.up2 = U2UpBlock(in_channels*8, in_channels*2)
        self.up1 = U2UpBlock(in_channels*4, in_channels)

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)

        # Down-sampling steps
        x2_standard, x2_dilated = self.down1(x1)
        x3_standard, x3_dilated = self.down2(torch.cat([x2_standard, x2_dilated], dim=1))
        x4_standard, x4_dilated = self.down3(torch.cat([x3_standard, x3_dilated], dim=1))
        x5_standard, x5_dilated = self.down4(torch.cat([x4_standard, x4_dilated], dim=1))

        # Channel Transformer
        x1,x2,x3,x4,att_weights = self.mtc(x1, x2_standard, x3_standard, x4_standard)

        # Up-sampling steps
        x = self.up4(torch.cat([x5_standard, x5_dilated], dim=1), x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        logits = self.last_activation(self.outc(x)) if self.n_classes == 1 else self.outc(x)

        return (logits, att_weights) if self.vis else logits






