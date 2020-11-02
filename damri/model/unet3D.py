import torch
import torch.nn as nn

from dpipe.layers import PreActivation3d


class InitConv(nn.Module):
    def __init__(self, in_ch, ch1, ch2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, ch1, kernel_size=3, padding=1, bias=False)
        self.conv2 = PreActivation3d(in_channels=ch1, out_channels=ch2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = PreActivation3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, ch1, ch2):
        super().__init__()

        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            PreActivation3d(in_channels=in_ch, out_channels=ch1, kernel_size=3, padding=1, bias=False),
            PreActivation3d(in_channels=ch1, out_channels=ch2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.conv1 = PreActivation3d(in_channels=in_ch+skip_ch, out_channels=out_ch,
                                     kernel_size=3, padding=1, bias=False)
        self.conv2 = PreActivation3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        return self.conv2(self.conv1(x))


class UNet3D(nn.Module):
    def __init__(self, n_chans_in, n_chans_out):
        super().__init__()

        # initial convolution
        self.inconv = InitConv(in_ch=n_chans_in, ch1=32, ch2=64)

        # encoder part
        self.down1 = DownBlock(in_ch=64, ch1=64, ch2=128)
        self.down2 = DownBlock(in_ch=128, ch1=128, ch2=256)
        self.down3 = DownBlock(in_ch=256, ch1=256, ch2=512)

        # decoder part
        self.up1 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.up2 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up3 = UpBlock(in_ch=128, skip_ch=64, out_ch=64)

        # conv 1x1 as FCL
        self.outconv = OutConv(in_ch=64, out_ch=n_chans_out)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outconv(x)
        return x
