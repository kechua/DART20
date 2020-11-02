import torch
import torch.nn as nn
from dpipe.layers.conv import PostActivation2d


class VanillaUNet(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, n_filters_init=64):
        super().__init__()
        n = n_filters_init

        self.init_path = nn.Sequential(
            PostActivation2d(n_chans_in, n, kernel_size=3, padding=1),
            PostActivation2d(n, n, kernel_size=3, padding=1),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            PostActivation2d(n, n * 2, kernel_size=3, padding=1),
            PostActivation2d(n * 2, n * 2, kernel_size=3, padding=1),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            PostActivation2d(n * 2, n * 4, kernel_size=3, padding=1),
            PostActivation2d(n * 4, n * 4, kernel_size=3, padding=1),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            PostActivation2d(n * 4, n * 8, kernel_size=3, padding=1),
            PostActivation2d(n * 8, n * 8, kernel_size=3, padding=1),
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            PostActivation2d(n * 8, n * 16, kernel_size=3, padding=1),
        )
        self.up4 = nn.Sequential(
            PostActivation2d(n * 16, n * 16, kernel_size=3, padding=1),
            nn.ConvTranspose2d(n * 16, n * 8, kernel_size=2, stride=2, bias=False),
        )
        self.up3 = nn.Sequential(
            PostActivation2d(n * 16, n * 8, kernel_size=3, padding=1),
            PostActivation2d(n * 8, n * 8, kernel_size=3, padding=1),
            nn.ConvTranspose2d(n * 8, n * 4, kernel_size=2, stride=2, bias=False),
        )
        self.up2 = nn.Sequential(
            PostActivation2d(n * 8, n * 4, kernel_size=3, padding=1),
            PostActivation2d(n * 4, n * 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(n * 4, n * 2, kernel_size=2, stride=2, bias=False),
        )
        self.up1 = nn.Sequential(
            PostActivation2d(n * 4, n * 2, kernel_size=3, padding=1),
            PostActivation2d(n * 2, n * 2, kernel_size=3, padding=1),
            nn.ConvTranspose2d(n * 2, n, kernel_size=2, stride=2, bias=False),
        )
        self.out_path = nn.Sequential(
            PostActivation2d(n * 2, n, kernel_size=3, padding=1),
            PostActivation2d(n, n, kernel_size=3, padding=1),
            nn.Conv2d(n, n_chans_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_chans_out),
        )

    def forward(self, x):
        x0 = self.init_path(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x4 = self.down4(x3)
        x3_up = self.up4(x4)

        x2_up = self.up3(torch.cat([x3, x3_up], dim=1))
        x1_up = self.up2(torch.cat([x2, x2_up], dim=1))
        x0_up = self.up1(torch.cat([x1, x1_up], dim=1))
        x_out = self.out_path(torch.cat([x0, x0_up], dim=1))
        return x_out
