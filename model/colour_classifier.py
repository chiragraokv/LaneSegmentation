import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneColorClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(64,3,1,padding=1),
        )

    def forward(self,raw_rgb,lane_mask):

        raw_rgb = raw_rgb / 255.0

        masked_rgb = (raw_rgb * lane_mask)

        return self.network(masked_rgb)

class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.block = nn.Sequential(

            # Depthwise
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),

            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # Pointwise
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class CClassifier_edge(nn.Module):
    def __init__(self, num_classes=3):

        super().__init__()
        self.enc1 = DSConv(3, 16)

        self.down1 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.enc2 = DSConv(32, 32)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=32,
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 1, bias=False),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )
        self.fuse = DSConv(16 + 16, 16)

        self.final = nn.Conv2d(
            16,
            num_classes,
            kernel_size=1
        )

    def forward(self, raw_rgb, lane_mask):
        raw_rgb = raw_rgb.float() / 255.0
        lane_mask = lane_mask.float()
        x = raw_rgb * lane_mask
        e1 = self.enc1(x)
        x = self.down1(e1)
        x = F.relu(x)
        x = self.enc2(x)
        x = self.bottleneck(x)
        x = self.up(x)
        x = torch.cat([x, e1], dim=1)
        x = self.fuse(x)
        x = self.final(x)
        return x