import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class LaneSegmentationNet(nn.Module):
    def __init__(self):

        super().__init__()
        self.unet = smp.Unet(encoder_name="mobilenet_v2",encoder_weights="imagenet",in_channels=3,classes=1)

    def forward(self,x):
        return self.unet(x)