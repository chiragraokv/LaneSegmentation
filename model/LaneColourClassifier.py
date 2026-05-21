import torch
import torch.nn as nn

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
