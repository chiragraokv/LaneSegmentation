import numpy as np
import torch

def mask_to_color(mask):

    colored = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3
        ),
        dtype=np.uint8
    )

    colored[
        mask == 1
    ] = [255,255,255]

    colored[
        mask == 2
    ] = [255,255,0]

    return colored