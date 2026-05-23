import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LaneDataset(Dataset):

    def __init__(self,images_dir,white_masks_dir,yellow_masks_dir,image_files,transform=None):

        self.images_dir = images_dir

        self.white_masks_dir = white_masks_dir

        self.yellow_masks_dir = yellow_masks_dir

        self.image_files = image_files

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        img_path = os.path.join(
            self.images_dir,
            img_name
        )

        image = cv.imread(img_path)

        image = cv.cvtColor(
            image,
            cv.COLOR_BGR2RGB
        )

        raw_rgb = image.copy()

        white_path = os.path.join(
            self.white_masks_dir,
            img_name
        )

        yellow_path = os.path.join(
            self.yellow_masks_dir,
            img_name
        )

        white_mask = cv.imread(
            white_path,
            cv.IMREAD_GRAYSCALE
        )

        yellow_mask = cv.imread(
            yellow_path,
            cv.IMREAD_GRAYSCALE
        )

        if white_mask is None:

            white_mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8
            )

        if yellow_mask is None:

            yellow_mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8
            )

        white_mask = (
            white_mask > 0
        ).astype(np.uint8)

        yellow_mask = (
            yellow_mask > 0
        ).astype(np.uint8)


        lane_mask = np.logical_or(

            white_mask,

            yellow_mask

        ).astype(np.float32)

        # 0 background
        # 1 white
        # 2 yellow
        class_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        class_mask[
            white_mask == 1
        ] = 1

        class_mask[
            yellow_mask == 1
        ] = 2


        if self.transform:
            augmented = self.transform(
                image=image,
                masks=[
                    lane_mask,
                    class_mask
                ]
            )

            image = augmented['image']

            lane_mask = (
                augmented['masks'][0]
            )

            class_mask = (
                augmented['masks'][1]
            )


        raw_rgb = cv.resize(
            raw_rgb,
            (512,256)
        )

        raw_rgb = torch.from_numpy(
            raw_rgb
        ).permute(2,0,1).float()


        lane_mask = torch.tensor(
            lane_mask
        ).unsqueeze(0).float()

        class_mask = torch.tensor(
            class_mask,
            dtype=torch.long
        )

        return (image,raw_rgb,lane_mask,class_mask)