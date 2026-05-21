import os
import cv2 as cv
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import wandb
from kaggle_secrets import UserSecretsClient
from torch.utils.data import DataLoader

from model.Unet import LaneSegmentationNet
from model.LaneColourClassifier import LaneColorClassifier
from loss import DiceLoss
from dataset import LaneDataset
from helper import mask_to_color
from metrics import calculate_iou


CONFIG = {

    "epochs": 100,
    "batch_size": 16,
    "lr_unet": 1e-4,
    "lr_classifier": 1e-4,
    "image_height": 256,
    "image_width": 512,
    "encoder": "mobilenet_v2",
    # LOSS WEIGHTS
    "loss_weight_bg": 0.2,
    "loss_weight_white": 1.0,
    "loss_weight_yellow": 8.0,
    # AUGMENTATIONS
    "horizontal_flip_prob": 0.5,
    "brightness_limit": 0.4,
    "contrast_limit": 0.4,
    "brightness_contrast_prob": 0.7,
    "gaussian_noise_prob": 0.2,
    # NORMALIZATION
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225)
}

IMG_DIR = (
    '/kaggle/input/datasets/'
    'adityawaldia/clean-lanes/'
    'IGVC-LANES/images-1'
)

WHITE_MASK_DIR = (
    '/kaggle/input/datasets/'
    'adityawaldia/clean-lanes/'
    'IGVC-LANES/images-1-white'
)

YELLOW_MASK_DIR = (
    '/kaggle/input/datasets/'
    'adityawaldia/clean-lanes/'
    'IGVC-LANES/images-1-yellow'
)

def main():
    torch.backends.cudnn.benchmark = True
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret(
        "WANDB_API_KEY"
    )
    wandb.login(key=wandb_key)

    SEG_DEVICE = torch.device("cuda:0")

    CLS_DEVICE = torch.device("cuda:1")

    print("UNET Device:", SEG_DEVICE)
    print("Classifier Device:", CLS_DEVICE)

    wandb.init(project="hierarchical-lane-segmentation",name="multi_gpu_unet_classifier",config=CONFIG)
    all_images = sorted([

        f for f in os.listdir(IMG_DIR)

        if f.endswith(('.png','.jpg','.jpeg'))
    ])

    train_images, val_images = train_test_split(all_images,test_size=0.2,random_state=42)

    train_transform = A.Compose([
        A.Resize(CONFIG["image_height"],CONFIG["image_width"]),
        A.HorizontalFlip(p=CONFIG["horizontal_flip_prob"]),
        A.RandomBrightnessContrast(brightness_limit=CONFIG["brightness_limit"],contrast_limit=CONFIG["contrast_limit"],p=CONFIG["brightness_contrast_prob"]),
        A.GaussNoise(p=CONFIG["gaussian_noise_prob"]),
        A.Normalize(mean=CONFIG["mean"],std=CONFIG["std"]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(CONFIG["image_height"],CONFIG["image_width"]),
        A.Normalize(mean=CONFIG["mean"],std=CONFIG["std"]),
        ToTensorV2()
    ])

    train_dataset = LaneDataset(IMG_DIR,WHITE_MASK_DIR,YELLOW_MASK_DIR,train_images,train_transform)
    val_dataset = LaneDataset(IMG_DIR,WHITE_MASK_DIR,YELLOW_MASK_DIR,val_images,val_transform)
    train_loader = DataLoader(train_dataset,batch_size=CONFIG['batch_size'],shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset,batch_size=CONFIG['batch_size'],shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)

    unet_model = (LaneSegmentationNet().to(SEG_DEVICE))
    classifier_model = (LaneColorClassifier().to(CLS_DEVICE))

    wandb.watch(unet_model,log="all")
    wandb.watch(classifier_model,log="all")

    bce_loss = (nn.BCEWithLogitsLoss().to(SEG_DEVICE))
    dice_loss = (DiceLoss().to(SEG_DEVICE))
    classification_loss_fn = (nn.CrossEntropyLoss(weight=torch.tensor([CONFIG["loss_weights_bg"], CONFIG["loss_weights_white"], CONFIG["loss_weights_yellow"]]).to(CLS_DEVICE)))

    unet_optimizer = torch.optim.Adam(unet_model.parameters(),lr=CONFIG['lr_unet'])
    classifier_optimizer = torch.optim.Adam(classifier_model.parameters(),lr=CONFIG['lr_classifier'])

    EPOCHS = CONFIG['epochs']

    best_lane_miou = 0.0

    for epoch in range(EPOCHS):

        unet_model.train()

        classifier_model.train()

        total_unet_loss = 0.0

        total_classifier_loss = 0.0

        train_bar = tqdm(

            train_loader,

            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

    

        for (
            images,
            raw_rgb,
            lane_mask,
            class_mask
        ) in train_bar:

        

            images_seg = images.to(
                SEG_DEVICE,
                non_blocking=True
            )

            lane_mask_seg = lane_mask.to(
                SEG_DEVICE,
                non_blocking=True
            )

            # =================================================
            # MOVE TO GPU1
            # =================================================

            raw_rgb_cls = raw_rgb.to(
                CLS_DEVICE,
                non_blocking=True
            )

            lane_mask_cls = lane_mask.to(
                CLS_DEVICE,
                non_blocking=True
            )

            class_mask_cls = class_mask.to(
                CLS_DEVICE,
                non_blocking=True
            )
            unet_optimizer.zero_grad()

            segmentation_logits = (
                unet_model(images_seg)
            )

            geometry_loss = (

                0.5 * bce_loss(
                    segmentation_logits,
                    lane_mask_seg
                )

                +

                0.5 * dice_loss(
                    segmentation_logits,
                    lane_mask_seg
                )
            )

            geometry_loss.backward()

            unet_optimizer.step()


            classifier_optimizer.zero_grad()

            class_logits = classifier_model(

                raw_rgb_cls,

                lane_mask_cls
            )

            lane_pixels = (
                lane_mask_cls.squeeze(1) > 0
            )

            masked_logits = (

                class_logits.permute(
                    0,
                    2,
                    3,
                    1
                )[lane_pixels]
            )

            masked_targets = (
                class_mask_cls[
                    lane_pixels
                ]
            )

            classification_loss = (
                classification_loss_fn(

                    masked_logits,

                    masked_targets
                )
            )

            classification_loss.backward()

            classifier_optimizer.step()

            total_unet_loss += (
                geometry_loss.item()
            )

            total_classifier_loss += (
                classification_loss.item()
            )

            train_bar.set_postfix({

                "unet_loss":
                geometry_loss.item(),

                "classifier_loss":
                classification_loss.item()
            })

        avg_unet_loss = (
            total_unet_loss / len(train_loader)
        )

        avg_classifier_loss = (
            total_classifier_loss / len(train_loader)
        )

        unet_model.eval()

        classifier_model.eval()

        val_loss = 0.0

        total_iou = np.zeros(3)

        with torch.no_grad():

            for batch_idx, (

                images,
                raw_rgb,
                lane_mask,
                class_mask

            ) in enumerate(val_loader):


                images_seg = images.to(
                    SEG_DEVICE,
                    non_blocking=True
                )

                segmentation_logits = (
                    unet_model(images_seg)
                )

                pred_lane = (

                    torch.sigmoid(
                        segmentation_logits
                    ) > 0.5

                ).float()

                pred_lane = pred_lane.to(
                    CLS_DEVICE,
                    non_blocking=True
                )

                raw_rgb_cls = raw_rgb.to(
                    CLS_DEVICE,
                    non_blocking=True
                )

                class_mask_cls = class_mask.to(
                    CLS_DEVICE,
                    non_blocking=True
                )

                class_logits = classifier_model(

                    raw_rgb_cls,

                    pred_lane
                )

                preds = torch.argmax(
                    class_logits,
                    dim=1
                )

                preds[
                    pred_lane.squeeze(1) == 0
                ] = 0

                lane_pixels = (
                    pred_lane.squeeze(1) > 0
                )

                masked_logits = (

                    class_logits.permute(
                        0,
                        2,
                        3,
                        1
                    )[lane_pixels]
                )

                masked_targets = (
                    class_mask_cls[
                        lane_pixels
                    ]
                )

                if masked_targets.numel() > 0:

                    classification_loss = (
                        classification_loss_fn(

                            masked_logits,

                            masked_targets
                        )
                    )

                    val_loss += (
                        classification_loss.item()
                    )

                ious = calculate_iou(

                    preds,

                    class_mask_cls,

                    num_classes=3
                )

                total_iou += np.array(
                    ious
                )

                if batch_idx == 0:
                    wandb_images = []
                    for i in range(
                        min(4, images.size(0))
                    ):
                        rgb = raw_rgb[i].permute(
                            1,
                            2,
                            0
                        ).cpu().numpy().astype(np.uint8)

                        gt = mask_to_color(

                            class_mask[i]
                            .cpu()
                            .numpy()
                        )

                        pred = mask_to_color(

                            preds[i]
                            .cpu()
                            .numpy()
                        )

                        lane = (
                            pred_lane[i]
                            .squeeze()
                            .cpu()
                            .numpy() * 255
                        ).astype(np.uint8)

                        lane = cv.cvtColor(
                            lane,
                            cv.COLOR_GRAY2RGB
                        )

                        combined = np.concatenate(

                            [
                                rgb,
                                lane,
                                gt,
                                pred
                            ],

                            axis=1
                        )

                        wandb_images.append(

                            wandb.Image(

                                combined,

                                caption=(
                                    "RGB | LaneMask | GT | Prediction"
                                )
                            )
                        )

                    wandb.log({

                        "Validation Predictions":
                        wandb_images
                    })

        avg_val_loss = (
            val_loss / len(val_loader)
        )
        avg_iou = (
            total_iou / len(val_loader)
        )
        mean_iou = avg_iou.mean()
        lane_miou = (
            avg_iou[1] + avg_iou[2]
        ) / 2

        wandb.log({

            "unet_loss":
            avg_unet_loss,

            "classifier_loss":
            avg_classifier_loss,

            "val_classifier_loss":
            avg_val_loss,

            "background_iou":
            avg_iou[0],

            "white_lane_iou":
            avg_iou[1],

            "yellow_lane_iou":
            avg_iou[2],

            "mean_iou":
            mean_iou,

            "lane_miou":
            lane_miou
        })



        print("\n===================================")

        print(f"Epoch {epoch+1}")

        print(f"UNET Loss: {avg_unet_loss:.4f}")

        print(f"Classifier Loss: {avg_classifier_loss:.4f}")

        print(f"Validation Loss: {avg_val_loss:.4f}")

        print(f"Background IoU: {avg_iou[0]:.4f}")

        print(f"White Lane IoU: {avg_iou[1]:.4f}")

        print(f"Yellow Lane IoU: {avg_iou[2]:.4f}")

        print(f"Lane mIoU: {lane_miou:.4f}")

        # =====================================================
        # SAVE BEST
        # =====================================================

        if lane_miou > best_lane_miou:

            best_lane_miou = lane_miou

            torch.save(

                unet_model.state_dict(),

                "/kaggle/working/best_unet.pth"
            )

            torch.save(

                classifier_model.state_dict(),

                "/kaggle/working/best_classifier.pth"
            )

            wandb.save(
                "/kaggle/working/best_unet.pth"
            )

            wandb.save(
                "/kaggle/working/best_classifier.pth"
            )

            print("Saved Best Models!")

    print("\nTraining Complete!")

    wandb.finish()