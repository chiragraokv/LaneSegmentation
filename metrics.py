import torch

def calculate_iou(preds,targets,num_classes=3):

    ious = []

    for cls in range(num_classes):

        pred_cls = (
            preds == cls
        )

        target_cls = (
            targets == cls
        )

        intersection = (
            pred_cls & target_cls
        ).sum().float()

        union = (
            pred_cls | target_cls
        ).sum().float()

        if union == 0:

            iou = torch.tensor(1.0)

        else:

            iou = (
                intersection / union
            )

        ious.append(
            iou.item()
        )

    return ious
