"""
Custom detection loss: IoU loss for box localization, SSE for objectness/class.

Follows the YOLOv1 weighting scheme (lambda_coord / lambda_noobj) but replaces the
sqrt(w)/sqrt(h) SSE coordinate trick with a direct (1 - IoU) localization term, which
is scale-invariant and avoids the sqrt gradient blowup near zero.
"""

import torch
from torch import nn

from bbox_utils import box_iou, grid_cell_to_corners, make_cell_grid


class DetectionLoss(nn.Module):
    def __init__(self, grid_size: int, num_classes: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        cell_i, cell_j = make_cell_grid(grid_size)
        self.register_buffer("cell_i", cell_i)
        self.register_buffer("cell_j", cell_j)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """predictions, targets: (N, S, S, 5 + C)."""
        obj_mask = targets[..., 0] == 1
        noobj_mask = ~obj_mask

        pred_obj = predictions[..., 0]
        target_obj = targets[..., 0]

        obj_loss = torch.sum((pred_obj[obj_mask] - target_obj[obj_mask]) ** 2)
        noobj_loss = torch.sum((pred_obj[noobj_mask] - target_obj[noobj_mask]) ** 2)

        if obj_mask.any():
            cell_i = self.cell_i.unsqueeze(0).expand_as(pred_obj)[obj_mask]
            cell_j = self.cell_j.unsqueeze(0).expand_as(pred_obj)[obj_mask]

            pred_corners = grid_cell_to_corners(predictions[..., 1:5][obj_mask], cell_i, cell_j, self.grid_size)
            target_corners = grid_cell_to_corners(targets[..., 1:5][obj_mask], cell_i, cell_j, self.grid_size)
            iou = box_iou(pred_corners, target_corners)
            box_loss = torch.sum(1.0 - iou)

            pred_classes = predictions[..., 5:][obj_mask]
            target_classes = targets[..., 5:][obj_mask]
            class_loss = torch.sum((pred_classes - target_classes) ** 2)
        else:
            box_loss = predictions.sum() * 0.0
            class_loss = predictions.sum() * 0.0

        batch_size = predictions.shape[0]
        total = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        ) / batch_size

        return {
            "total": total,
            "box_loss": box_loss / batch_size,
            "obj_loss": obj_loss / batch_size,
            "noobj_loss": noobj_loss / batch_size,
            "class_loss": class_loss / batch_size,
        }
