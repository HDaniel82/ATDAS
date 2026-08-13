"""
Shared bounding-box math used by the dataset encoder, the loss function, and inference."""

import torch


def grid_cell_to_corners(boxes: torch.Tensor, cell_i: torch.Tensor, cell_j: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert (x, y, w, h) predictions to (x1, y1, x2, y2) corners in image-normalized [0, 1] space.

    x, y are offsets within a grid cell (in [0, 1]); w, h are already relative to the
    full image. cell_i / cell_j are the column/row indices of the cell each box lives in.
    """
    x, y, w, h = boxes.unbind(-1)
    cx = (cell_i.to(boxes.dtype) + x) / grid_size
    cy = (cell_j.to(boxes.dtype) + y) / grid_size
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Elementwise IoU between two equal-shaped batches of (x1, y1, x2, y2) boxes."""
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    union = area1 + area2 - intersection

    return intersection / union.clamp(min=eps)


def make_cell_grid(grid_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cell_i, cell_j) index grids of shape (grid_size, grid_size)."""
    j, i = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )
    return i, j
