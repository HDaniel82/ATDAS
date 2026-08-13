"""Dataset pipeline: YOLO-format images/labels -> (image_tensor, grid_target_tensor) pairs.

Expected layout, given a root directory and a split name (e.g. "train", "val"):
    root/images/<split>/*.jpg
    root/labels/<split>/*.txt

Each label file has one row per object: "class_id cx cy w h", all normalized to [0, 1]
relative to image width/height (standard YOLO format). Images without a matching label
file are treated as background-only (no objects).
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_data_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


class ATDASDetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        class_names: list[str],
        img_size: int = 416,
        grid_size: int = 13,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.img_size = img_size
        self.grid_size = grid_size
        self.augment = augment

        image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_labels(self, image_path: Path) -> list[tuple[int, float, float, float, float]]:
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            return []

        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                class_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                labels.append((class_id, cx, cy, w, h))
        return labels

    def _build_target(self, labels: list[tuple[int, float, float, float, float]]) -> torch.Tensor:
        target = torch.zeros(self.grid_size, self.grid_size, 5 + self.num_classes)
        for class_id, cx, cy, w, h in labels:
            col = min(int(cx * self.grid_size), self.grid_size - 1)
            row = min(int(cy * self.grid_size), self.grid_size - 1)
            if target[row, col, 0] == 1:
                continue  # cell already has an assigned object (single-box-per-cell)

            x_cell = cx * self.grid_size - col
            y_cell = cy * self.grid_size - row
            target[row, col, 0] = 1.0
            target[row, col, 1:5] = torch.tensor([x_cell, y_cell, w, h])
            target[row, col, 5 + class_id] = 1.0
        return target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size))
        labels = self._read_labels(image_path)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            labels = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in labels]

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

        target_tensor = self._build_target(labels)
        return image_tensor, target_tensor
