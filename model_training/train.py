"""
Training loop for the SentinelDetector grid-based object detector.

Usage:
    python train.py --data data.yaml --epochs 100 --batch-size 16
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from architecture import SentinelDetector
from dataset import ATDASDetectionDataset, load_data_config
from loss_functions import DetectionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ATDAS grid-based detector")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"total": 0.0, "box_loss": 0.0, "obj_loss": 0.0, "noobj_loss": 0.0, "class_loss": 0.0}
    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            losses = loss_fn(predictions, targets)

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            for key in totals:
                totals[key] += losses[key].item()

    return {key: value / len(loader) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_data_config(args.data)
    class_names = config["names"]
    data_root = config["path"]

    device = torch.device(args.device)

    train_set = ATDASDetectionDataset(
        data_root, "train", class_names, img_size=args.img_size, grid_size=args.grid_size, augment=True
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )

    val_loader = None
    val_split_dir = Path(data_root) / "images" / "val"
    if val_split_dir.is_dir():
        val_set = ATDASDetectionDataset(
            data_root, "val", class_names, img_size=args.img_size, grid_size=args.grid_size, augment=False
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = SentinelDetector(num_classes=len(class_names), grid_size=args.grid_size).to(device)
    loss_fn = DetectionLoss(grid_size=args.grid_size, num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        log = f"epoch {epoch + 1}/{args.epochs} | train_loss {train_metrics['total']:.4f}"

        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, loss_fn, device)
            log += f" | val_loss {val_metrics['total']:.4f}"
        print(log)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {
                "num_classes": len(class_names),
                "class_names": class_names,
                "grid_size": args.grid_size,
                "img_size": args.img_size,
            },
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")

        current_val_loss = val_metrics["total"] if val_metrics else train_metrics["total"]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
