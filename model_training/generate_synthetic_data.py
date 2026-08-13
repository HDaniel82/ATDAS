"""Generate a tiny synthetic image/label dataset for smoke-testing the training pipeline
before real sensor data is available. Draws random colored rectangles as stand-ins for
detection targets, one class per color.

Usage:
    python generate_synthetic_data.py --out data/synthetic --train 40 --val 10
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_NAMES = ["red_target", "green_target", "blue_target"]
CLASS_COLORS = [(220, 40, 40), (40, 200, 80), (60, 90, 220)]


def generate_split(out_root: Path, split: str, count: int, img_size: int, seed: int) -> None:
    image_dir = out_root / "images" / split
    label_dir = out_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for idx in range(count):
        image = Image.new("RGB", (img_size, img_size), (30, 30, 30))
        draw = ImageDraw.Draw(image)

        num_objects = rng.randint(1, 3)
        lines = []
        for _ in range(num_objects):
            class_id = rng.randrange(len(CLASS_NAMES))
            w = rng.randint(img_size // 8, img_size // 4)
            h = rng.randint(img_size // 8, img_size // 4)
            x1 = rng.randint(0, img_size - w)
            y1 = rng.randint(0, img_size - h)
            draw.rectangle([x1, y1, x1 + w, y1 + h], fill=CLASS_COLORS[class_id])

            cx, cy = (x1 + w / 2) / img_size, (y1 + h / 2) / img_size
            norm_w, norm_h = w / img_size, h / img_size
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        image.save(image_dir / f"{split}_{idx:04d}.jpg")
        (label_dir / f"{split}_{idx:04d}.txt").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic smoke-test data")
    parser.add_argument("--out", type=str, default="data/synthetic")
    parser.add_argument("--train", type=int, default=40)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=416)
    args = parser.parse_args()

    out_root = Path(args.out)
    generate_split(out_root, "train", args.train, args.img_size, seed=0)
    generate_split(out_root, "val", args.val, args.img_size, seed=1)

    data_yaml = out_root / "data.yaml"
    names_list = "\n".join(f"  - {name}" for name in CLASS_NAMES)
    data_yaml.write_text(f"path: {out_root.resolve().as_posix()}\nnames:\n{names_list}\n")

    print(f"Synthetic dataset written to {out_root}")
    print(f"Data config: {data_yaml}")


if __name__ == "__main__":
    main()
