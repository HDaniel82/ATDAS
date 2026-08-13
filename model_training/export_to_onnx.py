"""Export a trained SentinelDetector checkpoint to ONNX for the C++ inference engine.

Usage:
    python export_to_onnx.py --checkpoint checkpoints/best.pt --output ../inference_engine/models/sentinel_v1.onnx
"""

import argparse

import numpy as np
import onnx
import onnxruntime
import torch

from architecture import SentinelDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="../inference_engine/models/sentinel_v1.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]

    model = SentinelDetector(num_classes=config["num_classes"], grid_size=config["grid_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img_size = config["img_size"]
    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    session = onnxruntime.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-5)
    print(f"Exported and verified: {args.output}")
    print(f"Input:  (batch, 3, {img_size}, {img_size})")
    print(f"Output: (batch, {config['grid_size']}, {config['grid_size']}, {5 + config['num_classes']})")
    print(f"Classes: {config['class_names']}")


if __name__ == "__main__":
    main()
