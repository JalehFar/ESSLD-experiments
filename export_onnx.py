import argparse
from pathlib import Path

import numpy as np
import torch

import config
from models.dceunet import DCEUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Export ESSLD DCEUNet weights to ONNX.")
    parser.add_argument("--weights", default=config.MODEL_PATH, help="Path to the .pth weights.")
    parser.add_argument("--output", default="weights/dceunetex.onnx", help="Output ONNX path.")
    parser.add_argument("--height", type=int, default=config.IMAGE_HEIGHT, help="Input image height.")
    parser.add_argument("--width", type=int, default=config.IMAGE_WIDTH, help="Input image width.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--verify", action="store_true", help="Compare ONNX Runtime output with PyTorch.")
    return parser.parse_args()


def load_model(weights_path, device):
    model = DCEUNet(input_channels=3, num_classes=1).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def verify_export(onnx_path, torch_output, dummy_input):
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["logits"], {"input": dummy_input.cpu().numpy()})[0]
    max_abs_diff = np.max(np.abs(ort_output - torch_output.cpu().numpy()))
    print(f"ONNX Runtime max absolute difference: {max_abs_diff:.6f}")


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model = load_model(weights_path, device)
    dummy_input = torch.randn(1, 3, args.height, args.width, device=device)

    with torch.no_grad():
        torch_output = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )

    print(f"Exported ONNX model: {output_path}")
    print(f"Input shape: 1x3x{args.height}x{args.width}")
    print(f"Output shape: {'x'.join(str(dim) for dim in torch_output.shape)}")

    if args.verify:
        verify_export(output_path, torch_output, dummy_input)


if __name__ == "__main__":
    main()
