import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

import config
from models.dceunet import DCEUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ESSLD PyTorch and ONNX outputs.")
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument("--weights", default=config.MODEL_PATH, help="Path to the .pth weights.")
    parser.add_argument("--onnx", default="weights/dceunetex.onnx", help="Path to the ONNX model.")
    parser.add_argument("--height", type=int, default=config.IMAGE_HEIGHT, help="Input image height.")
    parser.add_argument("--width", type=int, default=config.IMAGE_WIDTH, help="Input image width.")
    parser.add_argument("--output-dir", default="outputs/onnx_compare", help="Directory for comparison images.")
    return parser.parse_args()


def load_image(image_path, height, width):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (width, height))
    input_array = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return image_bgr, input_array[None, ...]


def run_pytorch(weights_path, input_array):
    device = torch.device("cpu")
    model = DCEUNet(input_channels=3, num_classes=1).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    input_tensor = torch.from_numpy(input_array).to(device)
    with torch.no_grad():
        logits = model(input_tensor).cpu().numpy()
    return logits


def run_onnx(onnx_path, input_array):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(["logits"], {"input": input_array})[0]


def mask_from_logits(logits):
    prob = 1.0 / (1.0 + np.exp(-logits))
    mask = (prob[0, 0] > 0.5).astype(np.uint8) * 255
    return prob[0, 0], mask


def save_visual_comparison(image_path, output_dir, pth_mask, onnx_mask, diff_mask):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    cv2.imwrite(str(output_dir / f"{stem}_pth_mask.png"), pth_mask)
    cv2.imwrite(str(output_dir / f"{stem}_onnx_mask.png"), onnx_mask)
    cv2.imwrite(str(output_dir / f"{stem}_diff.png"), diff_mask)

    comparison = np.hstack(
        [
            cv2.cvtColor(pth_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(onnx_mask, cv2.COLOR_GRAY2BGR),
            cv2.applyColorMap(diff_mask, cv2.COLORMAP_JET),
        ]
    )
    cv2.imwrite(str(output_dir / f"{stem}_comparison.png"), comparison)


def main():
    args = parse_args()
    image_path = Path(args.image)
    weights_path = Path(args.weights)
    onnx_path = Path(args.onnx)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    _, input_array = load_image(image_path, args.height, args.width)
    pth_logits = run_pytorch(weights_path, input_array)
    onnx_logits = run_onnx(onnx_path, input_array)

    pth_prob, pth_mask = mask_from_logits(pth_logits)
    onnx_prob, onnx_mask = mask_from_logits(onnx_logits)

    logits_abs_diff = np.abs(pth_logits - onnx_logits)
    prob_abs_diff = np.abs(pth_prob - onnx_prob)
    different_pixels = int(np.count_nonzero(pth_mask != onnx_mask))
    total_pixels = int(pth_mask.size)
    different_percent = 100.0 * different_pixels / total_pixels

    diff_mask = np.abs(pth_mask.astype(np.int16) - onnx_mask.astype(np.int16)).astype(np.uint8)
    save_visual_comparison(image_path, output_dir, pth_mask, onnx_mask, diff_mask)

    print(f"Image: {image_path}")
    print(f"PyTorch logits shape: {pth_logits.shape}")
    print(f"ONNX logits shape: {onnx_logits.shape}")
    print(f"Logits max abs diff: {logits_abs_diff.max():.8f}")
    print(f"Logits mean abs diff: {logits_abs_diff.mean():.8f}")
    print(f"Probability max abs diff: {prob_abs_diff.max():.8f}")
    print(f"Probability mean abs diff: {prob_abs_diff.mean():.8f}")
    print(f"Binary mask different pixels: {different_pixels}/{total_pixels} ({different_percent:.6f}%)")
    print(f"Comparison images saved to: {output_dir}")


if __name__ == "__main__":
    main()
