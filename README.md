# Efficient Sea-Sky Line Detection

This repository contains the official implementation code for the paper: "Efficient Sea-Sky Line Detection via Lightweight Segmentation and Dual Multi-Scale Fusion".
The system consists of a lightweight semantic segmentation network (DCEUNet) followed by a robust dual multi-scale fusion refinement module.

This repository is a modified fork of [kaibaJC/ESSLD](https://github.com/kaibaJC/ESSLD), with added ONNX export and verification support.

## 📝 Project Overview

This codebase serves as a consolidated reference implementation derived from the experimental framework used in the paper. It has been streamlined to facilitate the fundamental reproduction of the core algorithms (DCEUNet architecture + Fusion logic) while maintaining code clarity and ease of use for the research community.

## 📂 Project Structure

```text
.
├── config.py           # Configuration parameters (Generic defaults)
├── demo.py             # Image inference script (Visualization mode)
├── demo_video.py       # Real-time video inference player
├── models/             # DCEUNet and attention modules
├── utils/              # Refinement algorithms (HLDA, Geometric, etc.)
├── weights/            # Pre-trained models
├── samples/            # Test samples (images and video clips)
``` 

## 📷 Sample Data

The sample data provided in the samples/ directory are extracted from publicly available maritime video datasets used in the paper's experiments. They are provided here solely for the purpose of demonstrating the inference pipeline.

## 🚀 Quick Start

### 1. Requirements

Install dependencies using pip:

**pip install -r requirements.txt**

### 2. Run Image Demo (Instant View)

Run the script to visualize the segmentation mask and the refined horizon line side-by-side.

**python demo.py samples/test_1.png**

Output: A window will pop up showing the Coarse Mask (Left) and the Final Detection Result (Right).

Controls: Press any key to close the window.

### 3. Run Video Demo (Real-Time)

To test the detection performance on a video file. This script runs in FP16 half-precision mode (if CUDA is available) to demonstrate real-time capability.

**python demo_video.py --input samples/test_video.mp4**

Controls: Press 'Q' to quit playback.

### 4. Export ONNX Model

Export the provided PyTorch weights to ONNX:

**python export_onnx.py --verify**

This creates:

**weights/dceunetex.onnx**

To compare the PyTorch and ONNX outputs on a sample image:

**python compare_pth_onnx.py samples/test_1.png**

The comparison script reports logit/probability differences and binary mask pixel differences, and saves visual comparisons under **outputs/onnx_compare/**.

## 📊 Performance & Reproducibility

Lightweight Architecture: The provided DCEUNet features <0.4M parameters, specifically designed for edge devices.

FP16 Optimization: The video demo supports half-precision (FP16) inference on CUDA devices, significantly boosting FPS for real-time applications.

## ⚠️ Limitations of the Demo & Usage Notes

To ensure operational convenience and broad accessibility, this repository provides simplified inference demos. Please consider the following when reproducing results:

Image-Based vs. Video-Based: For ease of access, the core evaluation focuses on image-based metrics. The results reported in the paper (e.g., on MU-SID/SMD/BD/TMD datasets) leverage temporal consistency (e.g., tracking loops) to stabilize detection across continuous frames. This demo operates on a frame-by-frame basis to verify the core algorithm without the overhead of complex temporal modules.

Hardware-Dependent Tuning: Due to variations in computational performance (e.g., GPU memory bandwidth), the default parameters in config.py are set to generic robust values. To achieve optimal performance or match the specific results in the paper, users may need to fine-tune these parameters according to their specific hardware capabilities and environmental conditions.

Parameter Sensitivity: The fusion parameters were optimized for specific environmental conditions (e.g., heavy fog vs. clear day) in the paper's experiments. If testing on samples with significantly different distributions, parameter adjustment is recommended.

## 📜 Citation

If you find this work useful, please cite our paper:

@article{chen2025SeaSky,
  title={Efficient Sea-Sky Line Detection via Lightweight Segmentation and Dual Multi-Scale Fusion},
  author={Chen, Jialuo and Hu, Zhiwu},
  journal={The Visual Computer},
  year={2025}
}

## 📄 License

This project is released under the **MIT License**.

The code is provided "as is", without warranty of any kind. Users are responsible for parameter tuning and validation in their specific environments.

