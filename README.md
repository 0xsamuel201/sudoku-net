# Sudoku-net: real-time sudoku solver with deep learning

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED.svg?style=flat&logo=ONNX&logoColor=white)](https://onnx.ai/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.09%25-brightgreen.svg)]()
[![FPS](https://img.shields.io/badge/CPU_FPS-14547-blue.svg)]()

This repository demonstrates an end-to-end Machine Learning pipeline, from designing a lightweight Custom CNN architecture to deploying it for high-throughput inference using ONNX Runtime.

The project features **MiniMobileNet**, a heavily optimized variant of MobileNetV1 explicitly designed for 28x28 grayscale images (MNIST). It removes redundant layers and bottlenecks, making it an ideal foundational architecture for edge-deployed Optical Character Recognition (OCR) systems.

## Key Features

- **Custom Edge Architecture:** Stripped down 13-block MobileNet to a highly efficient 5-block `MiniMobileNet`, reducing parameters from ~3.2M to ~150K.
- **Native Resolution Processing:** Bypasses the standard 224x224 resize, processing 28x28 images directly to maximize throughput.
- **Production-Ready Export:** Utilizes PyTorch 2.x `onnxscript` and `opset_version=18` for modern, stable, and dynamic-batch ONNX generation.
- **Comprehensive Benchmarking:** Includes a robust evaluation script to verify mathematical correctness and measure latency/throughput against the full 10,000-image test set.

## Benchmark Results (CPU)

A head-to-head comparison between the native PyTorch engine and ONNX Runtime (`CPUExecutionProvider`) over the entire MNIST Test Dataset (10,000 images, Batch Size = 64).

> Note: Benchmark results below are on my local machine macbook air M2

### Accuracy Validation

Both models retain identical predictive performance, proving the mathematical integrity of the ONNX graph translation.

- **PyTorch Accuracy:** `99.09 %` (9909 / 10000)
- **ONNX Accuracy:** `99.09 %` (9909 / 10000)

### Speed & Throughput

ONNX Runtime significantly accelerates inference through operator fusion and removal of autograd overhead.

- **PyTorch Total Time:** `4.94 seconds` (~2023 FPS)
- **ONNX Total Time:** `0.69 seconds` (~14547 FPS)

> **Conclusion:** ONNX Runtime processes images **7.19x faster** than native PyTorch on the CPU without any degradation in accuracy.

## Project Structure

- `mobilenet.py`: Contains the model architecture (`DepthwiseSeparableConv` and `MiniMobileNet`).
- `train.py`: The training pipeline with CLI arguments, automatic checkpointing, and matplotlib metric tracking.
- `export_onnx.py`: Script to convert the `.pth` weights into an optimized `.onnx` graph.
- `benchmark_dataset.py`: The evaluation script comparing the two engines.

## Quick Start

**1. Install Dependencies**

```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision onnx onnxscript onnxruntime matplotlib tqdm
```

**2. Training**

```bash
python train.py --epochs 30 --batch-size 64 --lr 0.0005 --weight-file output/sudoku-net.pth --plot-file output/report.png
```

**3. Export to ONNX**

```bash
uv run export_onnx.py
```

**4. Benchmark**

```bash
uv run benchmark.py
```

**5. Compare Inference speed**

```bash
uv run compare_inference.py
```
