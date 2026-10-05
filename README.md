# U-LASTDE

Implementation of **U-LASTDE** for the paper:

> **Traffic Accident Detection on Road Networks: A U-Shaped Deep Model with Lane-Wise and Multidimensional Traffic Flow Data**  
> Yandong Li, Zhengchao Zhang, and Meng Li

U-LASTDE is a U-shaped spatiotemporal model for traffic accident detection on road networks. It combines lane fusion attention, spatiotemporal encoding and decoding, and differential gating to model lane-wise, multidimensional traffic observations.

## Installation

Use Python 3.10 or later and install the dependencies in a virtual environment:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with your CUDA runtime.

## Project structure

```text
U-LASTDE/
├── u_lastde/                 # Model, data loading, training, and evaluation
├── configs/default.json     # Default model and training settings
├── data/processed_dataset/  # train/, val/, and test/
├── checkpoints/             # Pretrained model weights
├── train.py                 # Command-line entry point
└── requirements.txt
```

## Data and checkpoints

Place the processed data under `data/processed_dataset/`. Each split contains the same four files:

```text
data/processed_dataset/
├── train/
├── val/
└── test/
    ├── traffic_data.npy
    ├── mask_data.npy
    ├── labels.npy
    └── adjacency_matrices.npy
```

The default configuration expects the following NumPy arrays, where `N` is the number of samples in a split:

| File | Shape |
| --- | --- |
| `traffic_data.npy` | `(N, 5, 60, 20, 7, 3)` |
| `mask_data.npy` | `(N, 5, 60, 20, 7)` |
| `labels.npy` | `(N, 60, 20, 2)` |
| `adjacency_matrices.npy` | `(N, 20, 20)` |

The pretrained checkpoint is included at `checkpoints/our_best.pth`; you can also supply another path with `--checkpoint`. The default configuration includes the paper model settings and evaluation thresholds.

## Usage

Run commands from the project root. Check the model with a single forward pass:

```bash
python train.py --dry-run --device cpu
```

Train and evaluate the best validation checkpoint:

```bash
python train.py --config configs/default.json
```

Evaluate a pretrained checkpoint:

```bash
python train.py --config configs/default.json --test-only --checkpoint checkpoints/our_best.pth
```

Use `--device cuda:0` or `--device cpu` to select a device. To use a different data location:

```bash
python train.py --data-dir /path/to/processed_dataset
```

Model and training parameters can be adjusted in `configs/default.json`. Run `python train.py --help` for the available command-line options.

## Outputs

Each run writes to `runs/U-LASTDE_<timestamp>/`. Training saves `best.pth`, a configuration snapshot, and loss histories. Evaluation writes `test_metrics.json` and `test_summary.txt`, including classification, regression, and spatiotemporal detection metrics.
