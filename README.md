# U-LASTDE

Implementation of **U-LASTDE** for the paper:

> **Traffic Accident Detection on Road Networks: A U-Shaped Deep Model with Lane-Wise and Multidimensional Traffic Flow Data**  
> Yandong Li, Zhengchao Zhang, and Meng Li

U-LASTDE is a U-shaped spatiotemporal model for traffic accident detection on road networks. It combines lane fusion attention, spatiotemporal encoding and decoding, and differential gating to model lane-wise, multidimensional traffic observations.

## Installation

Clone the repository and create a virtual environment with Python 3.10 or later:

```bash
git clone https://github.com/hahaxs0514/U-LASTDE.git
cd U-LASTDE
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
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

Download [`processed_dataset.zip` from Google Drive](https://drive.google.com/file/d/1IvN61ber8X4d_RCnULB5feISaoZhIaMz/view?usp=drive_link). Extract its contents into `data/processed_dataset/` in the cloned repository. The archive already contains the `train/`, `val/`, and `test/` folders at its top level, so the resulting paths should look like this:

```text
data/processed_dataset/
├── train/
│   ├── traffic_data.npy
│   ├── mask_data.npy
│   ├── labels.npy
│   └── adjacency_matrices.npy
├── val/                     # Same four files
└── test/                    # Same four files
```

For example, `data/processed_dataset/test/traffic_data.npy` should exist after extraction. Avoid an extra `processed_dataset/` folder inside that directory.

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
