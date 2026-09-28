import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict, List, Literal

import torch


class Config:
    """Model, data, and runtime settings for U-LASTDE."""

    def __init__(self):
        self.model_name = "U-LASTDE"
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.output_root = os.path.join(self.project_root, "data", "processed_dataset")
        self.save_root = os.path.join(self.project_root, "runs")

        # Paper checkpoint: checkpoints/our_best.pth.
        self.threshold_mode = "threshold"
        self.threshold = [0.2, 0.09]

        self.n_days_original = 6
        self.n_days_normal = 4
        self.n_days = 5
        self.time_steps = 60
        self.seq_length = 60
        self.n_segments = 20
        self.n_nodes = 20
        self.n_lanes = 7
        self.n_features = 3
        self.n_anomaly_types = 2
        self.output_dim = 2
        self.horizon = 60
        self.forecast_dim = 256

        self.random_seed = 42
        self.normalization = "standardization"

        self.lane_fusion_type: Literal["statistics", "attention", "deepsets", "attn_weighted_sum"] = "attention"
        self.lane_cnn_channels: List[int] = [32, 64]
        self.use_all_days = True
        self.use_temporal_cnn = True
        self.day_dropout = 0.1
        self.lane_dropout = 0.1
        self.st_dropout = 0.1
        self.lane_cnn_kernel = 5
        self.lane_num_layers = 2
        self.lane_num_heads = 4
        self.deepsets_pool = "mean"
        self.statistics_eps = 1e-6
        self.abnormal_day_index = 0

        self.num_encoder_blocks = 2
        self.channel_multiplier = 2
        self.activation = "gelu"
        self.temporal_variant = "inception"
        self.temporal_kernel = 3
        self.temporal_kernel_set = [3, 5]
        self.two_conv_residual = True
        self.spatial_mode = "attnA"
        self.spatial_K = 1
        self.spatial_num_layers = 1
        self.spatial_separate_dirs = True
        self.spatial_bias = True
        self.spatial_num_heads = 4
        self.spatial_clamp_val = 6.0
        self.spatial_eps = 1e-6

        self.batch_size = 32
        self.epochs = 200
        self.learning_rate = 1e-3
        self.weight_decay = 1e-7
        self.grad_clip = None
        self.loss_type = "mse"
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.early_stopping_patience = 20
        self.save_interval = 10
        self.print_interval = 10
        self.num_workers = 0
        self.pin_memory = True
        self.use_dataparallel = False
        self.gpu_ids = [0]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.save_dir = None

        self.deepsets_hidden = self.lane_cnn_channels[-1] * 2
        self.lane_attn_hidden = self.lane_cnn_channels[-1] * 2

    def _resolve_path(self, path_value, base_dir):
        if path_value is None:
            return None
        path_value = os.path.expanduser(str(path_value))
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(base_dir, path_value))

    def apply(self, values: Dict[str, Any]):
        for key, value in values.items():
            setattr(self, key, value)
        self._sync_derived()
        return self

    def _sync_derived(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.output_root = self._resolve_path(self.output_root, self.project_root)
        self.save_root = self._resolve_path(self.save_root, self.project_root)
        if self.save_dir is not None:
            self.save_dir = self._resolve_path(self.save_dir, self.project_root)
        self.seq_length = getattr(self, "seq_length", self.time_steps) or self.time_steps
        self.n_nodes = getattr(self, "n_nodes", self.n_segments) or self.n_segments
        self.output_dim = getattr(self, "output_dim", self.n_anomaly_types) or self.n_anomaly_types
        self.horizon = getattr(self, "horizon", self.time_steps) or self.time_steps
        self.forecast_dim = getattr(self, "forecast_dim", 256) or 256
        self.lane_outdim = self.lane_cnn_channels[-1]
        self.deepsets_hidden = getattr(self, "deepsets_hidden", self.lane_cnn_channels[-1] * 2)
        self.lane_attn_hidden = getattr(self, "lane_attn_hidden", self.lane_cnn_channels[-1] * 2)
        if not isinstance(self.device, torch.device):
            self.device = torch.device(str(self.device) if str(self.device) != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    def finalize(self):
        self._sync_derived()
        if self.save_dir is None:
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.save_dir = os.path.join(self.save_root, f"{self.model_name}_{stamp}")
        return self

    def get_channel_dims(self):
        start_dim = self.lane_cnn_channels[-1]
        return [start_dim * (2 ** i) for i in range(self.num_encoder_blocks + 1)]

    def validate(self, require_data=True):
        if self.model_name.lower() not in {"our", "u-lastde", "u_lastde"}:
            raise ValueError("model_name must be U-LASTDE")
        if require_data and not os.path.isdir(self.output_root):
            raise FileNotFoundError(f"Data directory does not exist: {self.output_root}")
        for name in ("n_days", "time_steps", "n_segments", "n_lanes", "n_features", "n_anomaly_types", "batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.abnormal_day_index < self.n_days:
            raise ValueError("abnormal_day_index must index an input day")
        return True

    @classmethod
    def from_json(cls, path):
        cfg = cls()
        if path:
            with open(cfg._resolve_path(path, cfg.project_root), "r", encoding="utf-8") as f:
                cfg.apply(json.load(f))
        return cfg.finalize()

    def __repr__(self):
        keys = [k for k in sorted(self.__dict__) if not k.startswith("_")]
        return "\n".join(["U-LASTDE configuration:"] + [f"  {k}: {getattr(self, k)}" for k in keys])


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate U-LASTDE")
    parser.add_argument("--config", default="configs/default.json", help="JSON configuration path")
    parser.add_argument("--model", default=None, help="Override model_name")
    parser.add_argument("--data-dir", default=None, help="Override the processed dataset directory")
    parser.add_argument("--save-dir", default=None, help="Directory for this run's outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run one forward pass only")
    parser.add_argument("--test-only", action="store_true", help="Load a checkpoint and run the test split only")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for --test-only")
    return parser.parse_args()


def load_config_from_args(args=None):
    args = args or parse_args()
    cfg = Config.from_json(args.config)
    overrides = {}
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.data_dir is not None:
        overrides["output_root"] = args.data_dir
    if args.save_dir is not None:
        overrides["save_dir"] = args.save_dir
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if overrides:
        cfg.apply(overrides).finalize()
    cfg.validate(require_data=not args.dry_run)
    if args.checkpoint is not None:
        args.checkpoint = cfg._resolve_path(args.checkpoint, cfg.project_root)
    return cfg, args
