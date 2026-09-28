import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficAnomalyDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.data_dir = os.path.join(config.output_root, split)
        required = ["traffic_data.npy", "mask_data.npy", "labels.npy", "adjacency_matrices.npy"]
        missing = [name for name in required if not os.path.exists(os.path.join(self.data_dir, name))]
        if missing:
            raise FileNotFoundError(f"Missing files in {self.data_dir}: {missing}")
        self.traffic_data = np.load(os.path.join(self.data_dir, "traffic_data.npy"), mmap_mode="r")
        self.mask_data = np.load(os.path.join(self.data_dir, "mask_data.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(self.data_dir, "labels.npy"), mmap_mode="r")
        self.adjacency_matrices = np.load(os.path.join(self.data_dir, "adjacency_matrices.npy"), mmap_mode="r")
        self._validate_shapes()
        print(f"Loaded {split}: traffic={self.traffic_data.shape}, labels={self.labels.shape}")

    def _validate_shapes(self):
        if self.traffic_data.ndim != 6:
            raise ValueError(f"traffic_data must be 6D, got {self.traffic_data.shape}")
        if self.mask_data.shape != self.traffic_data.shape[:-1]:
            raise ValueError(f"mask shape {self.mask_data.shape} does not match traffic {self.traffic_data.shape}")
        samples, d, t, n, l, f = self.traffic_data.shape
        if (d, t, n, l, f) != (self.config.n_days, self.config.time_steps, self.config.n_segments, self.config.n_lanes, self.config.n_features):
            raise ValueError(f"Data shape {self.traffic_data.shape} does not match config")
        if samples == 0:
            raise ValueError(f"The {self.split} split is empty")
        expected_labels = (samples, t, n, self.config.n_anomaly_types)
        if self.labels.shape != expected_labels:
            raise ValueError(f"labels must have shape {expected_labels}, got {self.labels.shape}")
        expected_adjacency = (samples, n, n)
        if self.adjacency_matrices.shape != expected_adjacency:
            raise ValueError(f"adjacency_matrices must have shape {expected_adjacency}, got {self.adjacency_matrices.shape}")

    def __len__(self):
        return int(self.traffic_data.shape[0])

    def __getitem__(self, idx):
        adj = np.array(self.adjacency_matrices[idx], dtype=np.float32, copy=True)
        adj_forward = adj.copy()
        np.fill_diagonal(adj_forward, 1.0)
        adj_backward = adj.T.copy()
        np.fill_diagonal(adj_backward, 1.0)
        return {
            "traffic": torch.from_numpy(np.array(self.traffic_data[idx], dtype=np.float32, copy=True)),
            "mask": torch.from_numpy(np.array(self.mask_data[idx], dtype=np.float32, copy=True)),
            "label": torch.from_numpy(np.array(self.labels[idx], dtype=np.float32, copy=True)),
            "adj_forward": torch.from_numpy(adj_forward),
            "adj_backward": torch.from_numpy(adj_backward),
        }


def get_dataloader(config, split="train", shuffle=None):
    if shuffle is None:
        shuffle = split == "train"
    dataset = TrafficAnomalyDataset(config, split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=bool(config.pin_memory) and config.device.type == "cuda",
    )
