import sys
sys.dont_write_bytecode = True

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .config import load_config_from_args
from .dataset import get_dataloader
from .runtime_utils import EarlyStopping, compute_metrics, load_checkpoint, save_checkpoint
from .model import OurAnomalyDetection


MODEL_REGISTRY = {
    "our": OurAnomalyDetection,
    "u-lastde": OurAnomalyDetection,
    "u_lastde": OurAnomalyDetection,
}


def json_default(obj):
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return str(obj)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def build_model(config) -> nn.Module:
    name = config.model_name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{config.model_name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class Trainer:
    def __init__(self, config, test_only=False):
        set_seed(config.random_seed)
        self.config = config
        self.device = config.device
        os.makedirs(config.save_dir, exist_ok=True)

        print("Loading datasets...")
        self.train_loader = None
        self.val_loader = None
        if not test_only:
            self.train_loader = get_dataloader(config, "train", shuffle=True)
            self.val_loader = get_dataloader(config, "val", shuffle=False)
        self.test_loader = get_dataloader(config, "test", shuffle=False)

        print("Building model...")
        self.model = build_model(config)
        if config.device.type == "cuda" and config.use_dataparallel and len(config.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=config.gpu_ids)
        self.model = self.model.to(self.device)
        print(f"Model: {config.model_name}; parameters={sum(p.numel() for p in self.model.parameters()):,}")

        self.criterion = nn.MSELoss() if config.loss_type == "mse" else FocalLoss(config.focal_alpha, config.focal_gamma)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)
        self.early_stopping = EarlyStopping(config.early_stopping_patience)
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.train_losses = []
        self.val_losses = []
        self._save_config_snapshot()

    def _save_config_snapshot(self):
        path = os.path.join(self.config.save_dir, "config_snapshot.json")
        payload = {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)

    def _save_loss_history(self):
        np.save(os.path.join(self.config.save_dir, "train_losses.npy"), np.asarray(self.train_losses, dtype=np.float64))
        np.save(os.path.join(self.config.save_dir, "val_losses.npy"), np.asarray(self.val_losses, dtype=np.float64))

    def _forward_loss(self, batch):
        traffic = batch["traffic"].to(self.device)
        mask = batch["mask"].to(self.device)
        label = batch["label"].to(self.device)
        adj_forward = batch["adj_forward"].to(self.device)
        adj_backward = batch["adj_backward"].to(self.device)
        output = self.model(traffic, mask, adj_forward, adj_backward)
        if self.config.loss_type == "mse":
            prob = torch.sigmoid(output)
            loss = self.criterion(prob, label)
        else:
            loss = self.criterion(output, label)
            prob = torch.sigmoid(output)
        return loss, prob, label

    def train_epoch(self, epoch):
        self.model.train()
        total = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            loss, _, _ = self._forward_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            if batch_idx % self.config.print_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.8f}")
        return total / max(len(self.train_loader), 1)

    def evaluate(self, loader, desc="Evaluating", detail_level="summary"):
        self.model.eval()
        total = 0.0
        outputs, labels, adjs = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                loss, prob, label = self._forward_loss(batch)
                total += loss.item()
                outputs.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())
                adjs.append(batch["adj_forward"].cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        adjs = np.concatenate(adjs, axis=0)
        return total / max(len(loader), 1), compute_metrics(
            outputs,
            labels,
            adjs,
            threshold=getattr(self.config, "threshold", 0.5),
            threshold_mode=getattr(self.config, "threshold_mode", "auto"),
            detail_level=detail_level,
        )

    def train(self):
        print(self.config)
        for epoch in range(self.config.epochs):
            start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.evaluate(self.val_loader, "Validating")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self._save_loss_history()
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch}: train={train_loss:.8f}, val={val_loss:.8f}, time={time.time()-start:.2f}s")
            print(f"Val metrics: {val_metrics}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, os.path.join(self.config.save_dir, "best.pth"))
                print("Best model saved")
            if (epoch + 1) % self.config.save_interval == 0:
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, os.path.join(self.config.save_dir, f"epoch_{epoch}.pth"))
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        print(f"Best epoch={self.best_epoch}, best val={self.best_val_loss:.8f}")
        self.test()

    def test(self, checkpoint_path=None):
        best_path = os.path.join(self.config.save_dir, "best.pth")
        checkpoint_path = checkpoint_path or best_path
        print(f"Loading checkpoint: {checkpoint_path}")
        load_checkpoint(checkpoint_path, self.model)
        test_loss, test_metrics = self.evaluate(self.test_loader, "Testing", detail_level="detailed")
        result = {
            "checkpoint": checkpoint_path,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        }
        json_path = os.path.join(self.config.save_dir, "test_metrics.json")
        txt_path = os.path.join(self.config.save_dir, "test_summary.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=json_default)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"checkpoint: {checkpoint_path}\n")
            f.write(f"test_loss: {test_loss:.8f}\n")
            f.write(f"test_metrics: {test_metrics}\n")
        print("=" * 60)
        print(f"Test loss: {test_loss:.8f}")
        print(f"Test metrics: {test_metrics}")
        print(f"Saved test metrics: {json_path}")
        print("=" * 60)


def dry_run(config):
    set_seed(config.random_seed)
    model = build_model(config).to(config.device).eval()
    b = 1
    traffic = torch.randn(b, config.n_days, config.time_steps, config.n_segments, config.n_lanes, config.n_features, device=config.device)
    mask = torch.ones(b, config.n_days, config.time_steps, config.n_segments, config.n_lanes, device=config.device)
    adj = torch.eye(config.n_segments, device=config.device).unsqueeze(0).repeat(b, 1, 1)
    with torch.no_grad():
        out = model(traffic, mask, adj, adj)
    expected = (b, config.time_steps, config.n_segments, config.n_anomaly_types)
    print(f"dry-run output shape: {tuple(out.shape)} expected: {expected}")
    if tuple(out.shape) != expected:
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")


def main():
    config, args = load_config_from_args()
    if args.dry_run:
        dry_run(config)
    elif args.test_only:
        if not args.checkpoint:
            raise ValueError("--test-only requires --checkpoint")
        Trainer(config, test_only=True).test(args.checkpoint)
    else:
        Trainer(config).train()


if __name__ == "__main__":
    main()

