"""Training loop of the segmentation baseline.

Plain PyTorch, no framework: AdamW, a one-cycle learning-rate schedule, mixed precision on the
GPU, validation after every epoch, a checkpoint of the best and of the last epoch, and early
stopping. The history goes to ``history.csv`` / ``history.json`` next to the checkpoints.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from robosoccer_perception import config
from robosoccer_perception.datasets.io import SoccerDataset
from robosoccer_perception.inference import profiling
from robosoccer_perception.segmentation import data as seg_data
from robosoccer_perception.segmentation.losses import SegmentationLoss
from robosoccer_perception.segmentation.metrics import SegmentationMeter
from robosoccer_perception.segmentation.model import build_model, count_parameters
from robosoccer_perception.segmentation.segmenter import save_checkpoint

MONITORS = {"line_iou": lambda m: m["line"]["iou"], "mean_iou": lambda m: m["mean_iou"],
            "line_f1": lambda m: m["line"]["f1"]}


def load_segmentation_config(name_or_path: str = "unet_small",
                             overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config.deep_update(config.load_config("segmentation", name_or_path), overrides)
    for key in ("name", "model", "epochs", "batch", "learning_rate"):
        if key not in cfg:
            raise ValueError(f"segmentation configuration is missing '{key}'")
    if cfg.get("monitor", "line_iou") not in MONITORS:
        raise ValueError(f"monitor must be one of {sorted(MONITORS)}")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: str, loss_fn=None,
             amp: bool = False) -> dict[str, Any]:
    """Metrics of ``model`` over a loader of un-augmented, padded images."""
    model.eval()
    meter = SegmentationMeter()
    losses = []
    for batch in loader:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        with torch.autocast(device.split(":")[0], enabled=amp):
            logits = model(images)
        if loss_fn is not None:
            losses.append(float(loss_fn(logits, masks)))
        predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for prediction, mask in zip(predictions, batch["mask"].numpy()):
            meter.update(prediction, mask.astype(np.uint8))
    result = meter.compute()
    result["loss"] = float(np.mean(losses)) if losses else None
    return result


def train_segmentation(cfg: dict[str, Any], dataset_root=None, output_dir=None,
                       progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Train the model described by ``cfg``; returns a summary with checkpoint paths."""
    dataset_root = dataset_root or cfg["dataset"]
    run_dir = Path(output_dir or cfg.get("output", "outputs/segmentation")) / str(cfg["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = profiling.resolve_device(cfg.get("device", "auto"))
    amp = bool(cfg.get("amp", True)) and device.startswith("cuda")
    if bool(cfg.get("deterministic", False)):
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    model = build_model(cfg["model"]).to(device)
    stride = getattr(model, "stride", 16)
    normalization = cfg.get("normalization") or seg_data.DEFAULT_NORMALIZATION
    augment = cfg.get("augment") or {}
    train_set = seg_data.SegmentationDataset(
        dataset_root, "train", augment=True, crop_size=tuple(augment.get("crop_size", (512, 352))),
        scale_range=tuple(augment.get("scale_range", (0.8, 1.25))),
        flip_prob=float(augment.get("flip_prob", 0.5)), stride=stride,
        normalization=normalization, seed=seed, limit=cfg.get("limit_train"))
    val_set = seg_data.SegmentationDataset(dataset_root, "val", stride=stride,
                                           normalization=normalization, limit=cfg.get("limit_val"))
    workers = int(cfg.get("workers", 2))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=int(cfg["batch"]), shuffle=True,
                              num_workers=workers, drop_last=len(train_set) > int(cfg["batch"]),
                              pin_memory=device.startswith("cuda"), generator=generator,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=int(cfg.get("val_batch", cfg["batch"])),
                            shuffle=False, num_workers=workers)

    dataset = SoccerDataset(dataset_root)
    class_weights = cfg.get("class_weights", "auto")
    if class_weights == "auto":
        class_weights = seg_data.class_weights_from_counts(
            seg_data.class_pixel_counts(dataset, "train"),
            power=float(cfg.get("class_weight_power", 0.5)))
    loss_fn = SegmentationLoss(class_weights, float(cfg.get("dice_weight", 0.5))).to(device)
    epochs = int(cfg["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(cfg["learning_rate"]), total_steps=epochs * len(train_loader),
        pct_start=float(cfg.get("warmup_fraction", 0.1)))
    scaler = torch.amp.GradScaler(enabled=amp)
    monitor_name = cfg.get("monitor", "line_iou")
    monitor = MONITORS[monitor_name]
    patience = int(cfg.get("patience", 10))
    image_width = int(dataset.header["config"]["resolution"][0])
    extra = {"config_name": str(cfg["name"]), "dataset_name": dataset.header["name"],
             "dataset_manifest_sha256": dataset.header["manifest_sha256"],
             "class_weights": [float(w) for w in class_weights] if class_weights else None}
    if progress:
        progress(f"{cfg['name']}: {count_parameters(model):,} parameters on {device}, "
                 f"{len(train_set)} train / {len(val_set)} val images, class weights "
                 f"{[round(w, 2) for w in class_weights] if class_weights else None}")

    history: list[dict[str, Any]] = []
    best_score, best_epoch, started = -1.0, -1, time.time()
    profiling.reset_gpu_peak(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_set.set_epoch(epoch)
        epoch_started, losses = time.time(), []
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.split(":")[0], enabled=amp):
                logits = model(images)
            loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        metrics = validate(model, val_loader, device, loss_fn, amp)
        score = monitor(metrics) or 0.0
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": metrics["loss"],
               "val_mean_iou": metrics["mean_iou"], "val_line_iou": metrics["line"]["iou"],
               "val_line_precision": metrics["line"]["precision"],
               "val_line_recall": metrics["line"]["recall"], "val_line_f1": metrics["line"]["f1"],
               "val_line_f1_tolerant": metrics["line_tolerant"]["f1"],
               "val_field_iou": metrics["per_class"]["field"]["iou"],
               "learning_rate": float(scheduler.get_last_lr()[0]),
               "seconds": round(time.time() - epoch_started, 1)}
        history.append(row)
        checkpoint_extra = {**extra, "epoch": epoch, "monitor": monitor_name,
                            "val_metrics": {k: row[k] for k in row if k.startswith("val_")}}
        save_checkpoint(run_dir / "last.pt", model, cfg["model"], normalization, image_width,
                        checkpoint_extra)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            save_checkpoint(run_dir / "best.pt", model, cfg["model"], normalization, image_width,
                            checkpoint_extra)
        if progress:
            progress(f"epoch {epoch:3d}/{epochs}  train loss {row['train_loss']:.4f}  "
                     f"val loss {row['val_loss']:.4f}  mIoU {row['val_mean_iou']:.3f}  "
                     f"line IoU {row['val_line_iou']:.3f}  line F1 {row['val_line_f1']:.3f}  "
                     f"{row['seconds']:.0f} s{'  *' if improved else ''}")
        _write_history(run_dir, history)
        if epoch - best_epoch >= patience:
            if progress:
                progress(f"early stop: no improvement of {monitor_name} for {patience} epochs")
            break

    summary = {"run_dir": str(run_dir), "best": str(run_dir / "best.pt"),
               "last": str(run_dir / "last.pt"), "best_epoch": best_epoch,
               f"best_val_{monitor_name}": best_score, "epochs_run": len(history),
               "parameters": count_parameters(model), "device": device, "amp": amp,
               "training_seconds": round(time.time() - started, 1),
               "gpu_memory": profiling.gpu_memory_mb(device),
               "hardware": profiling.hardware_summary(device), "config": cfg,
               "dataset": {"root": str(dataset_root), "name": dataset.header["name"],
                           "manifest_sha256": dataset.header["manifest_sha256"]}}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        from robosoccer_perception.visualization import vision_plots
        vision_plots.plot_training_history(
            history, run_dir / "history.png",
            {"Loss": ["train_loss", "val_loss"],
             "Validation IoU": ["val_line_iou", "val_field_iou", "val_mean_iou"],
             "Validation line F1": ["val_line_f1", "val_line_f1_tolerant"]},
            f"Segmentation training: {cfg['name']}")
    except ImportError:
        pass
    return summary


def _write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
