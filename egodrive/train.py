from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .config import get_section, load_config
from .data import DrivingDataset, compute_real_style_stats, list_pickle_files
from .evaluation import predict_to_csv
from .factory import build_model
from .metrics import auxiliary_losses, build_time_weights, displacement_errors, trajectory_objective
from .utils import ensure_dir, get_device, move_batch_to_device, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EgoDrive trajectory planner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    return parser


def _dataset(config: dict[str, Any], files: list[Path], *, test: bool, augment: bool, style_stats=None):
    model_cfg = get_section(config, "model")
    data_cfg = get_section(config, "data")
    return DrivingDataset(
        files,
        test=test,
        augment=augment,
        require_command=bool(model_cfg.get("use_command", False)),
        include_auxiliary=bool(model_cfg.get("use_depth_head", False) or model_cfg.get("use_segmentation_head", False)),
        image_height=int(data_cfg.get("image_height", 224)),
        image_width=int(data_cfg.get("image_width", 336)),
        style_stats=style_stats,
        style_alpha=float(data_cfg.get("style_alpha", 0.5)),
    )


def _run_epoch(model, loader, optimizer, device, config, time_weights, train: bool) -> dict[str, float]:
    model.train(train)
    training_cfg = get_section(config, "training")
    totals: dict[str, float] = {}
    sample_count = 0
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        kwargs = {"command": batch["command"]} if "command" in batch else {}
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(batch["camera"], batch["history"], **kwargs)
            loss, metrics = trajectory_objective(
                outputs["trajectory"], batch["future"], time_weights=time_weights,
                heading_weight=float(training_cfg.get("heading_weight", 0.1)),
                ade_weight=float(training_cfg.get("ade_weight", 1.0)),
                fde_weight=float(training_cfg.get("fde_weight", 0.15)),
            )
            aux_loss, aux_metrics = auxiliary_losses(
                outputs, batch,
                depth_weight=float(training_cfg.get("depth_loss_weight", 0.1)),
                segmentation_weight=float(training_cfg.get("segmentation_loss_weight", 0.2)),
            )
            loss = loss + aux_loss
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("gradient_clip", 5.0)))
                optimizer.step()
        batch_size = batch["camera"].size(0)
        sample_count += batch_size
        for key, value in {**metrics, **aux_metrics, "loss": float(loss.detach())}.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    if not sample_count:
        raise RuntimeError("The data loader yielded no samples")
    return {key: value / sample_count for key, value in totals.items()}


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_cfg = get_section(config, "data")
    model_cfg = get_section(config, "model")
    training_cfg = get_section(config, "training")
    output_dir = ensure_dir(data_cfg.get("output_dir", "outputs/run"))
    set_seed(int(training_cfg.get("seed", 42)))
    device = get_device(args.device)

    train_files = list_pickle_files(data_cfg["train_dir"], data_cfg.get("max_train_samples"))
    real_dir = data_cfg.get("real_dir")
    style_stats = None
    if real_dir:
        real_files = list_pickle_files(real_dir, data_cfg.get("max_real_samples"))
        fraction = float(data_cfg.get("real_val_fraction", 0.5))
        split = max(1, min(len(real_files) - 1, int(len(real_files) * (1.0 - fraction))))
        real_train, val_files = real_files[:split], real_files[split:]
        if data_cfg.get("use_adain", False):
            style_stats = compute_real_style_stats(real_files, image_height=int(data_cfg.get("image_height", 224)), image_width=int(data_cfg.get("image_width", 336)), max_samples=int(data_cfg.get("style_stat_samples", 200)))
        real_train = real_train * int(data_cfg.get("real_oversample", 1))
        train_dataset = ConcatDataset([_dataset(config, train_files, test=False, augment=True, style_stats=style_stats), _dataset(config, real_train, test=False, augment=True)])
    else:
        val_files = list_pickle_files(data_cfg["val_dir"], data_cfg.get("max_val_samples"))
        train_dataset = _dataset(config, train_files, test=False, augment=True)
    if not train_files or not val_files:
        raise RuntimeError("Training and validation data must both be non-empty")
    val_dataset = _dataset(config, val_files, test=False, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=int(training_cfg.get("batch_size", 32)), shuffle=True, num_workers=int(training_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=int(training_cfg.get("batch_size", 32)), shuffle=False, num_workers=int(training_cfg.get("num_workers", 0)))

    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("lr", 1e-3)), weight_decay=float(training_cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training_cfg.get("epochs", 1)), eta_min=1e-6)
    time_weights = build_time_weights(int(model_cfg.get("future_steps", 60)), float(training_cfg.get("time_weight_start", 1.0)), float(training_cfg.get("time_weight_end", 2.0))).to(device)

    best_state = None
    best_ade = float("inf")
    history = []
    for epoch in range(1, int(training_cfg.get("epochs", 1)) + 1):
        train_stats = _run_epoch(model, train_loader, optimizer, device, config, time_weights, train=True)
        with torch.no_grad():
            val_stats = _run_epoch(model, val_loader, None, device, config, time_weights, train=False)
        scheduler.step()
        summary = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(summary)
        print(f"Epoch {epoch:03d} | train_loss={train_stats['loss']:.4f} | val_loss={val_stats['loss']:.4f} | val_ADE={val_stats['ade']:.4f} | val_FDE={val_stats['fde']:.4f}")
        checkpoint = {"model_state_dict": model.state_dict(), "model_config": model_cfg, "config": config, "epoch": epoch, "best_val_ade": best_ade}
        torch.save(checkpoint, output_dir / "last_checkpoint.pt")
        if val_stats["ade"] < best_ade:
            best_ade = val_stats["ade"]
            checkpoint["best_val_ade"] = best_ade
            best_state = copy.deepcopy(checkpoint)
            torch.save(best_state, output_dir / "best_checkpoint.pt")
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    save_json(output_dir / "train_history.json", {"history": history})
    test_dir = data_cfg.get("test_dir")
    if test_dir:
        model.load_state_dict(best_state["model_state_dict"])
        output_csv = output_dir / data_cfg.get("submission_name", "submission.csv")
        files = list_pickle_files(test_dir, data_cfg.get("max_test_samples"))
        predict_to_csv(model, files, output_csv, device=device, image_height=int(data_cfg.get("image_height", 224)), image_width=int(data_cfg.get("image_width", 336)), batch_size=int(training_cfg.get("batch_size", 32)), num_workers=int(training_cfg.get("num_workers", 0)), require_command=bool(model_cfg.get("use_command", False)))
        print(f"Saved submission to {output_csv}")


if __name__ == "__main__":
    main()
