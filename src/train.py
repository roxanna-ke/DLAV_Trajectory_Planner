from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DrivingDataset, list_pickle_files
from src.metrics import displacement_errors, trajectory_loss
from src.model import EgoDrivePlanner
from src.utils import ensure_dir, get_device, save_json, set_seed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Milestone 1 planner.")
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--val-dir", default="val")
    parser.add_argument("--output-dir", default="outputs/phase1_resnet_gru")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-feature-dim", type=int, default=128)
    parser.add_argument("--history-feature-dim", type=int, default=64)
    parser.add_argument("--command-feature-dim", type=int, default=16)
    parser.add_argument("--gru-hidden-dim", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser


def freeze_backbone_parameters(model: EgoDrivePlanner) -> None:
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad = False


def evaluate(
    model: EgoDrivePlanner,
    dataloader: DataLoader,
    *,
    device: torch.device,
    heading_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)
            future = batch["future"].to(device)

            prediction = model(camera, history, command)
            loss, _ = trajectory_loss(
                prediction, future, heading_weight=heading_weight
            )
            ade, fde = displacement_errors(prediction, future)

            batch_size = camera.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_ade += ade.item() * batch_size
            total_fde += fde.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
    }


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = get_device()

    output_dir = ensure_dir(args.output_dir)

    train_files = list_pickle_files(args.train_dir, args.max_train_samples)
    val_files = list_pickle_files(args.val_dir, args.max_val_samples)

    train_dataset = DrivingDataset(train_files, image_size=args.image_size)
    val_dataset = DrivingDataset(val_files, image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = EgoDrivePlanner(
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained,
        image_feature_dim=args.image_feature_dim,
        history_feature_dim=args.history_feature_dim,
        command_feature_dim=args.command_feature_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        future_steps=args.future_steps,
    ).to(device)

    if args.freeze_backbone:
        freeze_backbone_parameters(model)

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_state = None
    best_val_ade = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_xy_loss = 0.0
        running_heading_loss = 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            camera = batch["camera"].to(device)
            history_batch = batch["history"].to(device)
            command = batch["command"].to(device)
            future = batch["future"].to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(camera, history_batch, command)
            loss, loss_metrics = trajectory_loss(
                prediction,
                future,
                heading_weight=args.heading_weight,
            )
            loss.backward()
            optimizer.step()

            batch_size = camera.size(0)
            total_samples += batch_size
            running_loss += loss_metrics["loss"] * batch_size
            running_xy_loss += loss_metrics["xy_loss"] * batch_size
            running_heading_loss += loss_metrics["heading_loss"] * batch_size
            progress.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                xy=f"{running_xy_loss / total_samples:.4f}",
                heading=f"{running_heading_loss / total_samples:.4f}",
            )

        scheduler.step()
        train_stats = {
            "loss": running_loss / total_samples,
            "xy_loss": running_xy_loss / total_samples,
            "heading_loss": running_heading_loss / total_samples,
        }
        val_stats = evaluate(
            model,
            val_loader,
            device=device,
            heading_weight=args.heading_weight,
        )
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_xy_loss": train_stats["xy_loss"],
            "train_heading_loss": train_stats["heading_loss"],
            "val_loss": val_stats["loss"],
            "val_ade": val_stats["ade"],
            "val_fde": val_stats["fde"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_ADE={val_stats['ade']:.4f} | "
            f"val_FDE={val_stats['fde']:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_ade": best_val_ade,
            "config": vars(args),
        }
        torch.save(checkpoint, output_dir / "last_checkpoint.pt")

        if val_stats["ade"] < best_val_ade:
            best_val_ade = val_stats["ade"]
            checkpoint["best_val_ade"] = best_val_ade
            best_state = copy.deepcopy(checkpoint)
            torch.save(checkpoint, output_dir / "best_checkpoint.pt")

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    save_json(output_dir / "train_history.json", {"history": history})
    save_json(
        output_dir / "summary.json",
        {
            "device": str(device),
            "best_val_ade": best_val_ade,
            "epochs": args.epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "best_epoch": best_state["epoch"],
        },
    )
    print(f"Best checkpoint saved to {output_dir / 'best_checkpoint.pt'}")
    print(f"Best validation ADE: {best_val_ade:.4f}")


if __name__ == "__main__":
    main()
