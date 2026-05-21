from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DrivingDataset, decode_xy_from_ego, list_pickle_files
from src.metrics import ade_loss, displacement_errors, fde_loss
from src.model import EgoDrivePlanner
from src.utils import ensure_dir, get_device, rename_legacy_checkpoint_keys, save_json, set_seed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the hybrid planner with local trajectory targets and multiscale fusion."
    )
    parser.add_argument("--train-dir", default="~/data/DLAV/train")
    parser.add_argument("--val-dir", default="~/data/DLAV/val")
    parser.add_argument(
        "--output-dir",
        default="outputs/hybrid_resnet34_local_fusion",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=336)
    parser.add_argument("--image-feature-dim", type=int, default=256)
    parser.add_argument("--history-hidden-dim", type=int, default=128)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--fde-weight", type=float, default=0.15)
    parser.add_argument("--ade-weight", type=float, default=1.0)
    parser.add_argument("--time-weight-start", type=float, default=1.0)
    parser.add_argument("--time-weight-end", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--test-dir", default="test_public")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--submission-csv-name",
        default="submission_best_checkpoint.csv",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from. Use 'weights-only' to load only model weights.",
    )
    parser.add_argument(
        "--reset-trajectory-head-on-resume",
        action="store_true",
        help="Drop trajectory decoder/output head weights when loading a resume checkpoint.",
    )
    return parser


def freeze_backbone_parameters(model: EgoDrivePlanner) -> None:
    for parameter in model.stem.parameters():
        parameter.requires_grad = False


def freeze_backbone_batchnorm(model: EgoDrivePlanner) -> None:
    for module in model.stem.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)


def build_time_weights(
    future_steps: int,
    device: torch.device,
    *,
    start: float,
    end: float,
) -> torch.Tensor:
    weights = torch.linspace(start, end, future_steps, device=device)
    weights = weights.view(1, future_steps, 1)
    return weights / weights.mean()


def trajectory_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    time_weights: torch.Tensor,
    heading_weight: float,
    fde_weight: float,
    ade_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction_xy = prediction[..., :2].contiguous()
    target_xy = target[..., :2].contiguous()
    prediction_heading = prediction[..., 2:].contiguous()
    target_heading = target[..., 2:].contiguous()

    # Explicit ADE and FDE losses (L2 displacement)
    ade, ade_metrics = ade_loss(prediction, target)
    fde, fde_metrics = fde_loss(prediction, target)

    # Smooth-L1 xy loss with time weighting (regularizer)
    xy_residual = torch_f.smooth_l1_loss(
        prediction_xy,
        target_xy,
        reduction="none",
        beta=1.0,
    )
    xy_loss = (xy_residual * time_weights).mean()

    # Smooth-L1 heading loss (regularizer)
    heading_loss = torch_f.smooth_l1_loss(
        prediction_heading,
        target_heading,
        beta=1.0,
    )

    total_loss = (
        ade_weight * ade
        + xy_loss
        + heading_weight * heading_loss
        + fde_weight * fde
    )
    metrics = {
        "ade_loss": ade_metrics["ade_loss"],
        "fde_loss": fde_metrics["fde_loss"],
        "xy_loss": float(xy_loss.detach().item()),
        "heading_loss": float(heading_loss.detach().item()),
        "loss": float(total_loss.detach().item()),
    }
    return total_loss, metrics


def evaluate(
    model: EgoDrivePlanner,
    dataloader: DataLoader,
    *,
    device: torch.device,
    time_weights: torch.Tensor,
    heading_weight: float,
    fde_weight: float,
    ade_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ade_loss = 0.0
    total_xy_loss = 0.0
    total_heading_loss = 0.0
    total_fde_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            outputs = model(camera, history)
            loss, metrics = trajectory_objective(
                outputs["trajectory"],
                future,
                time_weights=time_weights,
                heading_weight=heading_weight,
                fde_weight=fde_weight,
                ade_weight=ade_weight,
            )

            ade, fde = displacement_errors(outputs["trajectory"], future)

            batch_size = camera.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_ade_loss += metrics["ade_loss"] * batch_size
            total_xy_loss += metrics["xy_loss"] * batch_size
            total_heading_loss += metrics["heading_loss"] * batch_size
            total_fde_loss += metrics["fde_loss"] * batch_size
            total_ade += ade.item() * batch_size
            total_fde += fde.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "ade_loss": total_ade_loss / total_samples,
        "xy_loss": total_xy_loss / total_samples,
        "heading_loss": total_heading_loss / total_samples,
        "fde_loss": total_fde_loss / total_samples,
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
    }


def generate_submission(
    model: EgoDrivePlanner,
    *,
    test_dir: str | Path,
    output_csv: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_height: int,
    image_width: int,
    max_test_samples: int | None,
) -> None:
    test_files = list_pickle_files(test_dir, max_test_samples)
    if not test_files:
        raise RuntimeError(f"No test samples found in {Path(test_dir).expanduser()}")

    dataset = DrivingDataset(
        test_files,
        test=True,
        image_height=image_height,
        image_width=image_width,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    predictions = []
    sample_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test inference", leave=False):
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            last_pos = batch["last_pos"].to(device)
            last_heading = batch["last_heading"].to(device)

            outputs = model(camera, history)
            prediction_xy = decode_xy_from_ego(
                outputs["trajectory"][..., :2],
                origin_xy=last_pos,
                origin_heading=last_heading,
            )
            predictions.append(prediction_xy.cpu())
            sample_ids.extend(batch["sample_id"].tolist())

    all_predictions = torch.cat(predictions, dim=0).numpy()
    flat_predictions = all_predictions.reshape(all_predictions.shape[0], -1)

    column_names = ["id"]
    for step in range(1, all_predictions.shape[1] + 1):
        column_names.extend([f"x_{step}", f"y_{step}"])

    submission = pd.DataFrame(flat_predictions)
    submission.insert(0, "id", sample_ids)
    submission.columns = column_names
    submission.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")
    print(f"Submission rows: {len(submission)}, columns: {len(submission.columns)}")


def main() -> None:
    args = build_argparser().parse_args()
    if args.image_size is not None:
        args.image_height = args.image_size
        args.image_width = args.image_size

    set_seed(args.seed)
    device = get_device()
    time_weights = build_time_weights(
        args.future_steps,
        device,
        start=args.time_weight_start,
        end=args.time_weight_end,
    )

    output_dir = ensure_dir(args.output_dir)

    train_files = list_pickle_files(args.train_dir, args.max_train_samples)
    val_files = list_pickle_files(args.val_dir, args.max_val_samples)

    train_dataset = DrivingDataset(
        train_files,
        augment=True,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    val_dataset = DrivingDataset(
        val_files,
        augment=False,
        image_height=args.image_height,
        image_width=args.image_width,
    )

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
        pretrained_backbone=not args.no_pretrained,
        image_feature_dim=args.image_feature_dim,
        history_hidden_dim=args.history_hidden_dim,
        history_layers=args.history_layers,
        fusion_dim=args.fusion_dim,
        fusion_heads=args.fusion_heads,
        dropout=args.dropout,
        future_steps=args.future_steps,
    ).to(device)

    if args.freeze_backbone:
        freeze_backbone_parameters(model)

    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("stem."):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": args.lr})

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    best_state = None
    best_val_ade = float("inf")
    start_epoch = 1
    history: list[dict[str, float | int]] = []

    if args.resume:
        resume_path = args.resume
        resume_mode = "full"
        if resume_path.endswith(":weights-only"):
            resume_path = resume_path[: -len(":weights-only")]
            resume_mode = "weights-only"
        if args.reset_trajectory_head_on_resume and resume_mode == "full":
            print("Resetting trajectory decoder requires weights-only resume; optimizer and scheduler will be reset.")
            resume_mode = "weights-only"

        print(f"Resuming from {resume_path} (mode={resume_mode})")
        checkpoint = torch.load(resume_path, map_location=device)
        loaded_state = checkpoint["model_state_dict"]
        model_state = model.state_dict()

        loaded_state = rename_legacy_checkpoint_keys(loaded_state)

        reset_keys = []
        if args.reset_trajectory_head_on_resume:
            drop_prefixes = (
                "trajectory_decoder_cell.",
                "trajectory_output_head.",
            )
            filtered_for_reset = {}
            for key, value in loaded_state.items():
                if any(key.startswith(prefix) for prefix in drop_prefixes):
                    reset_keys.append(key)
                    continue
                filtered_for_reset[key] = value
            loaded_state = filtered_for_reset

        # Filter out keys with shape mismatches and keys that no longer exist in the model
        filtered_state = {}
        skipped_keys = []
        for key, value in loaded_state.items():
            if key not in model_state:
                skipped_keys.append(f"{key}: removed from model")
            elif value.shape != model_state[key].shape:
                skipped_keys.append(f"{key}: {value.shape} -> {model_state[key].shape}")
            else:
                filtered_state[key] = value

        load_result = model.load_state_dict(filtered_state, strict=False)
        if skipped_keys:
            print(f"  Skipped keys (re-initialized or removed):")
            for sk in skipped_keys:
                print(f"    {sk}")
        if reset_keys:
            print(f"  Reset trajectory decoder/head weights: {reset_keys}")
        if load_result.missing_keys:
            print(f"  Missing keys (randomly initialized): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  Unexpected keys (ignored): {load_result.unexpected_keys}")

        if resume_mode == "full":
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_ade = checkpoint.get("best_val_ade", float("inf"))
            # Re-wind scheduler to the correct step
            for _ in range(checkpoint["epoch"]):
                scheduler.step()
            print(
                f"  Restored epoch {checkpoint['epoch']}, "
                f"best_val_ade={best_val_ade:.4f}, continuing from epoch {start_epoch}"
            )
        else:
            print("  Weights-only mode: optimizer and scheduler reset from scratch")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            freeze_backbone_batchnorm(model)
        running_loss = 0.0
        running_ade_loss = 0.0
        running_xy_loss = 0.0
        running_heading_loss = 0.0
        running_fde_loss = 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            camera = batch["camera"].to(device)
            history_batch = batch["history"].to(device)
            future = batch["future"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(camera, history_batch)
            loss, loss_metrics = trajectory_objective(
                outputs["trajectory"],
                future,
                time_weights=time_weights,
                heading_weight=args.heading_weight,
                fde_weight=args.fde_weight,
                ade_weight=args.ade_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = camera.size(0)
            total_samples += batch_size
            running_loss += loss_metrics["loss"] * batch_size
            running_ade_loss += loss_metrics["ade_loss"] * batch_size
            running_xy_loss += loss_metrics["xy_loss"] * batch_size
            running_heading_loss += loss_metrics["heading_loss"] * batch_size
            running_fde_loss += loss_metrics["fde_loss"] * batch_size
            progress.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                ade=f"{running_ade_loss / total_samples:.4f}",
                xy=f"{running_xy_loss / total_samples:.4f}",
                fde=f"{running_fde_loss / total_samples:.4f}",
            )

        scheduler.step()
        train_stats = {
            "loss": running_loss / total_samples,
            "ade_loss": running_ade_loss / total_samples,
            "xy_loss": running_xy_loss / total_samples,
            "heading_loss": running_heading_loss / total_samples,
            "fde_loss": running_fde_loss / total_samples,
        }
        val_stats = evaluate(
            model,
            val_loader,
            device=device,
            time_weights=time_weights,
            heading_weight=args.heading_weight,
            fde_weight=args.fde_weight,
            ade_weight=args.ade_weight,
        )
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ade_loss": train_stats["ade_loss"],
            "train_xy_loss": train_stats["xy_loss"],
            "train_heading_loss": train_stats["heading_loss"],
            "train_fde_loss": train_stats["fde_loss"],
            "val_loss": val_stats["loss"],
            "val_ade_loss": val_stats["ade_loss"],
            "val_xy_loss": val_stats["xy_loss"],
            "val_heading_loss": val_stats["heading_loss"],
            "val_fde_loss": val_stats["fde_loss"],
            "val_ade": val_stats["ade"],
            "val_fde": val_stats["fde"],
            "backbone_lr": scheduler.get_last_lr()[0],
            "head_lr": scheduler.get_last_lr()[-1],
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_ade={train_stats['ade_loss']:.4f} | "
            f"train_xy={train_stats['xy_loss']:.4f} | "
            f"train_fde={train_stats['fde_loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_ade_loss={val_stats['ade_loss']:.4f} | "
            f"val_xy={val_stats['xy_loss']:.4f} | "
            f"val_fde_loss={val_stats['fde_loss']:.4f} | "
            f"val_ADE={val_stats['ade']:.4f} | "
            f"val_FDE={val_stats['fde']:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_ade": best_val_ade,
            "config": vars(args)
            | {
                "coordinate_frame": "ego",
            },
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
            "ade_weight": args.ade_weight,
            "fde_weight": args.fde_weight,
            "submission_csv": str(output_dir / args.submission_csv_name),
        },
    )

    model.load_state_dict(best_state["model_state_dict"])
    submission_path = output_dir / args.submission_csv_name
    generate_submission(
        model,
        test_dir=args.test_dir,
        output_csv=submission_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_height=args.image_height,
        image_width=args.image_width,
        max_test_samples=args.max_test_samples,
    )

    print(f"Best checkpoint saved to {output_dir / 'best_checkpoint.pt'}")
    print(f"Best validation ADE: {best_val_ade:.4f}")


if __name__ == "__main__":
    main()