from __future__ import annotations

import argparse
import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DrivingDataset, list_pickle_files
from src.metrics import (
    compute_total_loss,
    displacement_errors,
)
from src.model import EgoDrivePlanner
from src.utils import (
    ensure_dir,
    get_device,
    rename_legacy_checkpoint_keys,
    save_json,
    set_seed,
)


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
    parser.add_argument("--command-feature-dim", type=int, default=8)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--num-segmentation-classes", type=int, default=15)
    parser.add_argument("--use-segmentation-head", action="store_true")
    parser.add_argument("--use-depth-head", action="store_true")
    parser.add_argument("--depth-loss-weight", type=float, default=0.1)
    parser.add_argument("--use-layer3-spatial-pooling", action="store_true")
    parser.add_argument("--layer3-spatial-scale", type=float, default=0.1)
    parser.add_argument("--layer3-spatial-grid-height", type=int, default=4)
    parser.add_argument("--layer3-spatial-grid-width", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--fde-weight", type=float, default=0.15)
    parser.add_argument("--ade-weight", type=float, default=1.0)
    parser.add_argument("--segmentation-loss-weight", type=float, default=0.2)
    parser.add_argument("--aux-warmup-epochs", type=int, default=5)
    parser.add_argument("--aux-ramp-epochs", type=int, default=10)
    parser.add_argument("--time-weight-start", type=float, default=1.0)
    parser.add_argument("--time-weight-end", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
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
    for parameter in model.stem_layer3.parameters():
        parameter.requires_grad = False
    for parameter in model.stem_layer4.parameters():
        parameter.requires_grad = False


def freeze_backbone_batchnorm(model: EgoDrivePlanner) -> None:
    for module in model.stem_layer3.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)
    for module in model.stem_layer4.modules():
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


def compute_aux_scale(
    epoch: int,
    *,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = (epoch - warmup_epochs) / ramp_epochs
    return float(max(0.0, min(1.0, progress)))


def evaluate(
    model: EgoDrivePlanner,
    dataloader: DataLoader,
    *,
    device: torch.device,
    time_weights: torch.Tensor,
    heading_weight: float,
    fde_weight: float,
    ade_weight: float,
    segmentation_loss_weight: float,
    depth_loss_weight: float,
    use_segmentation_head: bool,
    use_depth_head: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ade_loss = 0.0
    total_xy_loss = 0.0
    total_heading_loss = 0.0
    total_fde_loss = 0.0
    total_segmentation_loss = 0.0
    total_weighted_segmentation_loss = 0.0
    total_depth_loss = 0.0
    total_weighted_depth_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)

            outputs = model(camera, history, command)

            targets = {
                "future": batch["future"].to(device),
                "semantic_label": batch["semantic_label"].to(device),
            }
            if use_depth_head and "depth" in batch:
                targets["depth"] = batch["depth"].to(device)

            loss, metrics = compute_total_loss(
                outputs,
                targets,
                time_weights=time_weights,
                heading_weight=heading_weight,
                fde_weight=fde_weight,
                ade_weight=ade_weight,
                segmentation_loss_weight=segmentation_loss_weight,
                depth_loss_weight=depth_loss_weight,
                use_segmentation_head=use_segmentation_head,
                use_depth_head=use_depth_head,
            )

            ade, fde = displacement_errors(outputs["trajectory"], targets["future"])

            batch_size = camera.size(0)
            total_samples += batch_size
            total_loss += metrics["loss"] * batch_size
            total_ade_loss += metrics["ade_loss"] * batch_size
            total_xy_loss += metrics["xy_loss"] * batch_size
            total_heading_loss += metrics["heading_loss"] * batch_size
            total_fde_loss += metrics["fde_loss"] * batch_size
            total_segmentation_loss += metrics["segmentation_loss"] * batch_size
            total_weighted_segmentation_loss += metrics["weighted_segmentation_loss"] * batch_size
            total_depth_loss += metrics["depth_loss"] * batch_size
            total_weighted_depth_loss += metrics["weighted_depth_loss"] * batch_size
            total_ade += ade.item() * batch_size
            total_fde += fde.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "ade_loss": total_ade_loss / total_samples,
        "xy_loss": total_xy_loss / total_samples,
        "heading_loss": total_heading_loss / total_samples,
        "fde_loss": total_fde_loss / total_samples,
        "segmentation_loss": total_segmentation_loss / total_samples,
        "weighted_segmentation_loss": total_weighted_segmentation_loss / total_samples,
        "depth_loss": total_depth_loss / total_samples,
        "weighted_depth_loss": total_weighted_depth_loss / total_samples,
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
    }


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
        command_feature_dim=args.command_feature_dim,
        history_layers=args.history_layers,
        fusion_dim=args.fusion_dim,
        fusion_heads=args.fusion_heads,
        dropout=args.dropout,
        future_steps=args.future_steps,
        use_segmentation_head=args.use_segmentation_head,
        use_depth_head=args.use_depth_head,
        num_segmentation_classes=args.num_segmentation_classes,
        use_layer3_spatial_pooling=args.use_layer3_spatial_pooling,
        layer3_spatial_scale=args.layer3_spatial_scale,
        layer3_spatial_grid_height=args.layer3_spatial_grid_height,
        layer3_spatial_grid_width=args.layer3_spatial_grid_width,
    ).to(device)

    if args.freeze_backbone:
        freeze_backbone_parameters(model)

    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("stem_layer3.") or name.startswith("stem_layer4."):
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

        # Filter out keys with shape mismatches
        filtered_state = {}
        skipped_keys = []
        for key, value in loaded_state.items():
            if key in model_state and value.shape != model_state[key].shape:
                skipped_keys.append(f"{key}: {value.shape} -> {model_state[key].shape}")
            else:
                filtered_state[key] = value

        load_result = model.load_state_dict(filtered_state, strict=False)
        if skipped_keys:
            print(f"  Skipped size-mismatched keys (re-initialized):")
            for sk in skipped_keys:
                print(f"    {sk}")
        if reset_keys:
            print(f"  Reset trajectory decoder/head weights: {reset_keys}")
        if load_result.missing_keys:
            print(f"  Missing keys (randomly initialized): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  Unexpected keys (ignored): {load_result.unexpected_keys}")

        if resume_mode == "full":
            try:
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
            except ValueError as exc:
                print(
                    "  Optimizer state could not be restored after the fusion-architecture change; "
                    f"falling back to weights-only resume. Details: {exc}"
                )
        else:
            print("  Weights-only mode: optimizer and scheduler reset from scratch")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            freeze_backbone_batchnorm(model)
        aux_scale = compute_aux_scale(
            epoch,
            warmup_epochs=args.aux_warmup_epochs,
            ramp_epochs=args.aux_ramp_epochs,
        )
        effective_segmentation_weight = (
            args.segmentation_loss_weight * aux_scale if args.use_segmentation_head else 0.0
        )
        effective_depth_weight = (
            args.depth_loss_weight * aux_scale if args.use_depth_head else 0.0
        )
        running_loss = 0.0
        running_ade_loss = 0.0
        running_xy_loss = 0.0
        running_heading_loss = 0.0
        running_fde_loss = 0.0
        running_segmentation_loss = 0.0
        running_weighted_segmentation_loss = 0.0
        running_depth_loss = 0.0
        running_weighted_depth_loss = 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            camera = batch["camera"].to(device)
            history_batch = batch["history"].to(device)
            command = batch["command"].to(device)

            targets = {
                "future": batch["future"].to(device),
                "semantic_label": batch["semantic_label"].to(device),
            }
            if args.use_depth_head and "depth" in batch:
                targets["depth"] = batch["depth"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(camera, history_batch, command)
            loss, loss_metrics = compute_total_loss(
                outputs,
                targets,
                time_weights=time_weights,
                heading_weight=args.heading_weight,
                fde_weight=args.fde_weight,
                ade_weight=args.ade_weight,
                segmentation_loss_weight=effective_segmentation_weight,
                depth_loss_weight=effective_depth_weight,
                use_segmentation_head=args.use_segmentation_head,
                use_depth_head=args.use_depth_head,
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
            running_segmentation_loss += loss_metrics["segmentation_loss"] * batch_size
            running_weighted_segmentation_loss += loss_metrics["weighted_segmentation_loss"] * batch_size
            running_depth_loss += loss_metrics["depth_loss"] * batch_size
            running_weighted_depth_loss += loss_metrics["weighted_depth_loss"] * batch_size
            progress.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                ade=f"{running_ade_loss / total_samples:.4f}",
                xy=f"{running_xy_loss / total_samples:.4f}",
                fde=f"{running_fde_loss / total_samples:.4f}",
                auxs=f"{running_weighted_segmentation_loss / total_samples:.4f}",
                auxd=f"{running_weighted_depth_loss / total_samples:.4f}",
            )

        scheduler.step()
        train_stats = {
            "loss": running_loss / total_samples,
            "ade_loss": running_ade_loss / total_samples,
            "xy_loss": running_xy_loss / total_samples,
            "heading_loss": running_heading_loss / total_samples,
            "fde_loss": running_fde_loss / total_samples,
            "segmentation_loss": running_segmentation_loss / total_samples,
            "weighted_segmentation_loss": running_weighted_segmentation_loss / total_samples,
            "depth_loss": running_depth_loss / total_samples,
            "weighted_depth_loss": running_weighted_depth_loss / total_samples,
        }
        val_stats = evaluate(
            model,
            val_loader,
            device=device,
            time_weights=time_weights,
            heading_weight=args.heading_weight,
            fde_weight=args.fde_weight,
            ade_weight=args.ade_weight,
            segmentation_loss_weight=effective_segmentation_weight,
            depth_loss_weight=effective_depth_weight,
            use_segmentation_head=args.use_segmentation_head,
            use_depth_head=args.use_depth_head,
        )
        epoch_summary = {
            "epoch": epoch,
            "aux_scale": aux_scale,
            "train_loss": train_stats["loss"],
            "train_ade_loss": train_stats["ade_loss"],
            "train_xy_loss": train_stats["xy_loss"],
            "train_heading_loss": train_stats["heading_loss"],
            "train_fde_loss": train_stats["fde_loss"],
            "train_segmentation_loss": train_stats["segmentation_loss"],
            "train_weighted_segmentation_loss": train_stats["weighted_segmentation_loss"],
            "train_depth_loss": train_stats["depth_loss"],
            "train_weighted_depth_loss": train_stats["weighted_depth_loss"],
            "val_loss": val_stats["loss"],
            "val_ade_loss": val_stats["ade_loss"],
            "val_xy_loss": val_stats["xy_loss"],
            "val_heading_loss": val_stats["heading_loss"],
            "val_fde_loss": val_stats["fde_loss"],
            "val_segmentation_loss": val_stats["segmentation_loss"],
            "val_weighted_segmentation_loss": val_stats["weighted_segmentation_loss"],
            "val_depth_loss": val_stats["depth_loss"],
            "val_weighted_depth_loss": val_stats["weighted_depth_loss"],
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
            f"train_auxs={train_stats['weighted_segmentation_loss']:.4f} | "
            f"train_auxd={train_stats['weighted_depth_loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_ade_loss={val_stats['ade_loss']:.4f} | "
            f"val_xy={val_stats['xy_loss']:.4f} | "
            f"val_fde_loss={val_stats['fde_loss']:.4f} | "
            f"val_auxs={val_stats['weighted_segmentation_loss']:.4f} | "
            f"val_auxd={val_stats['weighted_depth_loss']:.4f} | "
            f"aux_scale={aux_scale:.2f} | "
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
                "use_segmentation_head": args.use_segmentation_head,
                "use_depth_head": args.use_depth_head,
                "depth_loss_weight": args.depth_loss_weight,
                "use_layer3_spatial_pooling": args.use_layer3_spatial_pooling,
                "layer3_spatial_scale": args.layer3_spatial_scale,
                "layer3_spatial_grid_height": args.layer3_spatial_grid_height,
                "layer3_spatial_grid_width": args.layer3_spatial_grid_width,
                "ade_weight": args.ade_weight,
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
            "fde_weight": args.fde_weight,
            "ade_weight": args.ade_weight,
            "use_layer3_spatial_pooling": args.use_layer3_spatial_pooling,
            "layer3_spatial_scale": args.layer3_spatial_scale,
            "use_segmentation_head": args.use_segmentation_head,
            "segmentation_loss_weight": args.segmentation_loss_weight,
            "use_depth_head": args.use_depth_head,
            "depth_loss_weight": args.depth_loss_weight,
            "aux_warmup_epochs": args.aux_warmup_epochs,
            "aux_ramp_epochs": args.aux_ramp_epochs,
        },
    )
    print(f"Best checkpoint saved to {output_dir / 'best_checkpoint.pt'}")
    print(f"Best validation ADE: {best_val_ade:.4f}")


if __name__ == "__main__":
    main()
