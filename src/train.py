from __future__ import annotations

import argparse
import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DrivingDataset, list_pickle_files
from src.metrics import displacement_errors, multitask_loss
from src.model import EgoDrivePlanner
from src.utils import ensure_dir, get_device, save_json, set_seed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the hybrid planner with local trajectory targets and multiscale fusion."
    )
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--val-dir", default="val")
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
    parser.add_argument("--command-feature-dim", type=int, default=32)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--num-segmentation-classes", type=int, default=15)
    parser.add_argument("--use-depth-head", action="store_true")
    parser.add_argument("--use-segmentation-head", action="store_true")
    parser.add_argument("--use-depth-token", action="store_true")
    parser.add_argument("--use-segmentation-token", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--depth-loss-weight", type=float, default=0.01)
    parser.add_argument("--segmentation-loss-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser


def freeze_backbone_parameters(model: EgoDrivePlanner) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def freeze_backbone_batchnorm(model: EgoDrivePlanner) -> None:
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)


def evaluate(
    model: EgoDrivePlanner,
    dataloader: DataLoader,
    *,
    device: torch.device,
    heading_weight: float,
    depth_loss_weight: float,
    segmentation_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_xy_loss = 0.0
    total_heading_loss = 0.0
    total_depth_loss = 0.0
    total_segmentation_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)
            future = batch["future"].to(device)
            depth = batch["depth"].to(device)
            semantic_label = batch["semantic_label"].to(device)

            outputs = model(camera, history, command)
            loss, metrics = multitask_loss(
                outputs["trajectory"],
                future,
                outputs.get("depth"),
                depth,
                outputs.get("segmentation_logits"),
                semantic_label,
                heading_weight=heading_weight,
                depth_loss_weight=depth_loss_weight,
                segmentation_loss_weight=segmentation_loss_weight,
            )
            ade, fde = displacement_errors(outputs["trajectory"], future)

            batch_size = camera.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_xy_loss += metrics["xy_loss"] * batch_size
            total_heading_loss += metrics["heading_loss"] * batch_size
            total_depth_loss += metrics["depth_loss"] * batch_size
            total_segmentation_loss += metrics["segmentation_loss"] * batch_size
            total_ade += ade.item() * batch_size
            total_fde += fde.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "xy_loss": total_xy_loss / total_samples,
        "heading_loss": total_heading_loss / total_samples,
        "depth_loss": total_depth_loss / total_samples,
        "segmentation_loss": total_segmentation_loss / total_samples,
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
        use_depth_head=args.use_depth_head,
        use_segmentation_head=args.use_segmentation_head,
        use_depth_token=args.use_depth_token,
        use_segmentation_token=args.use_segmentation_token,
        num_segmentation_classes=args.num_segmentation_classes,
    ).to(device)

    if args.freeze_backbone:
        freeze_backbone_parameters(model)

    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
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
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_backbone_batchnorm(model)
        running_loss = 0.0
        running_xy_loss = 0.0
        running_heading_loss = 0.0
        running_depth_loss = 0.0
        running_segmentation_loss = 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            camera = batch["camera"].to(device)
            history_batch = batch["history"].to(device)
            command = batch["command"].to(device)
            future = batch["future"].to(device)
            depth = batch["depth"].to(device)
            semantic_label = batch["semantic_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(camera, history_batch, command)
            loss, loss_metrics = multitask_loss(
                outputs["trajectory"],
                future,
                outputs.get("depth"),
                depth,
                outputs.get("segmentation_logits"),
                semantic_label,
                heading_weight=args.heading_weight,
                depth_loss_weight=args.depth_loss_weight,
                segmentation_loss_weight=args.segmentation_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = camera.size(0)
            total_samples += batch_size
            running_loss += loss_metrics["loss"] * batch_size
            running_xy_loss += loss_metrics["xy_loss"] * batch_size
            running_heading_loss += loss_metrics["heading_loss"] * batch_size
            running_depth_loss += loss_metrics["depth_loss"] * batch_size
            running_segmentation_loss += loss_metrics["segmentation_loss"] * batch_size
            progress.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                xy=f"{running_xy_loss / total_samples:.4f}",
                depth=f"{running_depth_loss / total_samples:.4f}",
                seg=f"{running_segmentation_loss / total_samples:.4f}",
            )

        scheduler.step()
        train_stats = {
            "loss": running_loss / total_samples,
            "xy_loss": running_xy_loss / total_samples,
            "heading_loss": running_heading_loss / total_samples,
            "depth_loss": running_depth_loss / total_samples,
            "segmentation_loss": running_segmentation_loss / total_samples,
        }
        val_stats = evaluate(
            model,
            val_loader,
            device=device,
            heading_weight=args.heading_weight,
            depth_loss_weight=args.depth_loss_weight,
            segmentation_loss_weight=args.segmentation_loss_weight,
        )
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_xy_loss": train_stats["xy_loss"],
            "train_heading_loss": train_stats["heading_loss"],
            "train_depth_loss": train_stats["depth_loss"],
            "train_segmentation_loss": train_stats["segmentation_loss"],
            "val_loss": val_stats["loss"],
            "val_xy_loss": val_stats["xy_loss"],
            "val_heading_loss": val_stats["heading_loss"],
            "val_depth_loss": val_stats["depth_loss"],
            "val_segmentation_loss": val_stats["segmentation_loss"],
            "val_ade": val_stats["ade"],
            "val_fde": val_stats["fde"],
            "backbone_lr": scheduler.get_last_lr()[0],
            "head_lr": scheduler.get_last_lr()[-1],
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_xy={train_stats['xy_loss']:.4f} | "
            f"train_depth={train_stats['depth_loss']:.4f} | "
            f"train_seg={train_stats['segmentation_loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_xy={val_stats['xy_loss']:.4f} | "
            f"val_depth={val_stats['depth_loss']:.4f} | "
            f"val_seg={val_stats['segmentation_loss']:.4f} | "
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
                "use_depth_head": args.use_depth_head,
                "use_segmentation_head": args.use_segmentation_head,
                "use_depth_token": args.use_depth_token,
                "use_segmentation_token": args.use_segmentation_token,
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
            "depth_loss_weight": args.depth_loss_weight,
            "segmentation_loss_weight": args.segmentation_loss_weight,
        },
    )
    print(f"Best checkpoint saved to {output_dir / 'best_checkpoint.pt'}")
    print(f"Best validation ADE: {best_val_ade:.4f}")


if __name__ == "__main__":
    main()
