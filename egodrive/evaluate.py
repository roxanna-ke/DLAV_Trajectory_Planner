from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import DrivingDataset, list_pickle_files
from .evaluation import load_checkpoint_model
from .metrics import displacement_errors
from .utils import get_device, move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trajectory checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    device = get_device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    files = list_pickle_files(data.get("val_dir") or data.get("real_dir"))
    dataset = DrivingDataset(files, image_height=int(data.get("image_height", 224)), image_width=int(data.get("image_width", 336)), require_command=bool(checkpoint["model_config"].get("use_command", False)), include_auxiliary=False)
    loader = DataLoader(dataset, batch_size=int(config.get("training", {}).get("batch_size", 32)), shuffle=False)
    ade_total = fde_total = count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            kwargs = {"command": batch["command"]} if "command" in batch else {}
            outputs = model(batch["camera"], batch["history"], **kwargs)
            ade, fde = displacement_errors(outputs["trajectory"], batch["future"])
            size = batch["camera"].size(0)
            ade_total += ade.item() * size
            fde_total += fde.item() * size
            count += size
    print(f"ADE: {ade_total / count:.6f}")
    print(f"FDE: {fde_total / count:.6f}")


if __name__ == "__main__":
    main()
