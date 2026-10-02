from __future__ import annotations

import argparse

from .config import load_config
from .data import list_pickle_files
from .evaluation import load_checkpoint_model, predict_to_csv, validate_submission
from .utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a trajectory submission")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    model, checkpoint = load_checkpoint_model(args.checkpoint, get_device(args.device))
    files = list_pickle_files(data["test_dir"], data.get("max_test_samples"))
    output = predict_to_csv(model, files, args.output_csv, device=get_device(args.device), image_height=int(data.get("image_height", 224)), image_width=int(data.get("image_width", 336)), batch_size=int(config.get("training", {}).get("batch_size", 32)), num_workers=int(config.get("training", {}).get("num_workers", 0)), require_command=bool(checkpoint["model_config"].get("use_command", False)))
    validate_submission(output, int(checkpoint["model_config"].get("future_steps", 60)))
    print(f"Saved validated submission to {output}")


if __name__ == "__main__":
    main()
