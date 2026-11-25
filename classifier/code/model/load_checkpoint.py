"""Inference functions from the MORAY v7 deployment module evaluate_step54_multicand.py."""


from __future__ import annotations


from pathlib import Path


import torch


from reranker import HierarchyK25LoraReranker, adapt_state_dict_for_model, build_backbone, move_model_to_device_with_retry


def load_model_for_task(config: dict, task: dict, task_index: int, num_labels: int, device: torch.device,
                        epoch: int | None = None, output_dir_override: str | None = None):
    output_dir = Path(output_dir_override) if output_dir_override else Path(config["outputs"]["output_dir"])
    prefix = f"step43_task{task_index:03d}_{task['name']}"
    if epoch is not None:
        checkpoint_path = output_dir / "epoch_checkpoints" / f"{prefix}_epoch{int(epoch):03d}.pt"
    else:
        checkpoint_path = output_dir / f"{prefix}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Step43 checkpoint: {checkpoint_path}")
    backbone = build_backbone(config, task, device)
    model = HierarchyK25LoraReranker(backbone, task, num_labels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(adapt_state_dict_for_model(checkpoint["model_state"], model))
    return move_model_to_device_with_retry(model, device), checkpoint_path
