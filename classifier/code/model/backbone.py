"""Inference functions from the MORAY v7 deployment module train_lora_generator_phylum.py."""


from __future__ import annotations


import importlib.util


import json


import time


from pathlib import Path


import torch


import torch.nn as nn


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def move_model_to_device_with_retry(model: nn.Module, device: torch.device, attempts: int = 20, sleep_seconds: int = 60) -> nn.Module:
    for attempt in range(1, attempts + 1):
        try:
            return model.to(device)
        except Exception as exc:
            message = str(exc)
            retryable = "busy or unavailable" in message or "CUDA-capable device" in message
            if not retryable or attempt == attempts:
                raise
            print(
                f"CUDA device temporarily unavailable during model.to(device); "
                f"attempt {attempt}/{attempts}, sleeping {sleep_seconds}s",
                flush=True,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(sleep_seconds)
    return model


def load_local_tokenizer(model_dir: Path):
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    tokenizer_path = model_dir / "tokenizer.py"
    spec = importlib.util.spec_from_file_location("generator_local_tokenizer", tokenizer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    tokenizer = module.DNAKmerTokenizer(
        k=int(tokenizer_config.get("k", tokenizer_config.get("kmer", 6))),
        model_max_length=int(tokenizer_config.get("model_max_length", 16384)),
        padding_side="right",
        bos_token=tokenizer_config.get("bos_token", "<s>"),
        eos_token=tokenizer_config.get("eos_token", "</s>"),
        pad_token=tokenizer_config.get("pad_token", "<pad>"),
        unk_token=tokenizer_config.get("unk_token", "<oov>"),
    )
    tokenizer.padding_side = "right"
    return tokenizer


def prepare_sequence(sequence: str) -> str:
    sequence = sequence.upper()
    usable_length = len(sequence) // 6 * 6
    return sequence[:usable_length]


def masked_mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    summed = (token_embeddings * expanded_mask).sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def masked_max_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = token_embeddings.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
    pooled = masked.max(dim=1).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def select_token_embeddings(hidden_states: tuple[torch.Tensor, ...], sequence_source: str) -> torch.Tensor:
    if sequence_source == "last4avg":
        return torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
    if sequence_source != "last":
        raise ValueError(f"Unsupported sequence_source={sequence_source}")
    return hidden_states[-1]
