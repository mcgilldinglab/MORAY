"""Inference functions from the MORAY v7 deployment module train_step54_multicand.py."""


from __future__ import annotations


import csv


import random


from pathlib import Path


import torch


import torch.nn as nn


from peft import LoraConfig, TaskType, get_peft_model


from transformers import AutoModelForCausalLM


from sequence import BASE_TO_ID


from backbone import load_local_tokenizer, masked_max_pool, masked_mean_pool, move_model_to_device_with_retry, prepare_sequence, select_token_embeddings


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def true_column(rank: str) -> str:
    return f"{rank}_label"


def parent_rank(rank: str) -> str | None:
    if rank == "species":
        return "genus"
    if rank == "genus":
        return "phylum"
    return None


def read_truth_iter(path: Path):
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            yield {
                "read_id": row["read_id"],
                "sequence": row["sequence"],
                "phylum": int(row["phylum_label"]),
                "genus": int(row["genus_label"]),
                "species": int(row["species_label"]),
            }


def load_label_parent_map(train_truth_tsv: Path, rank: str) -> dict[int, int]:
    parent = parent_rank(rank)
    if parent is None:
        return {}
    mapping = {}
    for row in read_truth_iter(train_truth_tsv):
        mapping[int(row[rank])] = int(row[parent])
    return mapping


def load_num_labels(metadata_tsv: Path, rank: str, truth_paths: list[Path]) -> int:
    max_label = -1
    if metadata_tsv.exists():
        with metadata_tsv.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or [])
            for row in reader:
                if fields >= {"level", "label"}:
                    if row["level"] == rank:
                        max_label = max(max_label, int(row["label"]))
                else:
                    column = true_column(rank)
                    if column in fields and row.get(column, "") != "":
                        max_label = max(max_label, int(row[column]))
    for path in truth_paths:
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                max_label = max(max_label, int(row[true_column(rank)]))
    if max_label < 0:
        raise ValueError(f"No labels found for rank={rank}")
    return max_label + 1


def load_k25_predictions(path: Path) -> dict[str, dict[str, int]]:
    out = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            out[row["read_id"]] = {
                "phylum": int(row.get("pred_phylum_forced", -1) or -1),
                "genus": int(row.get("pred_genus_forced", -1) or -1),
                "species": int(row.get("pred_species_forced", -1) or -1),
            }
    return out


def parse_int_list(value: str) -> list[int]:
    if value == "":
        return []
    return [int(x) for x in value.split(",") if x != ""]


def parse_str_list(value: str) -> list[str]:
    if value == "":
        return []
    return [x for x in value.split(",") if x != ""]


def load_mask_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            records[row["read_id"]] = {
                "labels": parse_int_list(row.get("top_labels", "")),
                "scores": parse_int_list(row.get("top_scores", "")),
                "masks": parse_str_list(row.get("top_k25_masks_hex", "")),
                "matched": int(row.get("matched_kmers", row.get("species_matched_kmers", 0)) or 0),
                "total": int(row.get("total_valid_kmers", 0) or 0),
            }
    return records


def one_hot_encode(seq: str, length: int) -> torch.Tensor:
    encoded = torch.zeros((5, length), dtype=torch.float32)
    seq = seq.upper()
    limit = min(len(seq), length)
    for idx in range(limit):
        encoded[BASE_TO_ID.get(seq[idx], 4), idx] = 1.0
    if limit < length:
        encoded[4, limit:] = 1.0
    return encoded


def hex_to_bits(value: str, length: int) -> list[float]:
    if not value:
        return [0.0] * length
    mask = int(value, 16)
    return [1.0 if ((mask >> idx) & 1) else 0.0 for idx in range(length)]


def kmer_bits_to_coverage(bits: list[float], read_length: int = 150, k: int = 25) -> list[float]:
    coverage = [0.0] * read_length
    for idx, bit in enumerate(bits):
        if bit <= 0.0:
            continue
        for pos in range(idx, min(read_length, idx + k)):
            coverage[pos] = 1.0
    return coverage


def route_candidates(mask_record: dict, k25_label: int, route_parent: int, label_to_parent: dict[int, int], task: dict):
    max_candidates = int(task.get("max_candidates", 20))
    labels_in = [int(v) for v in mask_record.get("labels", [])]
    masks_in = list(mask_record.get("masks", []))
    selected = []
    selected_masks = []
    seen = set()
    mask_by_label = {label: masks_in[idx] if idx < len(masks_in) else "" for idx, label in enumerate(labels_in)}
    restrict_parent = bool(task.get("restrict_candidates_to_parent", bool(label_to_parent)))
    for idx, label in enumerate(labels_in):
        if restrict_parent and label_to_parent and label_to_parent.get(label, -999999) != route_parent:
            continue
        if label in seen:
            continue
        selected.append(label)
        selected_masks.append(masks_in[idx] if idx < len(masks_in) else "")
        seen.add(label)
        if len(selected) >= max_candidates:
            break
    k25_allowed = k25_label >= 0
    if restrict_parent and label_to_parent:
        k25_allowed = k25_allowed and label_to_parent.get(k25_label, -999999) == route_parent
    if k25_allowed and k25_label not in seen:
        selected.insert(0, k25_label)
        selected_masks.insert(0, mask_by_label.get(k25_label, ""))
        selected = selected[:max_candidates]
        selected_masks = selected_masks[:max_candidates]
    return selected, selected_masks


class HierarchyMaskCollator:
    def __init__(self, tokenizer, task: dict, training: dict):
        self.tokenizer = tokenizer
        self.task = task
        self.sequence_length = int(training.get("sequence_length", 150))
        self.kmer_length = int(training.get("kmer_mask_length", 126))

    def __call__(self, batch: list[dict]) -> dict:
        max_candidates = max(len(item["candidate_labels"]) for item in batch)
        labels = torch.full((len(batch), max_candidates), -1, dtype=torch.long)
        mask_len = self.sequence_length if str(self.task.get("mask_view", "kmer")) == "coverage" else self.kmer_length
        hit_masks = torch.zeros((len(batch), max_candidates, mask_len), dtype=torch.float32)
        candidate_mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
        for row_i, item in enumerate(batch):
            n = len(item["candidate_labels"])
            labels[row_i, :n] = torch.tensor(item["candidate_labels"], dtype=torch.long)
            candidate_mask[row_i, :n] = True
            for cand_i, hex_value in enumerate(item["candidate_masks_hex"]):
                bits = hex_to_bits(hex_value, self.kmer_length)
                if str(self.task.get("mask_view", "kmer")) == "coverage":
                    bits = kmer_bits_to_coverage(bits, self.sequence_length, 25)
                hit_masks[row_i, cand_i] = torch.tensor(bits, dtype=torch.float32)
        out = {
            "read_ids": [item["read_id"] for item in batch],
            "candidate_labels": labels,
            "k25_hit_masks": hit_masks,
            "candidate_mask": candidate_mask,
            "target_index": torch.tensor([item["target_index"] for item in batch], dtype=torch.long),
            "k25_index": torch.tensor([item["k25_index"] for item in batch], dtype=torch.long),
            "gate_label": torch.tensor([item["gate_label"] for item in batch], dtype=torch.float32),
            "true_label": torch.tensor([item["true_label"] for item in batch], dtype=torch.long),
            "true_parent": torch.tensor([item["true_parent"] for item in batch], dtype=torch.long),
            "pred_parent": torch.tensor([item["pred_parent"] for item in batch], dtype=torch.long),
            "k25_pred_label": torch.tensor([item["k25_pred_label"] for item in batch], dtype=torch.long),
            "domain_id": torch.tensor([item["domain_id"] for item in batch], dtype=torch.long),
            "row_type": [item["row_type"] for item in batch],
        }
        sequences = [item["sequence"] for item in batch]
        encoded = self.tokenizer(
            [prepare_sequence(seq) for seq in sequences],
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.task.get("max_token_length", 64)),
        )
        out["input_ids"] = encoded["input_ids"]
        out["attention_mask"] = encoded["attention_mask"]
        if self.task.get("use_ohe", False):
            out["ohe"] = torch.stack([one_hot_encode(seq, self.sequence_length) for seq in sequences], dim=0)
        return out


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([nn.LayerNorm(prev), nn.Linear(prev, int(hidden)), nn.GELU(), nn.Dropout(dropout)])
            prev = int(hidden)
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=kernel // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=kernel // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class OHEEncoder(nn.Module):
    def __init__(self, task: dict, dropout: float):
        super().__init__()
        channels = int(task.get("ohe_channels", 128))
        output_dim = int(task.get("ohe_output_dim", 160))
        self.stem = nn.Conv1d(5, channels, kernel_size=9, padding=4)
        self.blocks = nn.Sequential(*[ResidualConvBlock(channels, 9, dropout) for _ in range(2)])
        self.proj = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.blocks(self.stem(x)))


class HitMaskEncoder(nn.Module):
    def __init__(self, task: dict, dropout: float):
        super().__init__()
        channels = int(task.get("mask_channels", 96))
        output_dim = int(task.get("mask_output_dim", 128))
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.GELU(),
            ResidualConvBlock(channels, 7, dropout),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_cand, length = x.shape
        encoded = self.net(x.reshape(bsz * n_cand, 1, length))
        return encoded.reshape(bsz, n_cand, self.output_dim)


class HierarchyK25LoraReranker(nn.Module):
    def __init__(self, backbone, task: dict, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.num_labels = int(num_labels)
        self.padding_label = self.num_labels
        dropout = float(task.get("dropout", 0.12))

        pooling = str(task.get("pooling_type", "mean"))
        generator_dim = int(task.get("embedding_dim", 2048)) * (2 if pooling == "meanmax" else 1)
        self.generator_encoder = MLPEncoder(generator_dim, [int(v) for v in task.get("generator_hidden_dims", [1024, 384])], dropout)
        read_parts = [self.generator_encoder.output_dim]
        self.ohe_encoder = OHEEncoder(task, dropout) if task.get("use_ohe", False) else None
        if self.ohe_encoder is not None:
            read_parts.append(self.ohe_encoder.output_dim)
        # The domain embedding indexed the ERROR-RATE stratum (err000->0, err005->1, err010->2). Training
        # only ever saw train_sp2000_err000, so rows 1-4 never received a gradient: every evaluation on a
        # noisy split fed the model a random, never-learned vector. And it can never be used in production
        # anyway -- a real read does not arrive with its error rate. Keeping it and always passing 0 would
        # be indefensible ("then why is it there?"), so it is off. The model does not condition on the
        # error rate. Default True so the old checkpoints still load.
        self.domain_embedding = None
        if bool(task.get("use_domain_embedding", True)):
            self.domain_embedding_bins = int(task.get("domain_embedding_bins", 5))
            self.domain_embedding = nn.Embedding(self.domain_embedding_bins, int(task.get("domain_embedding_dim", 16)))
            read_parts.append(int(task.get("domain_embedding_dim", 16)))
        self.parent_embedding = None
        if task.get("use_parent_embedding", False):
            self.parent_embedding = nn.Embedding(int(task.get("parent_embedding_bins", 4096)), int(task.get("parent_embedding_dim", 48)))
            read_parts.append(int(task.get("parent_embedding_dim", 48)))
        self.read_encoder = MLPEncoder(sum(read_parts), [int(v) for v in task.get("read_hidden_dims", [448])], dropout)
        read_dim = self.read_encoder.output_dim

        cand_dim = int(task.get("candidate_embedding_dim", 128))
        k25_dim = int(task.get("k25_embedding_dim", cand_dim))
        self.candidate_embedding = nn.Embedding(self.num_labels + 1, cand_dim, padding_idx=self.padding_label)
        self.k25_embedding = nn.Embedding(self.num_labels + 1, k25_dim, padding_idx=self.padding_label)
        self.mask_encoder = HitMaskEncoder(task, dropout)
        candidate_repr_dim = cand_dim + k25_dim + self.mask_encoder.output_dim
        self.scorer = MLPEncoder(read_dim + candidate_repr_dim, [int(v) for v in task.get("scorer_hidden_dims", [576, 128])], dropout)
        self.score_head = nn.Linear(self.scorer.output_dim, 1)
        self.gate = MLPEncoder(read_dim + candidate_repr_dim * 2 + 3, [int(v) for v in task.get("gate_hidden_dims", [576, 128])], dropout)
        self.gate_head = nn.Linear(self.gate.output_dim, 1)

    def encode_generator(self, batch: dict) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        token_embeddings = select_token_embeddings(outputs.hidden_states, str(self.task.get("sequence_source", "last")))
        mean = masked_mean_pool(token_embeddings, batch["attention_mask"])
        if str(self.task.get("pooling_type", "mean")) == "meanmax":
            return torch.cat([mean, masked_max_pool(token_embeddings, batch["attention_mask"])], dim=1)
        return mean

    def encode_read(self, batch: dict) -> torch.Tensor:
        parts = [self.generator_encoder(self.encode_generator(batch))]
        if self.ohe_encoder is not None:
            parts.append(self.ohe_encoder(batch["ohe"]))
        if self.domain_embedding is not None:
            parts.append(self.domain_embedding(batch["domain_id"]))
        if self.parent_embedding is not None:
            parent_ids = batch["pred_parent"].clamp_min(0) % int(self.task.get("parent_embedding_bins", 4096))
            parts.append(self.parent_embedding(parent_ids))
        return self.read_encoder(torch.cat(parts, dim=1))

    def candidate_repr(self, batch: dict) -> torch.Tensor:
        cand_labels = batch["candidate_labels"].clamp_min(0)
        cand_labels = torch.where(batch["candidate_mask"], cand_labels, torch.full_like(cand_labels, self.padding_label))
        cand_labels = torch.where(cand_labels < self.num_labels, cand_labels, torch.full_like(cand_labels, self.padding_label))
        cand_emb = self.candidate_embedding(cand_labels)
        k25 = batch["k25_pred_label"].clamp_min(0)
        k25 = torch.where(k25 < self.num_labels, k25, torch.full_like(k25, self.padding_label))
        k25_emb = self.k25_embedding(k25).unsqueeze(1).expand(-1, cand_labels.size(1), -1)
        mask_emb = self.mask_encoder(batch["k25_hit_masks"])
        return torch.cat([cand_emb, k25_emb, mask_emb], dim=2)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        read = self.encode_read(batch)
        cand_repr = self.candidate_repr(batch)

        # THE HEADS RUN IN fp32, THE BACKBONE STAYS IN bf16.
        #
        # bfloat16 keeps fp32's exponent range but only 8 mantissa bits: around magnitude 1 its resolution
        # is ~0.008. Two things depend on finer distinctions than that.
        #   * The scorer's whole job is to separate 12 candidates. Any two whose true logits differ by less
        #     than a bf16 ulp collapse to the SAME value, and argmax then silently falls back to index
        #     order -- i.e. to K25's ranking, the very thing the re-ranker exists to revise.
        #   * The gate's sigmoid is the number the threshold is calibrated against. In bf16 it takes only
        #     ~256 distinct values, which is why every calibrated threshold came out an exact binary
        #     fraction (0.9765625 = 125/128, 0.98828125 = 253/256). The threshold search was quantised.
        # These heads are 576->128->1 MLPs; all the compute is in the 1.2B backbone, so fp32 here is free.
        with torch.autocast(device_type=read.device.type, enabled=False):
            read = read.float()
            cand_repr = cand_repr.float()
            read_exp = read.unsqueeze(1).expand(-1, cand_repr.size(1), -1)
            scores = self.score_head(self.scorer(torch.cat([read_exp, cand_repr], dim=2))).squeeze(2)
            scores = scores.masked_fill(~batch["candidate_mask"], -1.0e9)
            top_idx = scores.argmax(dim=1)
            top_repr = cand_repr.gather(1, top_idx.view(-1, 1, 1).expand(-1, 1, cand_repr.size(2))).squeeze(1)
            safe_k25 = batch["k25_index"].clamp_min(0)
            k25_repr = cand_repr.gather(1, safe_k25.view(-1, 1, 1).expand(-1, 1, cand_repr.size(2))).squeeze(1)
            top_score = scores.gather(1, top_idx.view(-1, 1)).squeeze(1)
            k25_score = scores.gather(1, safe_k25.view(-1, 1)).squeeze(1)
            k25_score = torch.where(batch["k25_index"] >= 0, k25_score, torch.full_like(k25_score, -1.0e9))
            margin = top_score - k25_score
            gate_in = torch.cat([read, top_repr, k25_repr, top_score.unsqueeze(1), k25_score.unsqueeze(1), margin.unsqueeze(1)], dim=1)
            return scores, self.gate_head(self.gate(gate_in)).squeeze(1)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def final_pred(base: dict, rec: dict | None, threshold: float) -> tuple[int, int, list[int]]:
    k25 = int(base["k25_pred_label"])
    if rec is None:
        return k25, 0, [k25]
    if bool(rec.get("k25_in_candidates", False)) and int(rec["pred_label"]) != k25 and float(rec["gate_prob"]) >= threshold:
        return int(rec["pred_label"]), 1, rec["top3_labels"]
    return k25, 0, [k25] + [v for v in rec["top3_labels"] if v != k25]


def threshold_for_read(mode: str, thresholds: dict, base: dict) -> float:
    if mode in {"domain", "domain_global"}:
        return float(thresholds.get(str(base.get("domain_id", 0)), 1.0e9))
    if mode == "parent":
        return float(thresholds.get(str(base.get("k25_pred_parent", -1)), 1.0e9))
    return float(thresholds.get("global", 1.0e9))


def build_backbone(config: dict, task: dict, device: torch.device):
    model_dir = Path(config["inputs"]["model_dir"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    base_model.config.use_cache = False
    lora_config = LoraConfig(
        r=int(task["lora_r"]),
        lora_alpha=int(task["lora_alpha"]),
        lora_dropout=float(task.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=task["target_modules"],
    )
    return move_model_to_device_with_retry(get_peft_model(base_model, lora_config), device)


def adapt_state_dict_for_model(state_dict: dict, model: nn.Module) -> dict:
    state_dict = dict(state_dict)
    key = "domain_embedding.weight"
    if key not in state_dict or not hasattr(model, "domain_embedding"):
        return state_dict
    src = state_dict[key]
    dst = model.state_dict()[key]
    if tuple(src.shape) == tuple(dst.shape):
        return state_dict
    if src.ndim == 2 and dst.ndim == 2 and src.shape[1] == dst.shape[1] and src.shape[0] < dst.shape[0]:
        expanded = dst.detach().clone()
        expanded[: src.shape[0]] = src
        expanded[src.shape[0] :] = src[-1:].expand(dst.shape[0] - src.shape[0], -1)
        state_dict[key] = expanded
        print(
            f"adapted checkpoint {key} from shape={tuple(src.shape)} to shape={tuple(dst.shape)}",
            flush=True,
        )
        return state_dict
    raise ValueError(f"Cannot adapt checkpoint {key} from shape={tuple(src.shape)} to shape={tuple(dst.shape)}")
