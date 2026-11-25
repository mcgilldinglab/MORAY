#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "read_id",
    "true_phylum_label",
    "true_genus_label",
    "true_species_label",
    "total_valid_kmers",
    "matched_kmers",
    "top_labels",
    "top_scores",
    "top_k25_masks_hex",
]


def refuse_login_node() -> None:
    host = socket.gethostname().split(".")[0]
    if host.startswith("login") and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(f"Refusing to aggregate k25 masks on login node {host}; submit the Slurm job.")


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_csv_ints(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part != ""]


def parse_csv_strs(value: str) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(",") if part != ""]


def parse_mask(value: str) -> tuple[int, int]:
    if not value:
        return 0, 0
    mask = int(value, 16)
    low = mask & ((1 << 64) - 1)
    high = mask >> 64
    return low, high


def mask_hex(low: int, high: int) -> str:
    return f"{high:016x}{low:016x}"


def load_species_to_parent(metadata_tsv: Path, level: str) -> tuple[dict[int, int], dict[int, int]]:
    parent_col = f"{level}_label"
    prior: dict[int, int] = defaultdict(int)
    mapping: dict[int, int] = {}
    with metadata_tsv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        for row in reader:
            if "species_label" in fields and parent_col in fields:
                species = int(row["species_label"])
                parent = int(row[parent_col])
                kmer_count = int(float(row.get("species_kmer_count", row.get("kmer_count", "0")) or 0))
            elif fields >= {"level", "label", "kmer_count"}:
                # Long-format k25 metadata does not contain species->parent mappings.
                # It is usable for parent priors but not for aggregating species candidates.
                if row.get("level") == level:
                    prior[int(row["label"])] += int(float(row.get("kmer_count", "0") or 0))
                continue
            else:
                raise KeyError(f"Unsupported metadata format for {metadata_tsv}: {sorted(fields)}")
            mapping[species] = parent
            prior[parent] += kmer_count
    return mapping, dict(prior)


def aggregate_row(row: dict[str, str], species_to_parent: dict[int, int], parent_prior: dict[int, int], top_k: int) -> dict[str, str]:
    labels = parse_csv_ints(row.get("top_labels", ""))
    scores = parse_csv_ints(row.get("top_scores", ""))
    masks = parse_csv_strs(row.get("top_k25_masks_hex", ""))
    grouped: dict[int, dict[str, int]] = {}
    for idx, species_label in enumerate(labels):
        if species_label not in species_to_parent:
            continue
        parent = species_to_parent[species_label]
        score = scores[idx] if idx < len(scores) else 0
        low, high = parse_mask(masks[idx] if idx < len(masks) else "")
        rec = grouped.setdefault(parent, {"score": 0, "low": 0, "high": 0})
        rec["score"] += score
        rec["low"] |= low
        rec["high"] |= high
    ordered = sorted(
        grouped,
        key=lambda label: (-grouped[label]["score"], -parent_prior.get(label, 0), label),
    )[:top_k]
    return {
        "read_id": row["read_id"],
        "true_phylum_label": row.get("true_phylum_label", ""),
        "true_genus_label": row.get("true_genus_label", ""),
        "true_species_label": row.get("true_species_label", ""),
        "total_valid_kmers": row.get("total_valid_kmers", ""),
        "matched_kmers": row.get("species_matched_kmers", row.get("matched_kmers", "")),
        "top_labels": ",".join(str(label) for label in ordered),
        "top_scores": ",".join(str(grouped[label]["score"]) for label in ordered),
        "top_k25_masks_hex": ",".join(mask_hex(grouped[label]["low"], grouped[label]["high"]) for label in ordered),
    }


def main() -> None:
    refuse_login_node()
    parser = argparse.ArgumentParser(description="Aggregate species k25 hit masks into phylum/genus parent hit masks.")
    parser.add_argument("--species-mask-tsv", required=True)
    parser.add_argument("--metadata-tsv", required=True)
    parser.add_argument("--level", choices=["phylum", "genus"], required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    species_mask_tsv = Path(args.species_mask_tsv)
    metadata_tsv = Path(args.metadata_tsv)
    output_tsv = Path(args.output_tsv)
    summary_json = Path(args.summary_json)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    species_to_parent, parent_prior = load_species_to_parent(metadata_tsv, args.level)
    read_count = 0
    nonempty = 0
    with species_mask_tsv.open(newline="") as in_handle, output_tsv.open("w", newline="") as out_handle:
        reader = csv.DictReader(in_handle, delimiter="\t")
        writer = csv.DictWriter(out_handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            out = aggregate_row(row, species_to_parent, parent_prior, args.top_k)
            read_count += 1
            nonempty += int(bool(out["top_labels"]))
            writer.writerow(out)

    summary_json.write_text(json.dumps({
        "status": "completed",
        "created_at": timestamp(),
        "level": args.level,
        "species_mask_tsv": str(species_mask_tsv),
        "metadata_tsv": str(metadata_tsv),
        "output_tsv": str(output_tsv),
        "read_count": read_count,
        "nonempty_parent_candidate_reads": nonempty,
        "top_k": args.top_k,
        "note": "Parent masks are aggregated from the same species-level k25 hit-mask evidence used by the 36B reranker. The k25 default predictions still come from the exact-match prediction table.",
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
