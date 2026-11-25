#!/usr/bin/env python3
"""Resumable variant of predict_chain_54.py -- adds checkpoint/resume so a job killed by a wall-time
limit partway through does not lose all its progress. See pitfalls_and_fixes.md section 8.8 for the
full rationale; summary:

  - Per-RANK checkpoint (out.checkpoint.{rank}.tsv + a .done marker): once a rank's full final[rank]
    (every read that has a mask + candidates at that rank, not just the ones scored by the model) is
    computed, it's written to disk. On the next invocation, if a rank's .done marker exists, that
    rank is skipped ENTIRELY -- no dataset build, no model load, no forward pass -- final[rank] and
    route_parent_by_read are just read back from the checkpoint file.
  - Per-BATCH checkpoint within an incomplete rank (out.checkpoint.{rank}.dlbatches.tsv, no .done
    marker until the rank's own loop finishes): infer_split_vec_resumable() below appends each
    batch's model output to this file as soon as it's computed, and on resume, skips re-running the
    model for any batch whose read_ids are ALL already present in this file. A batch that's only
    partially covered is recomputed whole (batches are small; this is a minor, not correctness,
    inefficiency).
  - Memory: once a rank's checkpoint is written, this script keeps only route_parent_by_read (a
    read_id->label map) in memory for that rank, not the full final[rank] dict (label+gate_prob for
    every read) -- the final merged output is built by reading all three rank checkpoints back from
    disk at the very end, not by holding three ranks' worth of full results in memory throughout.

Per-read output values are required to be identical to predict_chain_54.py's regardless of whether a
read's result came from a fresh forward pass in this invocation or was reloaded from an earlier
interrupted run's checkpoint -- verified by diff before this script is used for real samples.

Original predict_chain_54.py docstring follows unchanged:

Strict-chain species prediction for reads with NO ground truth. The deployment inference path.

The Step43 evaluator cannot do this: it builds its dataset by iterating a truth TSV and needs
true_label/true_parent for every read. On real data the labels are the thing we are predicting, so
that path is unusable. This script is the missing piece.

REUSE, DO NOT REWRITE. Every decision function -- route_candidates, final_pred, threshold_for_read,
the collator, the model, the checkpoint loader -- is imported from the original training/eval modules.
Reimplementing them would let deployment silently drift from what was evaluated. The only thing this
file adds is a dataset that supplies reads instead of truth rows, with the truth-only fields filled
with dummies. That is provably safe: the model's forward() reads exactly input_ids, attention_mask,
candidate_labels, candidate_mask, domain_id, ohe, pred_parent, k25_index -- and none of
true_label/true_parent/target_index/gate_label/row_type, which exist only for loss and metrics.

The chain (each rank's prediction restricts the next rank's candidate scope):
    phylum  -> restricts genus candidates
    genus   -> restricts species candidates
    species -> the answer
Routing always uses the PREDICTED parent, never the truth.

Thresholds: calibrated on validation, applied unchanged. The gate's threshold is looked up by
`domain_id`, which despite the name is the ERROR-RATE stratum (err000->0, err005->1, err010->2), not
the taxonomic domain. Real reads have no known error rate, so deployment uses stratum 0 and the
val_err000 thresholds -- the documented deployable default. Getting this wrong is silent: an unknown
stratum returns a 1e9 threshold, the gate never fires, and the DL contributes nothing while still
appearing to run.

usage:
  predict_strict_chain.py --config <domain config.json> --task-group <group> \
      --reads reads.tsv --masks-phylum p.tsv --masks-genus g.tsv --masks-species s.tsv \
      --thresholds-dir <dir with {rank}__val_err000.thresholds.json> --out predictions.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

# The inference functions and model definitions are in code/model/.
CODE = Path(__file__).resolve().parent
PKG = CODE.parent                       # the copyable code package root ({PKG} in configs)
sys.path.insert(0, str(CODE / "model"))

from reranker import (  # noqa: E402
    HierarchyMaskCollator,
    final_pred,
    load_k25_predictions,
    load_label_parent_map,
    load_local_tokenizer,
    load_mask_records,
    load_num_labels,
    move_batch,
    parent_rank,
    route_candidates,
    set_seed,
    threshold_for_read,
)
from load_checkpoint import load_model_for_task  # noqa: E402

RANKS = ["phylum", "genus", "species"]
DEPLOY_STRATUM = 0          # err000 stratum; see the module docstring
DEPLOY_SPLIT = "val_err000"  # thresholds calibrated here are the deployable default


class PredictDataset(Dataset):
    """The evaluator's dataset with reads in place of truth rows.

    Truth-only fields are dummies. `k25_index` is 0 and the k25 default label is labels[0] -- i.e.
    the top candidate AFTER the parent restriction, not the raw mask top-1. That distinction changes
    the prediction, so it is copied from the evaluator exactly rather than reasoned about.
    """

    def __init__(self, reads, mask_records, k25_top1, task, label_to_parent, route_parent_by_read):
        self.task = task
        self.rank = task["target_rank"]
        self.parent = parent_rank(self.rank)
        self.domain_id = DEPLOY_STRATUM

        self.read_ids, self.sequences = [], []
        self.pred_parent, self.k25_pred_label = [], []
        self.candidate_labels, self.candidate_masks_hex = [], []
        self.skipped_no_candidates = 0
        self.missing_mask = 0

        for read_id, sequence in reads:
            mask_record = mask_records.get(read_id)
            if mask_record is None:
                self.missing_mask += 1
                continue
            route_parent = int(route_parent_by_read.get(read_id, -1)) if self.parent else -1
            labels, masks = route_candidates(
                mask_record, int(k25_top1.get(read_id, -1)), route_parent, label_to_parent, task
            )
            if not labels:
                self.skipped_no_candidates += 1
                continue
            self.read_ids.append(read_id)
            self.sequences.append(sequence)
            self.pred_parent.append(route_parent)
            self.k25_pred_label.append(int(labels[0]))
            self.candidate_labels.append(labels)
            self.candidate_masks_hex.append(masks)

    def __len__(self) -> int:
        return len(self.read_ids)

    def __getitem__(self, i: int) -> dict:
        return {
            "read_id": self.read_ids[i],
            "sequence": self.sequences[i],
            "true_label": -1,        # dummy: loss/metrics only, never reaches forward()
            "true_parent": -1,       # dummy
            "target_index": -1,      # dummy
            "gate_label": 0.0,       # dummy
            "row_type": "predict",   # dummy
            "pred_parent": self.pred_parent[i],
            "k25_pred_label": self.k25_pred_label[i],
            "candidate_labels": self.candidate_labels[i],
            "candidate_masks_hex": self.candidate_masks_hex[i],
            "k25_index": 0,
            "domain_id": self.domain_id,
        }


def read_reads(path: Path):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            yield row["read_id"].strip(), row["sequence"].strip()


def mask_top1(mask_records: dict) -> dict[str, int]:
    """The K25 default prediction. Verified on all 5,490,500 virus test reads to equal the reference
    pipeline's pred_{rank}_forced column for every rank, so the separate k25_predictions file the
    evaluator reads is not needed at inference time."""
    out = {}
    for read_id, rec in mask_records.items():
        labels = rec.get("labels") or []
        out[read_id] = int(labels[0]) if labels else -1
    return out


def undecidable(dataset) -> list[int]:
    """Rows the classifier can actually change. A pre-classifier check -- the model is untouched.

    If a read has exactly ONE candidate at this rank, the model cannot alter the outcome:
      * scores has one column, so argmax picks that candidate;
      * that candidate IS the k25 default (route_candidates puts it at index 0, and it is labels[0]);
      * final_pred only overrides when `rec["pred_label"] != k25`, which is then false.
    So the call is the k25 default no matter what the network outputs. Feeding such a read to a 1.2B
    model is pure cost. Omitting it from the loader leaves it absent from rec_by_read, and
    final_pred(base, rec=None, ...) returns exactly that same k25 default with override=0 -- an
    existing branch of the original code, not new logic. The result is identical by construction.

    Why single-candidate reads are so common: candidates are the K25 mask's labels (only those with a
    non-zero score) restricted to the predicted parent. At phylum there is no parent restriction, so one
    candidate means the read's 25-mers hit exactly one phylum and scored zero on every other -- the
    normal case, because 25-mers are highly phylum-specific. At genus/species a read may hit several
    labels yet keep only one after the parent restriction. Measured on virus: 99.9% / 85.3% / 57.1% of
    reads at phylum / genus / species. The 42.9% of species reads with a real choice are where the DL
    does its work.
    """
    return [i for i, cands in enumerate(dataset.candidate_labels) if len(cands) > 1]


def make_loader(dataset, indices, tokenizer, task, training, batch_size_override: int = 0):
    """Same loader the evaluator builds, over the rows that are actually decidable. infer_split() then
    does the forward pass -- including k25_in_candidates, which is NOT always true: route_candidates
    only inserts the k25 label when it survives the parent restriction, and final_pred must not let the
    DL override when it did not. batch_size_override>0 replaces the config's per-rank eval_batch_size
    (a GPU-VRAM knob; batch size does NOT change per-read output -- no BatchNorm)."""
    workers = int(training.get("num_workers", 4))
    bs = int(batch_size_override) if batch_size_override else int(task.get("eval_batch_size", task["batch_size"]))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=HierarchyMaskCollator(tokenizer, task, training),
    )


_CKPT_FIELDS = ["read_id", "pred_label", "top3_labels", "gate_prob", "k25_in_candidates",
                "pred_parent", "candidate_count"]


def _load_batch_checkpoint(ckpt_path: Path) -> dict[str, dict]:
    # A row can be truncated (missing trailing columns) if the process was killed mid-write on a
    # walltime/OOM TIMEOUT -- csv.DictReader fills absent columns with None rather than raising, so
    # this must be checked explicitly. Such a row was never reliably flushed to begin with, so
    # dropping it is correctness-safe: its read is simply treated as not-yet-checkpointed and gets
    # recomputed by infer_split_vec_resumable, same as any other fresh read (see s011_2 bacteria
    # p4 TIMEOUT -> FAILED-on-resume incident, pitfalls_and_fixes.md 8.11).
    predictions: dict[str, dict] = {}
    if not ckpt_path.exists():
        return predictions
    n_dropped = 0
    with ckpt_path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if any(row.get(f) is None for f in _CKPT_FIELDS):
                n_dropped += 1
                continue
            predictions[row["read_id"]] = {
                "pred_label": int(row["pred_label"]),
                "top3_labels": json.loads(row["top3_labels"]),
                "gate_prob": float(row["gate_prob"]),
                "k25_in_candidates": row["k25_in_candidates"] == "1",
                "pred_parent": int(row["pred_parent"]),
                "candidate_count": int(row["candidate_count"]),
            }
    if n_dropped:
        print(f"[resume] {ckpt_path.name}: dropped {n_dropped} truncated/incomplete checkpoint row(s)"
              f" (likely cut off by a mid-write kill) -- those reads will be recomputed", flush=True)
    return predictions


def _load_checkpointed_ids(ckpt_path: Path) -> set[str]:
    """Lightweight twin of _load_batch_checkpoint that only reads the read_id column -- no
    json.loads/int/float parsing per row, no per-row dict allocation. Used for the skip-check
    during the loop below, where only membership ("has this read already been scored") matters,
    not the full record. On a large rank (tens of millions of rows) this is both faster to build
    and, more importantly, far smaller to hold for the whole multi-hour duration of the loop than
    a dict of full records would be (see pitfalls_and_fixes.md 8.14: this is what OOM'd s011_2's
    bacteria genus rank at ~300GB RSS after ~6h)."""
    ids: set[str] = set()
    if not ckpt_path.exists():
        return ids
    with ckpt_path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rid = row.get("read_id")
            if rid is not None and all(row.get(f) is not None for f in _CKPT_FIELDS):
                ids.add(rid)
    return ids


def infer_split_vec_resumable(model, loader, device, amp: bool, ckpt_path: Path,
                              ckpt_every_n_batches: int = 10) -> dict[str, dict]:
    """Bit-identical twin of evaluate_step54_multicand.infer_split_vec, with one addition: each
    batch's output is appended to ckpt_path and the file is flushed to the OS every
    ckpt_every_n_batches batches (not every batch -- rank-level resume alone loses too much, but
    flushing every single batch is 3e4+ write syscalls on the big genus/species ranks; batching the
    flush cuts that ~N-fold, and a kill between flushes only loses the <=N buffered batches, which are
    recomputed on resume, see pitfalls_and_fixes.md 8.14). On entry, any batch whose read_ids are ALL
    already present in ckpt_path is skipped -- the model is
    never called for reads that were already scored in an earlier, interrupted run. A batch that is
    only partially covered is recomputed whole (batches are small; recomputing a few already-known
    reads is cheap, and it keeps this function a straight read of the original loop instead of a
    tensor-slicing exercise that could itself introduce a subtle divergence).

    Memory note: the skip-check below only needs read_id membership, so it's tracked in a plain
    set (checkpointed_ids) for the entire loop -- NOT the full per-read record dict, which would
    otherwise grow unbounded with the rank's size for the whole multi-hour duration of the loop
    (see pitfalls_and_fixes.md 8.14). The full dict (needed by main()'s final_pred pass over every
    read) is only ever materialized once, in one shot, after the loop finishes and every read in
    this rank is guaranteed to be on disk -- so the "hold everything in memory" cost is paid once,
    briefly, instead of accumulating for hours.

    Not merged into evaluate_step54_multicand.py itself: that module is shared with the
    training/evaluation code path, and this checkpoint behavior is deployment-only."""
    checkpointed_ids = _load_checkpointed_ids(ckpt_path)
    n_preloaded = len(checkpointed_ids)
    if n_preloaded:
        print(f"[resume] {ckpt_path.name}: {n_preloaded:,} reads already checkpointed", flush=True)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not ckpt_path.exists() or ckpt_path.stat().st_size == 0
    # Large buffer so writerow() doesn't flush to the OS every ~8 KB (the default) mid-batch -- durability
    # is instead driven explicitly every ckpt_every_n_batches batches below, minimizing write syscalls.
    ckpt_fh = open(ckpt_path, "a", newline="", buffering=4 * 1024 * 1024)
    ckpt_writer = csv.writer(ckpt_fh, delimiter="\t", lineterminator="\n")
    if ckpt_every_n_batches < 1:
        ckpt_every_n_batches = 1
    if is_new:
        ckpt_writer.writerow(_CKPT_FIELDS)

    model.eval()
    n_fresh_batches = n_skipped_batches = 0
    with torch.inference_mode():
        for batch in loader:
            read_ids = [str(r) for r in batch["read_ids"]]
            if read_ids and all(rid in checkpointed_ids for rid in read_ids):
                n_skipped_batches += 1
                continue

            n_fresh_batches += 1
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                scores, gate_logit = model(batch)
            gate_prob = torch.sigmoid(gate_logit)
            top_idx = scores.argmax(dim=1)
            top_labels = batch["candidate_labels"].gather(1, top_idx.view(-1, 1)).squeeze(1)
            topk_idx = torch.topk(scores, k=min(3, scores.size(1)), dim=1).indices
            topk_labels = batch["candidate_labels"].gather(1, topk_idx)
            cand_cpu = batch["candidate_labels"].tolist()
            top_cpu = top_labels.tolist()
            topk_cpu = topk_labels.tolist()
            gate_cpu = gate_prob.tolist()
            k25_cpu = batch["k25_pred_label"].tolist()
            parent_cpu = batch["pred_parent"].tolist()
            for i, read_id in enumerate(read_ids):
                labels = [int(v) for v in cand_cpu[i] if int(v) >= 0]
                rec = {
                    "pred_label": int(top_cpu[i]),
                    "top3_labels": [int(v) for v in topk_cpu[i] if int(v) >= 0],
                    "gate_prob": float(gate_cpu[i]),
                    "k25_in_candidates": int(k25_cpu[i]) in labels,
                    "pred_parent": int(parent_cpu[i]),
                    "candidate_count": len(labels),
                }
                checkpointed_ids.add(read_id)
                ckpt_writer.writerow([read_id, rec["pred_label"], json.dumps(rec["top3_labels"]),
                                       rec["gate_prob"], int(rec["k25_in_candidates"]),
                                       rec["pred_parent"], rec["candidate_count"]])
            # Durability checkpoint every N fresh batches. A kill between flushes loses only the buffered
            # (<=N) batches: those reads were add()ed to the in-memory set but never reached disk, so on
            # resume _load_checkpointed_ids (which reads the DISK file) treats them as not-done and they
            # are recomputed -- <=N*batch reads, seconds of GPU. No effect on the final output.
            if n_fresh_batches % ckpt_every_n_batches == 0:
                ckpt_fh.flush()
    ckpt_fh.flush()
    ckpt_fh.close()
    print(f"[resume] {ckpt_path.name}: {n_fresh_batches} batches computed fresh, "
          f"{n_skipped_batches} skipped (already checkpointed)", flush=True)
    # Rank is now fully done -- every read's record is safely on disk in ckpt_path. This is the
    # one and only point where the full dict is materialized (needed by main()'s final_pred pass
    # over every read in this rank); see the memory note in this function's docstring.
    return _load_batch_checkpoint(ckpt_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--assets-root", required=True,
                    help="absolute path to the data bundle; substituted for {ASSETS} in the config")
    ap.add_argument("--task-group", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--masks-phylum", required=True)
    ap.add_argument("--masks-genus", required=True)
    ap.add_argument("--masks-species", required=True)
    ap.add_argument("--k25-predictions", default="",
                    help="OPTIONAL k25 predictions tsv. If given, use it as the k25 top-1 (exact benchmark "
                         "reproduction). If omitted, derive top-1 from the masks (pipeline-native default).")
    ap.add_argument("--fixed-threshold", type=float, default=0.5,
                    help="Fixed gate threshold applied to every read (deployable default τ=0.5).")
    ap.add_argument("--epochs", default="8,8,8", help="phylum,genus,species epoch numbers")
    ap.add_argument("--species-output-dir", default="",
                    help="Override output_dir for the SPECIES checkpoint (virusb256 lineage).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-every-n-batches", type=int, default=10,
                    help="Flush the batch checkpoint to disk every N DL batches (default 10). Larger = "
                         "fewer write syscalls but up to N batches recomputed if the job is killed "
                         "between flushes. Does NOT change the output, only checkpoint durability cadence.")
    ap.add_argument("--eval-batch-size", type=int, default=0,
                    help="Override the per-rank eval_batch_size from the config (0 = use config, default "
                         "1024). Tune to GPU VRAM: 1024-2048 fit a 20 GB MIG slice (verified on A100/H100), "
                         "lower it for <16 GB GPUs. Batch size does NOT change output (no BatchNorm).")
    args = ap.parse_args()
    _ev = [int(v) for v in args.epochs.split(",") if v != ""]
    epochs_map = dict(zip(RANKS, _ev)) if len(_ev) == 3 else {r: None for r in RANKS}

    # The shipped configs use {PKG} (this package) and {ASSETS} (the data bundle) instead of absolute
    # paths, so the release runs wherever it is unpacked. Substituting on load keeps the config a plain
    # JSON file with no import-time magic.
    assets = Path(args.assets_root).expanduser().resolve()
    labels = Path(os.environ.get("MBC_LABEL_SPACE", str(assets.parent / "config" / "label_space")))
    config = json.loads(
        Path(args.config).read_text()
        .replace("{PKG}", PKG.as_posix()).replace("{ASSETS}", assets.as_posix())
        .replace("{LABELS}", labels.expanduser().resolve().as_posix())
    )
    training = config["training"]
    set_seed(int(config.get("seed", 17)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = bool(training.get("amp", True))
    tokenizer = load_local_tokenizer(Path(config["inputs"]["model_dir"]))

    reads = list(read_reads(Path(args.reads)))
    print(f"[predict] {len(reads):,} reads", flush=True)

    # Masks are loaded LAZILY, one rank at a time, inside the loop below (B2 in V2_UPDATE_PLAN §4 / narval
    # 8.15): a phylum+genus+species mask set can be ~7+14+31 GB, holding all three for the whole run
    # wastes tens of GB, and a rank skipped via its .done marker would load its mask for nothing. Only the
    # small rank->path map and (optionally) the k25 predictions file are set up here; each rank's mask is
    # read right before that rank runs and freed right after (see `del mask_rank` at the end of the loop).
    mask_path_by_rank = {"phylum": Path(args.masks_phylum),
                         "genus": Path(args.masks_genus),
                         "species": Path(args.masks_species)}
    # k25 top-1 default: derived per-rank from that rank's mask (mask_top1) inside the loop -- pipeline-
    # native, no extra file (on ~0.006% of reads it differs from the offline 25-mer predictions, which
    # break ties with priors; that does NOT meaningfully change classification). Pass --k25-predictions
    # ONLY to reproduce the offline benchmark bit-for-bit; it is loaded once here (it is NOT a per-rank mask).
    k25_pred = load_k25_predictions(Path(args.k25_predictions)) if args.k25_predictions else None

    tasks = {t["target_rank"]: (i, t) for i, t in enumerate(config["tasks"])}
    metadata = Path(config["inputs"]["k25_metadata_tsv"])
    train_truth = Path(config["inputs"]["train_truth_tsv"])

    out_path = Path(args.out)

    def rank_ckpt_paths(rank: str) -> tuple[Path, Path, Path]:
        rank_tsv = out_path.with_suffix(f".checkpoint.{rank}.tsv")
        rank_done = out_path.with_suffix(f".checkpoint.{rank}.done")
        batch_tsv = out_path.with_suffix(f".checkpoint.{rank}.dlbatches.tsv")
        return rank_tsv, rank_done, batch_tsv

    route_parent_by_read: dict[str, int] = {}   # filled by each rank for the next one
    timing: list[dict] = []
    t_run = time.perf_counter()

    for rank in RANKS:                          # phylum -> genus -> species; order is the chain
        rank_tsv, rank_done, batch_tsv = rank_ckpt_paths(rank)

        if rank_done.exists():
            # Whole rank already finished in an earlier (interrupted) invocation -- skip the dataset
            # build, model load, and forward pass entirely; just reload what it produced.
            route_parent_by_read = {}
            n_loaded = 0
            with rank_tsv.open(newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    route_parent_by_read[row["read_id"]] = int(row["label"])
                    n_loaded += 1
            print(f"[resume] rank {rank} already complete ({n_loaded:,} reads) -- "
                  f"skipping dataset/model/forward entirely", flush=True)
            timing.append({"rank": rank, "rows": n_loaded, "single_candidate": None,
                           "scored_by_model": None, "dataset_s": 0.0, "model_load_s": 0.0,
                           "forward_s": 0.0, "reads_per_s": None, "resumed_from_checkpoint": True})
            continue

        task_index, task = tasks[rank]
        label_to_parent = load_label_parent_map(train_truth, rank)
        num_labels = load_num_labels(metadata, rank, [train_truth])
        thresholds = {"0": float(args.fixed_threshold)}
        mode = str(task.get("calibration_mode", "domain_global"))

        # Lazy per-rank mask load (freed at the end of this iteration; a .done-skipped rank never reaches
        # here, so its mask is never read -- that is the whole point of B2 / narval 8.15).
        mask_rank = load_mask_records(mask_path_by_rank[rank])
        k25_top1_rank = ({rid: int(k25_pred.get(rid, {}).get(rank, -1)) for rid, _ in reads}
                         if k25_pred is not None else mask_top1(mask_rank))

        t0 = time.perf_counter()
        dataset = PredictDataset(reads, mask_rank, k25_top1_rank, task, label_to_parent, route_parent_by_read)
        t_data = time.perf_counter() - t0
        singles = sum(1 for c in dataset.candidate_labels if len(c) == 1)
        print(f"[predict] {rank}: {len(dataset):,} rows "
              f"(no candidates: {dataset.skipped_no_candidates:,}, missing mask: {dataset.missing_mask:,}, "
              f"single-candidate: {singles:,} = {100*singles/max(len(dataset),1):.1f}%)", flush=True)

        t0 = time.perf_counter()
        odir = args.species_output_dir if (rank == "species" and args.species_output_dir) else None
        model, checkpoint = load_model_for_task(config, task, task_index, num_labels, device,
                                                epoch=epochs_map.get(rank), output_dir_override=odir)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_load = time.perf_counter() - t0

        # ALL THREE RANKS ALWAYS RUN. The only thing filtered is WHICH READS enter the classifier at
        # this rank -- the single-candidate ones cannot change their own outcome (see undecidable()).
        decidable = undecidable(dataset)
        t0 = time.perf_counter()
        rec_by_read = infer_split_vec_resumable(
            model, make_loader(dataset, decidable, tokenizer, task, training,
                               batch_size_override=args.eval_batch_size),
            device, amp, batch_tsv, ckpt_every_n_batches=args.ckpt_every_n_batches)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0
        del model
        torch.cuda.empty_cache()

        rate = len(dataset) / t_fwd if t_fwd else 0
        timing.append({"rank": rank, "rows": len(dataset), "single_candidate": singles,
                       "scored_by_model": len(decidable),
                       "dataset_s": round(t_data, 2), "model_load_s": round(t_load, 2),
                       "forward_s": round(t_fwd, 2), "reads_per_s": round(rate, 1),
                       "resumed_from_checkpoint": False})
        print(f"[time]    {rank}: dataset {t_data:6.2f}s | model load {t_load:6.2f}s | "
              f"forward {t_fwd:7.2f}s over {len(decidable):,} decidable rows "
              f"({100*len(decidable)/max(len(dataset),1):.1f}%) -> {rate:8.1f} reads/s", flush=True)

        # Rank fully finished (every read that has a mask+candidates at this rank has a final call) --
        # write the rank-level checkpoint + .done marker, then keep only route_parent_by_read (a
        # read_id->label map, far smaller than the full label+gate_prob dict) in memory going forward.
        # gate_prob exists only when the DL forward ran (multi-candidate). rec is None for
        # single-candidate/forced reads (DL skipped) -> report NA, not a fake 1.0 (see SKILL.md
        # 'Output semantics'). A value below the threshold means DL ran but fell back to k25.
        route_parent_by_read = {}
        with open(rank_tsv, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["read_id", "label", "gate_prob"])
            for i, read_id in enumerate(dataset.read_ids):
                base = {
                    "k25_pred_label": dataset.k25_pred_label[i],
                    "domain_id": DEPLOY_STRATUM,
                    "k25_pred_parent": dataset.pred_parent[i],
                }
                threshold = threshold_for_read(mode, thresholds, base)
                rec = rec_by_read.get(read_id)
                pred, override, _ = final_pred(base, rec, threshold)
                gate_prob = round(float(rec["gate_prob"]), 4) if rec is not None else "NA"
                w.writerow([read_id, int(pred), gate_prob])
                route_parent_by_read[read_id] = int(pred)
        rank_done.touch()
        del rec_by_read  # already on disk in rank_tsv; no need to keep the decidable-only dict around
        # Free this rank's mask (and the dataset that references it) BEFORE the next rank loads its own,
        # so at most one rank's mask is resident at a time instead of all three (B2 / narval 8.15).
        del dataset, mask_rank, k25_top1_rank

    with open(args.out, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        # User-facing output: the 3-level prediction + its per-level gate confidence. No k25/override
        # columns (those are internal DL-vs-k25 audit detail). pred = -1 means no call at that rank
        # (no candidate / no evidence); gate_prob = NA means the DL gate didn't run (single-candidate,
        # forced) -- NOT the same as a low gate_prob (DL ran but wasn't confident).
        w.writerow(["read_id"] + [f"pred_{r}" for r in RANKS] + [f"gate_prob_{r}" for r in RANKS])
        # Rebuilt from the three rank checkpoints on disk (one at a time) rather than three full
        # in-memory dicts held for the whole run -- see the module docstring's memory note.
        per_rank_final = {}
        for rank in RANKS:
            rank_tsv, _, _ = rank_ckpt_paths(rank)
            d = {}
            with rank_tsv.open(newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    d[row["read_id"]] = {"label": int(row["label"]), "gate_prob": row["gate_prob"]}
            per_rank_final[rank] = d
        for read_id, _seq in reads:
            row = [read_id]
            for rank in RANKS:
                row.append(per_rank_final[rank].get(read_id, {}).get("label", -1))
            for rank in RANKS:
                gp = per_rank_final[rank].get(read_id, {}).get("gate_prob", "NA")
                row.append("NA" if gp == "NA" else float(gp))
            w.writerow(row)
    print(f"[predict] wrote {args.out}", flush=True)

    total = time.perf_counter() - t_run
    load_total = sum(t["model_load_s"] for t in timing)
    fwd_total = sum(t["forward_s"] for t in timing)
    print(f"\n[time] === DL total {total:.1f}s for {len(reads):,} reads ===")
    print(f"[time]   model load : {load_total:7.1f}s ({100*load_total/total:4.1f}%)  <- fixed cost, 3 models")
    print(f"[time]   forward    : {fwd_total:7.1f}s ({100*fwd_total/total:4.1f}%)  <- scales with reads")
    print(f"[time]   end-to-end : {len(reads)/total:8.1f} reads/s")
    Path(args.out).with_suffix(".timing.json").write_text(json.dumps(
        {"reads": len(reads), "total_s": round(total, 1), "per_rank": timing}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
