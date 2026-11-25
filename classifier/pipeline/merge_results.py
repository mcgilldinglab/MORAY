#!/usr/bin/env python3
"""One row per input read (single-end) or per fragment (paired-end): the routed domain, the chained
call at each rank, and the gate confidence at each rank.

Every input read/fragment appears once, including ones no classifier saw (routed to human/no_call/
ambiguous, or no k25 evidence) -- a real outcome, not hidden.

Columns:
  read_id           read id (single-end) or the shared base id (paired-end, mate suffix stripped)
  domain            L1 call: bacteria | fungi | virus | human | no_call | ambiguous
  status            classified | routed_no_model | no_call | ambiguous | no_evidence
                    (paired) + pair_backoff (mates agree only at a coarser rank) | pair_disagree
  pred_{phylum,genus,species}_taxid / _name   the call as NCBI taxid + scientific name (mapped from the
                                domain's label space), or taxid -1 / name "unclassified" = no call at that
                                rank (no candidate / no evidence / mates disagreed and this rank backed off)
  gate_prob_{phylum,genus,species}   DL gate confidence at that rank, or NA. NA != low: NA means the DL
                                gate did not run (single-candidate, forced); a low VALUE means the DL
                                ran but wasn't confident (fell back to k25). (paired) = max over mates.

Paired-end (--paired-mode):
  lca (default)  one call per fragment at the FINEST rank both mates agree on; finer ranks -> -1.
  strict         species must agree for a species call; otherwise the fragment is left unclassified.
  independent    no reconciliation; each mate is its own row (like single-end).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

RANKS = ("phylum", "genus", "species")          # coarse -> fine
MATE_SUFFIX = re.compile(r"[/._][12]$")
PKG = Path(__file__).resolve().parent.parent    # microbiome_classifier/
LABEL_SPACE = Path(os.environ.get("MBC_LABEL_SPACE", str(PKG / "config" / "label_space")))


def base_id(rid: str) -> str:
    return MATE_SUFFIX.sub("", rid)


def load_label_maps(domains):
    """domain -> rank -> label(str) -> (taxid, name), from config/label_space/<domain>/metadata.tsv."""
    maps: dict[str, dict[str, dict[str, tuple[str, str]]]] = {}
    for d in domains:
        mpath = LABEL_SPACE / d / "metadata.tsv"
        if not mpath.exists():
            continue
        m = {r: {} for r in RANKS}
        with mpath.open(newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                for r in RANKS:
                    lab = row.get(f"{r}_label")
                    if lab not in (None, "", "-1"):
                        m[r][lab] = (row.get(f"{r}_taxid", ""), row.get(f"{r}_name", ""))
        maps[d] = m
    return maps


def to_name(maps, domain, rank, label):
    """label -> (taxid, name); -1/unknown -> ('-1', 'unclassified')."""
    if label in ("-1", "", None):
        return "-1", "unclassified"
    return maps.get(domain, {}).get(rank, {}).get(str(label), ("-1", "unclassified"))


def _gate_max(a, b):
    vals = [float(g) for g in (a, b) if g not in ("NA", "", None)]
    return round(max(vals), 4) if vals else "NA"


def reconcile(r1: dict, r2: dict, mode: str):
    """merge two mates -> (pred{rank}, gate{rank}, status_suffix)."""
    pred, gate = {}, {}
    if mode == "strict":
        s1, s2 = r1["pred_species"], r2["pred_species"]
        if s1 == s2 and s1 not in ("-1", "", None):
            for r in RANKS:
                pred[r] = r1[f"pred_{r}"]; gate[r] = _gate_max(r1[f"gate_prob_{r}"], r2[f"gate_prob_{r}"])
            return pred, gate, "classified"
        for r in RANKS:
            pred[r] = "-1"; gate[r] = "NA"
        return pred, gate, "pair_disagree"
    # lca: finest rank both agree on (agreement is monotone coarse->fine for a consistent taxonomy)
    finest = -1
    for i, r in enumerate(RANKS):
        if r1[f"pred_{r}"] == r2[f"pred_{r}"] and r1[f"pred_{r}"] not in ("-1", "", None):
            finest = i
        else:
            break
    for i, r in enumerate(RANKS):
        if i <= finest:
            pred[r] = r1[f"pred_{r}"]; gate[r] = _gate_max(r1[f"gate_prob_{r}"], r2[f"gate_prob_{r}"])
        else:
            pred[r] = "-1"; gate[r] = "NA"
    status = "classified" if finest == len(RANKS) - 1 else ("pair_backoff" if finest >= 0 else "pair_disagree")
    return pred, gate, status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paired", choices=["auto", "yes", "no"], default="auto",
                    help="auto = detect /1 /2 (._) mate suffixes in read_ids")
    ap.add_argument("--paired-mode", choices=["lca", "strict", "independent"], default="lca")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    call = {}
    with open(args.routing, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            call[row["read_id"]] = row["pred_domain"]
    pred = {}
    for domain in domains:
        path = Path(args.pred_dir) / f"{domain}_predictions.tsv"
        if path.exists():
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    pred[row["read_id"]] = row

    read_ids = [row["read_id"].strip() for row in csv.DictReader(open(args.reads, newline=""), delimiter="\t")]
    paired = args.paired == "yes" or (args.paired == "auto" and sum(bool(MATE_SUFFIX.search(r)) for r in read_ids[:1000]) > len(read_ids[:1000]) * 0.5)
    do_reconcile = paired and args.paired_mode != "independent"

    lmaps = load_label_maps(domains)
    header = (["read_id", "domain", "status"]
              + [c for r in RANKS for c in (f"pred_{r}_taxid", f"pred_{r}_name")]
              + [f"gate_prob_{r}" for r in RANKS])

    n = 0
    counts: dict[str, int] = {}
    NA = (["-1"] * 3, ["NA"] * 3)                  # (pred labels, gates) for un-run reads
    with open(args.out, "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t", lineterminator="\n")
        w.writerow(header)

        if do_reconcile:
            groups: dict[str, list[str]] = {}
            for rid in read_ids:
                groups.setdefault(base_id(rid), []).append(rid)
            units = list(groups.items())
        else:
            units = [(rid, [rid]) for rid in read_ids]

        for key, mates in units:
            n += 1
            domain = call.get(mates[0], "no_call")
            recs = [pred.get(m) for m in mates]
            if domain == "human":
                status = "routed_no_model"; preds, gates = NA
            elif domain in ("no_call", "ambiguous"):
                status = domain; preds, gates = NA
            elif all(r is None for r in recs):
                status = "no_evidence"; preds, gates = NA
            elif do_reconcile and len(mates) == 2 and all(recs):
                p, g, status = reconcile(recs[0], recs[1], args.paired_mode)
                preds, gates = [p[r] for r in RANKS], [g[r] for r in RANKS]
            else:
                rec = next(r for r in recs if r)          # single-end, or a lone mate
                status = "classified"
                preds = [rec[f"pred_{r}"] for r in RANKS]; gates = [rec[f"gate_prob_{r}"] for r in RANKS]
            counts[status] = counts.get(status, 0) + 1
            row = [key, domain, status]
            for r, lab in zip(RANKS, preds):          # label -> taxid + name for this read's domain
                row.extend(to_name(lmaps, domain, r, lab))
            row.extend(gates)
            w.writerow(row)

    unit = "fragments" if do_reconcile else "reads"
    print(f"{unit}: {n:,}  (paired={paired}, mode={args.paired_mode if paired else 'single-end'})")
    for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<16} {value:>12,}  ({100 * value / n:5.2f}%)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
