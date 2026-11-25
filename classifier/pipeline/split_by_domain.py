#!/usr/bin/env python3
"""Split reads by the L1 router's call.

The router emits one of: human | bacteria | fungi | virus | no_call | ambiguous.

  no_call    the read hit ZERO domain-unique 25-mers. There is no evidence to classify it.
  ambiguous  two or more domains tied for the most domain-unique hits.
  human      routed, but there is no human DL model, so it stops here.

None of these three are errors and none are dropped. On a real metagenome they can be a large
fraction of the sample, and reporting how large is part of the result. They are carried to the final
output with their status; only bacteria/fungi/virus reads continue to the classifiers.

The per-domain reads TSV carries the label columns the downstream tools expect. On real data the true
labels are unknown, so they are written as -1. Nothing downstream reads them: the K25 query only
echoes them, and the strict-chain predictor never looks at them (proved: the model's forward() never
touches a truth field).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DL_DOMAINS = ("bacteria", "fungi", "virus")
UNKNOWN = "-1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    call = {}
    with open(args.routing, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            call[row["read_id"]] = row["pred_domain"]

    cols = ["read_id", "sequence", "phylum_label", "genus_label", "species_label"]
    writers, handles = {}, {}
    for domain in DL_DOMAINS:
        handles[domain] = open(out / f"{domain}_reads.tsv", "w", newline="")
        writers[domain] = csv.DictWriter(handles[domain], fieldnames=cols, delimiter="\t", lineterminator="\n")
        writers[domain].writeheader()

    counts = Counter()
    with open(args.reads, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            read_id = row["read_id"].strip()
            domain = call.get(read_id, "no_call")
            counts[domain] += 1
            if domain in DL_DOMAINS:
                writers[domain].writerow({
                    "read_id": read_id,
                    "sequence": row["sequence"].strip(),
                    "phylum_label": UNKNOWN,
                    "genus_label": UNKNOWN,
                    "species_label": UNKNOWN,
                })
    for handle in handles.values():
        handle.close()

    total = sum(counts.values())
    summary = {
        "total_reads": total,
        "by_domain": dict(counts),
        "sent_to_classifiers": sum(counts[d] for d in DL_DOMAINS),
        "not_classified": {k: counts[k] for k in ("human", "no_call", "ambiguous") if counts[k]},
    }
    (out / "routing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
