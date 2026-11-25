#!/usr/bin/env python3
"""Adapter: turn real sequencing data into the reads TSV (read_id, sequence) the pipeline expects.

Handles FASTQ / FASTA (+ .gz), and BAM (via `samtools fastq`). Single-end or paired-end. For paired
input the two mates keep a shared base id with /1 and /2 suffixes, so merge_results.py can reconcile
the pair (see --paired-mode there). Warns if reads are not ~150 bp (the model was trained on 150 bp;
long reads e.g. Smart-seq/full-length are out-of-distribution -- fragment them first).

Host filtering is NOT done here: give it reads you already consider microbial candidates (e.g. reads
that did not map to the host genome).

Usage:
  prepare_reads.py --input reads.fastq.gz --out reads.tsv                 # single-end
  prepare_reads.py --r1 R1.fq.gz --r2 R2.fq.gz --out reads.tsv            # paired-end
  prepare_reads.py --input aln.bam --out reads.tsv [--bam-paired]         # BAM (needs samtools)
"""
from __future__ import annotations

import argparse
import gzip
import re
import statistics
import subprocess
import sys
from pathlib import Path

_MATE = re.compile(r"[/._][12]$")   # strip an EXISTING mate suffix (/1 .2 _1 ...), not a trailing digit


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _is_fasta(path: str) -> bool:
    p = str(path).lower()
    return any(p.endswith(e) for e in (".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fna", ".fna.gz"))


def iter_fastx(path: str):
    """yield (read_id, sequence) from FASTQ or FASTA (+.gz)."""
    with _open(path) as fh:
        if _is_fasta(path):
            rid, seq = None, []
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if rid is not None:
                        yield rid, "".join(seq)
                    rid = line[1:].split()[0]; seq = []
                else:
                    seq.append(line)
            if rid is not None:
                yield rid, "".join(seq)
        else:  # FASTQ
            while True:
                h = fh.readline()
                if not h:
                    break
                s = fh.readline().rstrip("\n"); fh.readline(); fh.readline()
                yield h[1:].split()[0].rstrip("\n"), s


def iter_bam(path: str, paired: bool):
    """stream reads out of a BAM via samtools fastq (-N tags mates with /1 /2)."""
    if subprocess.run(["bash", "-lc", "command -v samtools"], capture_output=True).returncode != 0:
        sys.exit("BAM input needs `samtools` (module load samtools / put it on PATH).")
    cmd = ["samtools", "fastq", "-N"] + (["-0", "/dev/null", "-s", "/dev/null"] if not paired else []) + [path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    fh = proc.stdout
    while True:
        h = fh.readline()
        if not h:
            break
        s = fh.readline().rstrip("\n"); fh.readline(); fh.readline()
        yield h[1:].split()[0].rstrip("\n"), s
    proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="single-end FASTQ/FASTA(.gz) or BAM")
    ap.add_argument("--r1"); ap.add_argument("--r2", help="paired-end FASTQ/FASTA(.gz)")
    ap.add_argument("--bam-paired", action="store_true", help="treat BAM as paired (keep mate /1 /2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lengths, n = [], 0
    with open(a.out, "w", newline="") as out:
        out.write("read_id\tsequence\n")

        def emit(rid, seq, suffix=""):
            nonlocal n
            if not seq:
                return
            base = _MATE.sub("", rid) if suffix else rid
            out.write(f"{base}{suffix}\t{seq.upper()}\n")
            n += 1
            if len(lengths) < 20000:
                lengths.append(len(seq))

        if a.r1 and a.r2:                                   # paired FASTQ/FASTA
            for rid, seq in iter_fastx(a.r1):
                emit(rid, seq, "/1")
            for rid, seq in iter_fastx(a.r2):
                emit(rid, seq, "/2")
        elif a.input and a.input.lower().endswith(".bam"):  # BAM
            for rid, seq in iter_bam(a.input, a.bam_paired):
                # samtools -N already tags mates; keep the id as-is (base_id in merge strips /1 /2)
                emit(rid, seq)
        elif a.input:                                       # single-end FASTQ/FASTA
            for rid, seq in iter_fastx(a.input):
                emit(rid, seq)
        else:
            ap.error("give --input, or --r1 and --r2")

    if lengths:
        med = statistics.median(lengths)
        note = "" if 100 <= med <= 250 else "  <-- NOT ~150 bp: model is 150-bp-trained; long/variable reads are out-of-distribution (fragment first)"
        print(f"wrote {n:,} reads to {a.out}  (median length {med:.0f} bp){note}")
    else:
        print(f"wrote {n:,} reads to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
