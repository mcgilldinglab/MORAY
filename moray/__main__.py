import argparse
import importlib
import subprocess
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def taxon_inputs(parser):
    parser.add_argument("--host", type=Path, required=True, help="QC-filtered host AnnData")
    parser.add_argument("--counts-layer", help="AnnData layer containing raw counts, default X")
    parser.add_argument("--taxon-counts", type=Path, required=True)
    parser.add_argument("--rank", choices=("genus", "species"), required=True)
    parser.add_argument("--group-column", required=True, help="sample, pool or patient column in .obs")
    parser.add_argument("--n-hvg", type=positive, default=2000)
    parser.add_argument("--epochs", type=positive, default=100)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--out", type=Path, required=True, help="output directory")


def build_parser():
    parser = argparse.ArgumentParser(description="MORAY single-cell host–microbe analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    counts = commands.add_parser("count-taxa", help="summarize classifier output by cell")
    counts.add_argument("--predictions", type=Path, required=True)
    counts.add_argument("--read-map", type=Path, required=True, help="read_id and cell_id TSV")
    counts.add_argument("--out", type=Path, required=True, help="taxon count TSV")
    counts.set_defaults(handler=("io", "count_taxa"))

    rank = commands.add_parser("rank-taxa", help="rank taxa by predictability from host expression")
    taxon_inputs(rank)
    rank.add_argument("--domain", choices=("Bacteria", "Fungi", "Viruses"))
    rank.add_argument("--min-positive", type=positive, default=100)
    rank.add_argument("--exclude-taxids", type=Path, help="optional TSV with taxid and optional rank")
    rank.add_argument("--bootstraps", type=positive, default=2000)
    rank.set_defaults(handler=("host", "rank"))

    permfit = commands.add_parser("select-host-genes", help="select host genes for one taxon using PermFIT")
    taxon_inputs(permfit)
    permfit.add_argument("--taxid", required=True)
    permfit.add_argument("--host-features", type=Path, help="host_features.tsv from rank-taxa")
    permfit.add_argument("--permutations", type=positive, default=100)
    permfit.add_argument("--feature-batch", type=positive, default=1)
    permfit.set_defaults(handler=("host", "permfit"))

    quant = commands.add_parser("quantify", help="realign reads and count uniquely assigned genes")
    reads = quant.add_mutually_exclusive_group(required=True)
    reads.add_argument("--single-fastq", type=Path)
    reads.add_argument("--paired-manifest", type=Path, help="TSV with cell_id, r1, r2")
    quant.add_argument("--read-map", type=Path, help="single-end read_id, cell_id and optional UMI columns")
    quant.add_argument("--positive-cells", type=Path, help="positive_cells.tsv from select-host-genes")
    quant.add_argument("--aligner", choices=("bowtie2", "star"), default="bowtie2")
    quant.add_argument("--reference-index", type=Path, required=True, help="Bowtie 2 prefix or STAR index directory")
    quant.add_argument("--annotation", type=Path, required=True, help="focal-reference SAF annotation")
    quant.add_argument("--threads", type=positive, default=4)
    quant.add_argument("--out", type=Path, required=True, help="microbial count TSV")
    quant.set_defaults(handler=("quantify", "quantify"))

    interactome = commands.add_parser("infer-interactome", help="infer host–microbial gene associations")
    interactome.add_argument("--host", type=Path, required=True)
    interactome.add_argument("--counts-layer", help="AnnData layer containing raw counts, default X")
    interactome.add_argument("--microbial-counts", type=Path, required=True)
    interactome.add_argument("--coding-genes", type=Path, required=True, help="TSV of all coding gene_id values")
    interactome.add_argument("--host-genes", type=Path, required=True,
                             help="selected_host_genes.tsv from select-host-genes")
    interactome.add_argument("--positive-cells", type=Path, help="positive_cells.tsv from select-host-genes")
    interactome.add_argument("--taxid", type=positive, required=True,
                             help="taxon ID used to seed the permutation test")
    interactome.add_argument("--mode", choices=("single", "knn", "group"), default="single")
    interactome.add_argument("--host-features", type=Path, help="HVG panel for --mode knn")
    interactome.add_argument("--neighbours", type=positive, default=10)
    interactome.add_argument("--group-columns", nargs="+", help="AnnData .obs grouping columns for --mode group")
    interactome.add_argument("--min-cells-per-unit", type=positive, default=3)
    interactome.add_argument("--min-detected-units", type=positive, default=20)
    interactome.add_argument("--permutations", type=positive, default=5000)
    interactome.add_argument("--workers", type=positive, default=1, help="parallel CPU workers")
    interactome.add_argument("--out", type=Path, required=True, help="output directory")
    interactome.set_defaults(handler=("network", "infer"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    module, name = args.handler
    try:
        getattr(importlib.import_module("." + module, __package__), name)(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"moray {args.command}: {error}\n")


if __name__ == "__main__":
    main()
