"""Quantify coding-gene inputs from host-unmapped reads and a focal reference."""

import csv
import gzip
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from . import io


def align(r1, r2, args, directory):
    directory.mkdir()
    bam = directory / "aligned.bam"
    if args.aligner == "star":
        command = ["STAR", "--runMode", "alignReads", "--runThreadN", str(args.threads),
                   "--genomeDir", str(args.reference_index), "--genomeLoad", "NoSharedMemory",
                   "--readFilesIn", str(r1)]
        if r2:
            command.append(str(r2))
        if any(str(read).endswith(".gz") for read in (r1, r2) if read):
            command += ["--readFilesCommand", "zcat", "-f"]
        command += ["--outSAMtype", "BAM", "Unsorted", "--outSAMunmapped", "None",
                    "--outSAMattributes", "NH", "HI", "AS", "nM",
                    "--alignIntronMin", "20", "--alignIntronMax", "5000",
                    "--outFilterMultimapNmax", "50", "--outFileNamePrefix", str(directory) + "/"]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return directory / "Aligned.out.bam"

    command = ["bowtie2", "--local", "-k", "50", "--no-unal", "-p", str(args.threads),
               "-x", str(args.reference_index)]
    command += ["-1", str(r1), "-2", str(r2)] if r2 else ["-U", str(r1)]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as bowtie:
        with subprocess.Popen([sys.executable, "-m", "moray._alignment_tags"],
                              stdin=bowtie.stdout, stdout=subprocess.PIPE) as tagger:
            bowtie.stdout.close()
            subprocess.run(["samtools", "view", "-b", "-o", str(bam), "-"],
                           stdin=tagger.stdout, check=True)
            tagger.stdout.close()
        if tagger.returncode:
            raise subprocess.CalledProcessError(tagger.returncode, tagger.args)
    if bowtie.returncode:
        raise subprocess.CalledProcessError(bowtie.returncode, bowtie.args)
    return bam


def featurecounts(args, bams, directory, paired):
    output = directory / "counts.tsv"
    command = ["featureCounts", "-T", str(args.threads), "-a", str(args.annotation),
               "-F", "SAF", "-o", str(output)]
    if paired:
        command += ["-p", "--countReadPairs"]
    else:
        command += ["-R", "CORE", "--Rpath", str(directory)]
    subprocess.run(command + [str(bam) for bam in bams], check=True)
    return output


def paired_counts(args, positive, directory):
    counts = Counter()
    manifest = Path(args.paired_manifest).resolve()
    reader = io.read_rows(manifest, ("cell_id", "r1", "r2"))
    bams, cells, seen = [], [], set()
    for i, row in enumerate(reader):
        cell = row["cell_id"]
        if not cell or cell in seen:
            raise ValueError("The paired FASTQ manifest requires unique nonempty cell IDs")
        seen.add(cell)
        if positive is not None and cell not in positive:
            continue
        bam = align(manifest.parent / row["r1"], manifest.parent / row["r2"],
                    args, directory / f"align_{i}")
        n_mapped = subprocess.check_output(["samtools", "view", "-c", "-F", "4", str(bam)], text=True)
        if int(n_mapped):
            bams.append(bam)
            cells.append(cell)
    for start in range(0, len(bams), 100):
        working = directory / f"batch_{start}"
        working.mkdir()
        batch = bams[start:start + 100]
        table = featurecounts(args, batch, working, paired=True)
        with table.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader((line for line in stream if not line.startswith("#")), delimiter="\t")
            header = next(reader)
            if header[:6] != ["Geneid", "Chr", "Start", "End", "Strand", "Length"]:
                raise ValueError("Unexpected featureCounts header")
            for row in reader:
                for cell, n in zip(cells[start:start + 100], row[6:], strict=True):
                    if int(n):
                        counts[cell, row[0]] += int(n)
    return counts


def selected_fastq(path, mapping, output):
    opener = gzip.open if str(path).endswith(".gz") else open
    selected = 0
    with opener(path, "rt") as reads, output.open("w", encoding="utf-8") as stream:
        while header := reads.readline():
            sequence, separator, qualities = reads.readline(), reads.readline(), reads.readline()
            if not header.startswith("@") or not separator.startswith("+") or not qualities:
                raise ValueError("Expected a four-line FASTQ record")
            read = header[1:].split()[0]
            if read in mapping:
                stream.writelines((header, sequence, separator, qualities))
                selected += 1
    if not selected:
        raise ValueError("No FASTQ reads match the read map and selected cells")


def umi_counts(core_file, read_map):
    deduplicated = set()
    with core_file.open(encoding="utf-8") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 4 or row[1] != "Assigned" or row[2] != "1":
                continue
            read, gene = row[0], row[3]
            mapped = read_map.get(read)
            if mapped is None:
                continue
            umi = mapped.get("umi") or mapped.get("raw_umi")
            molecule = ("umi", umi) if umi else ("read", read)
            deduplicated.add((mapped["cell_id"], gene, molecule))
    return Counter((cell, gene) for cell, gene, _ in deduplicated)


def quantify(args):
    if args.single_fastq and not args.read_map:
        raise ValueError("Single-end quantification requires --read-map")
    commands = ("STAR" if args.aligner == "star" else "bowtie2", "samtools", "featureCounts")
    missing = [name for name in commands if not shutil.which(name)]
    if missing:
        raise ValueError("Tools missing from PATH: " + ", ".join(missing))
    positive = None
    if args.positive_cells:
        positive = {row["cell_id"] for row in io.read_rows(args.positive_cells, ("cell_id",))}
        if not positive:
            raise ValueError("The positive-cell table is empty")
    with TemporaryDirectory(prefix="moray-quant-") as temp:
        directory = Path(temp)
        if args.paired_manifest:
            counts = paired_counts(args, positive, directory)
        else:
            mapping = io.read_mapping(args.read_map)
            if positive is not None:
                mapping = {read: row for read, row in mapping.items() if row["cell_id"] in positive}
            fastq = directory / "selected.fastq"
            selected_fastq(args.single_fastq, mapping, fastq)
            bam = align(fastq, None, args, directory / "align")
            featurecounts(args, [bam], directory, paired=False)
            counts = umi_counts(directory / (bam.name + ".featureCounts"), mapping)
    if not counts:
        raise ValueError("No unique microbial gene assignments were obtained")
    io.write_rows(args.out, ("cell_id", "gene_id", "count"),
                  ((cell, gene, n) for (cell, gene), n in sorted(counts.items())))
