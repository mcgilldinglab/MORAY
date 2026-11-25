# MORAY

MORAY uses host single-cell RNA sequencing data to identify microbial taxa, find taxon-associated host transcriptional signatures and infer signed associations between microbial and host genes.

## Hierarchical microbial classification

The classifier combines canonical 25-mer matches with genomic language model representations to assign host-unmapped reads at the phylum, genus and species levels.

Run from the repository root on Linux with Bash, a C++17 compiler with OpenMP, zlib development headers and the classifier's Python environment. The full v7 classifier, including the bacterial index, requires approximately 500 GB of RAM and a GPU with at least 16 GB of memory.

```bash
python3 -m venv .venv-classifier
source .venv-classifier/bin/activate
python -m pip install -r classifier/requirements.txt
bash classifier/bin/build.sh

export MBC_ASSETS_ROOT=/path/to/v7/assets
export MBC_LABEL_SPACE=/path/to/v7/config/label_space
python classifier/pipeline/prepare_reads.py --input unmapped.fastq.gz --out reads.tsv
python classifier/pipeline/run_pipeline.py --reads reads.tsv --out-dir runs/classifier --local
```

Use `--r1` and `--r2` with `prepare_reads.py` for paired-end reads. `runs/classifier/final_predictions.tsv` contains taxon assignments. Append `--dry-run` to inspect the generated stage scripts before inference. Follow the [classifier asset layout](classifier/ASSETS.md) when configuring the external v7 database, label tables and model weights. Use a separate output directory for each input library.

## Host-to-microbe prediction

Use a separate Python environment for host-to-microbe prediction and microbe-to-host gene modelling.

```bash
python3 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
```

Supply the QC-filtered host counts as AnnData (`host.h5ad`), with cell IDs in `.obs_names`, gene IDs in `.var_names` and a sample, pool or patient field in `.obs`. The commands read raw counts from `.X` by default. Use `--counts-layer` to select a raw-count layer.

If you have classifier output, map its `read_id` values to cell IDs in `read_to_cell.tsv` (`read_id`, `cell_id`). Use the fragment ID for reconciled pairs. Read and cell IDs should be unique across libraries.

```bash
python -m moray count-taxa --predictions runs/classifier/final_predictions.tsv \
  --read-map read_to_cell.tsv --out runs/taxon_counts.tsv
```

You can supply an existing taxon count TSV with columns `cell_id`, `rank`, `domain`, `taxid`, `name` and `count`. Use `Bacteria`, `Fungi` or `Viruses` for `domain` and include all taxa at the selected rank to retain the total-count covariate in the focal-taxon model.

### Rank taxa

```bash
python -m moray rank-taxa --host host.h5ad --taxon-counts runs/taxon_counts.tsv \
  --rank species --group-column sample --out runs/ranking
```

`taxon_ranking.tsv` reports out-of-fold AUROC and the sample-level bootstrap statistics. `host_features.tsv` records the host features used by the model. Use `--rank genus` for genus-level counts. To apply a lineage exclusion list, pass a TSV containing `taxid` and, optionally, `rank` to `--exclude-taxids`.

### Select host genes with PermFIT

Set `TAXID` to a taxon of interest from the ranking and use the same taxonomic rank.

```bash
python -m moray select-host-genes --host host.h5ad --taxon-counts runs/taxon_counts.tsv \
  --rank species --taxid "$TAXID" --group-column sample \
  --host-features runs/ranking/host_features.tsv --out runs/focal
```

The focal model selects presence or read counts as its target based on the count distribution. The outputs include the full PermFIT table, `selected_host_genes.tsv` and `positive_cells.tsv`. For the manuscript permutation batching, use `--feature-batch 8` for STORM-seq, `--feature-batch 1` for COVID-19 and `--feature-batch 4` for NSCLC. Host prediction can also run with `--device cuda`.

## Microbe-to-host gene modelling

### Quantify microbial genes

For quantification, provide a focal-reference index and its SAF gene annotation. Put Bowtie 2 (or STAR), SAMtools and featureCounts on the path. For multiplexed single-end reads, the read-map TSV can also supply corrected `umi` and `raw_umi` values.

```bash
python -m moray quantify --single-fastq unmapped.fastq.gz --read-map read_to_cell.tsv \
  --positive-cells runs/focal/positive_cells.tsv --reference-index /path/to/bowtie2/index \
  --annotation focal.saf --out runs/microbial_counts.tsv
```

For paired reads, use `--paired-manifest` with columns `cell_id`, `r1` and `r2`. Paths in the manifest are relative to the manifest file or absolute. Use `--aligner star` with a STAR index for splice-aware alignment. An existing microbial count TSV (`cell_id`, `gene_id`, `count`) can also be used directly.

### Infer the taxon-specific interactome

A dual-decoder variational autoencoder reconstructs microbial inputs and predicts the expression of selected host genes. Input ablation and permutation testing identify gene pairs, and model predictions determine the direction of each association.

```bash
python -m moray infer-interactome --host host.h5ad --microbial-counts runs/microbial_counts.tsv \
  --coding-genes coding_genes.tsv --host-genes runs/focal/selected_host_genes.tsv \
  --positive-cells runs/focal/positive_cells.tsv --taxid "$TAXID" \
  --mode single --out runs/network
```

`coding_genes.tsv` lists every protein-coding gene of the focal reference in a `gene_id` column. These genes define total coding `LOAD`. Use `--mode knn --host-features runs/ranking/host_features.tsv` for host-expression neighbourhoods or `--mode group --group-columns cell_type patient` for grouped analysis units.

`associations.tsv` contains tested pairs, permutation P and adjusted q values, and model-predicted directions. `edges.tsv` contains pairs passing q < 0.05 and identifies coding-load rows separately. Interactome inference runs 5,000 permutations by default and supports `--workers` for parallel CPU execution.

Run `python -m moray --help` for the command list. The classifier's vendored BBHash header uses the [upstream MIT license](classifier/src/BBHash-LICENSE.txt).
