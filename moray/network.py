"""Microbe-to-host gene modelling with a cross-modal VAE."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, identity, issparse
from threadpoolctl import threadpool_limits

from . import io
from ._permfit import bh_fdr_1d
from ._vae import counterfactual, insample_L, standardize_by_train, train_fixed_latent

SETTINGS = {"hidden": 128, "latent": 10, "dropout": 0.1, "batch": 128, "epochs": 18,
            "lr": 1e-3, "host_w": 1.0, "microbe_w": 1.0, "kl_w": 1e-3, "seed": 20260819}


def dense(matrix):
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def standardize(values):
    values = np.asarray(values, np.float64)
    if np.any(values.max(axis=0) == values.min(axis=0)):
        raise ValueError("An interactome model input or target is constant across analysis units")
    intercept, _ = np.linalg.qr(np.ones((len(values), 1)), mode="reduced")
    residual = values - intercept @ (intercept.T @ values)
    z, _, _ = standardize_by_train(residual, np.arange(len(values)))
    return np.clip(z, -10.0, 10.0).astype(np.float32)


def microbe_inputs(gene_counts, load):
    relative = np.divide(gene_counts, load[:, None], out=np.zeros_like(gene_counts, dtype=float),
                         where=load[:, None] > 0)
    return np.column_stack((np.log1p(relative * 1e4), np.log1p(load)))


def neighbourhoods(host, cell_ids, args):
    import anndata as ad
    import scanpy as sc

    from .host import host_features

    genes = io.gene_ids(args.host_features) if args.host_features else host_features(host, 2000).var_names
    counts = dense(host[cell_ids, genes].X).astype(np.float64)
    n = len(counts)
    if args.neighbours >= n:
        raise ValueError("--neighbours must be less than the number of analysis cells")
    components = min(50, n - 1, counts.shape[1] - 1)
    if components < 1:
        raise ValueError("Neighbourhoods require at least two host features")
    data = ad.AnnData(X=counts)
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.scale(data, max_value=10)
    sc.pp.pca(data, n_comps=components, svd_solver="arpack", random_state=0)
    pcs = np.asarray(data.obsm["X_pca"], dtype=np.float64)
    indices = np.arange(n)
    neighbours = np.empty((n, args.neighbours + 1), dtype=int)
    for i in range(n):
        others = indices[indices != i]
        d2 = np.square(pcs[others] - pcs[i]).sum(axis=1)
        neighbours[i, 0] = i
        neighbours[i, 1:] = others[np.lexsort((others, d2))[:args.neighbours]]
    rows = np.repeat(indices, args.neighbours + 1)
    return csr_matrix((np.ones(len(rows)), (rows, neighbours.ravel())), shape=(n, n)), cell_ids


def group_units(host, cell_ids, args):
    if not args.group_columns:
        raise ValueError("--group-columns is required for --mode group")
    columns = [io.groups(host[cell_ids], col) for col in args.group_columns]
    labels = list(zip(*columns))
    counts = pd.Series(labels).value_counts().to_dict()
    keys = sorted({label for label in labels if counts[label] >= args.min_cells_per_unit})
    lookup = {key: i for i, key in enumerate(keys)}
    pairs = [(lookup[key], i) for i, key in enumerate(labels) if key in lookup]
    if len(keys) < 2:
        raise ValueError("Fewer than two aggregates pass --min-cells-per-unit")
    rows, cols = zip(*pairs)
    matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), len(labels)))
    return matrix, [f"group_{i + 1}" for i in range(len(keys))]


def load_inputs(args):
    host = io.raw_host(args.host, args.counts_layer)
    host_genes = io.gene_ids(args.host_genes)
    absent = set(host_genes) - set(host.var_names)
    if absent:
        raise ValueError(f"{len(absent)} selected host genes are missing from host AnnData")

    microbial = io.frame(args.microbial_counts, ("cell_id", "gene_id", "count"))
    coding = pd.Index(io.gene_ids(args.coding_genes))
    coding_counts = io.sparse_counts(microbial, host.obs_names, coding, "gene_id")
    any_gene = io.sparse_counts(microbial, host.obs_names,
                                pd.Index(microbial["gene_id"].unique()), "gene_id")
    use = np.asarray(any_gene.sum(axis=1)).ravel() > 0
    if args.positive_cells:
        positives = {row["cell_id"] for row in io.read_rows(args.positive_cells, ("cell_id",))}
        use &= host.obs_names.isin(positives)
    cells = host.obs_names[use]
    if len(cells) < 2:
        raise ValueError("Fewer than two host cells have matched microbial gene counts")
    if args.mode == "group":
        grouping, unit_ids = group_units(host, cells, args)
    elif args.mode == "knn":
        grouping, unit_ids = neighbourhoods(host, cells, args)
    else:
        grouping, unit_ids = identity(len(cells), format="csr"), cells

    raw_host = dense(host[cells, host_genes].X)
    summed_host = grouping @ raw_host
    host_mask = (summed_host > 0).sum(axis=0) >= args.min_detected_units
    host_genes = np.asarray(host_genes)[host_mask]
    if not len(host_genes):
        raise ValueError("No selected host genes meet --min-detected-units")
    if args.mode == "single":
        import scanpy as sc

        normalized = host.copy()
        normalized.X = normalized.X.astype(np.float32)
        sc.pp.filter_genes(normalized, min_cells=3)
        sc.pp.normalize_total(normalized, target_sum=1e4)
        sc.pp.log1p(normalized)
        missing = set(host_genes) - set(normalized.var_names)
        if missing:
            raise ValueError("Some selected targets are missing from the normalized host matrix")
        host_values = dense(normalized[cells, host_genes].X)
    else:
        library = grouping @ np.asarray(host[cells].X.sum(axis=1)).ravel()
        if (library <= 0).any():
            raise ValueError("A retained analysis unit has an empty host library")
        host_values = np.log1p(summed_host[:, host_mask] * 1e4 / library[:, None])

    coding_counts = coding_counts[use]
    single_load = np.asarray(coding_counts.sum(axis=1)).ravel()
    aggregated = grouping @ coding_counts
    keep = np.asarray((aggregated > 0).sum(axis=0)).ravel() >= args.min_detected_units
    if not keep.any():
        raise ValueError("No microbial coding genes meet --min-detected-units")
    microbe_genes = coding[keep].to_list() + ["LOAD"]
    if "LOAD" in coding:
        raise ValueError("LOAD is reserved for total coding counts")
    summed_microbe = aggregated[:, keep].toarray()
    if args.mode == "knn":
        # The null permutes single-cell profiles before they are summed on the fixed graph.
        raw = coding_counts[:, keep].toarray()
        load = single_load
    else:
        raw = summed_microbe
        load = grouping @ single_load
    return (raw, load, grouping, dense(host_values), host_genes, microbe_genes,
            cells, unit_ids)


def ablation_statistics(x, y):
    full = train_fixed_latent(x, y, SETTINGS)["net"]
    full_loss, _ = insample_L(full, x, y)
    results = np.empty((x.shape[1], y.shape[1]))
    for j in range(x.shape[1]):
        reduced = x.copy()
        reduced[:, j] = 0.0
        fitted = train_fixed_latent(reduced, y, SETTINGS)["net"]
        loss, _ = insample_L(fitted, reduced, y)
        results[j] = loss - full_loss
    return results, full


def design(raw, load, graph, permutation):
    codes = raw[permutation]
    totals = load[permutation]
    if graph is not None:
        codes = graph @ codes
        totals = graph @ totals
    return standardize(microbe_inputs(codes, totals))


def null_exceedance(raw, load, graph, y, observed, permutation):
    null, _ = ablation_statistics(design(raw, load, graph, permutation), y)
    return null >= observed


_worker_inputs = None


def init_worker(raw, load, graph, y, observed):
    global _worker_inputs
    _worker_inputs = (raw, load, graph, y, observed)
    torch.set_num_threads(1)


def permuted_row(permutation):
    return null_exceedance(*_worker_inputs, permutation)


def infer(args):
    raw, load, grouping, host, host_genes, microbe_genes, cells, units = load_inputs(args)
    graph = grouping if args.mode == "knn" else None
    identity_order = np.arange(len(raw))
    seeds = {"single": (20261008, 2), "knn": (20261009, 0), "group": (20261009, 6101)}
    run_seed, mode_seed = seeds[args.mode]
    rng = np.random.default_rng(np.random.SeedSequence([20260819, run_seed, args.taxid, mode_seed]))
    y = standardize(host)
    x = design(raw, load, graph, identity_order)
    torch.set_num_threads(1)
    inputs = raw, load, graph, y
    with threadpool_limits(limits=1):
        observed, full = ablation_statistics(x, y)
        directions = np.vstack([counterfactual(full, x, j).mean(axis=0) for j in range(x.shape[1])])
        exceedances = np.zeros_like(observed, dtype=np.int64)
        permutations = (rng.permutation(identity_order) for _ in range(args.permutations))
        compare_inputs = (*inputs, observed)
        if args.workers == 1:
            for permutation in permutations:
                exceedances += null_exceedance(*compare_inputs, permutation)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                                     initializer=init_worker,
                                     initargs=compare_inputs) as workers:
                for row in workers.map(permuted_row, permutations):
                    exceedances += row
    p = (1 + exceedances) / (args.permutations + 1)
    q = np.column_stack([bh_fdr_1d(p[:, i]) for i in range(len(host_genes))])
    frame = pd.DataFrame({
        "microbial_gene": np.repeat(microbe_genes, len(host_genes)),
        "host_gene": np.tile(host_genes, len(microbe_genes)),
        "ablation_sse_increase": observed.ravel(), "p": p.ravel(), "q": q.ravel(),
        "predicted_change": directions.ravel(), "direction": np.sign(directions).ravel().astype(int),
    })
    frame["input_type"] = np.where(frame["microbial_gene"] == "LOAD", "coding_load", "gene")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "associations.tsv", sep="\t", index=False)
    frame.loc[frame["q"] < 0.05].to_csv(out / "edges.tsv", sep="\t", index=False)
    row, col = grouping.nonzero()
    io.write_rows(out / "analysis_units.tsv", ("unit_id", "cell_id"),
                  ((units[i], cells[j]) for i, j in zip(row, col)))
