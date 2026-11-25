"""Host-to-microbe prediction with taxon ranking and focal-taxon PermFIT."""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.sparse import issparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import io
from ._host_core import MLP, WAUC, auroc_cols, seed_all, within_sample_folds
from ._permfit import bh_fdr_1d, finalise_permfit, permfit_fold_sufficient_statistics


def host_features(host, n_genes):
    host = host.copy()
    host.X = host.X.astype(np.float32)
    sc.pp.filter_genes(host, min_cells=3)
    if not host.n_vars:
        raise ValueError("No host genes were detected in at least three cells")
    sc.pp.normalize_total(host, target_sum=1e4)
    sc.pp.log1p(host)
    sc.pp.highly_variable_genes(host, n_top_genes=min(n_genes, host.n_vars), flavor="seurat")
    return host[:, host.var["highly_variable"]]


def normalized_matrix(host, n_genes, feature_path=None):
    if feature_path is None:
        selected = host_features(host, n_genes)
    else:
        genes = io.gene_ids(feature_path)
        panel = set(genes)
        # Match the pre-HVG normalization denominator used for taxon ranking.
        filtered = host.copy()
        filtered.X = filtered.X.astype(np.float32)
        sc.pp.filter_genes(filtered, min_cells=3)
        absent = panel - set(filtered.var_names)
        if absent:
            raise ValueError(f"{len(absent)} supplied host features are absent after filtering")
        sc.pp.normalize_total(filtered, target_sum=1e4)
        sc.pp.log1p(filtered)
        selected = filtered[:, genes]
    x = selected.X
    return np.ascontiguousarray(x.toarray() if issparse(x) else x, np.float32), selected.var_names


def taxon_matrix(path, host, rank):
    counts = io.frame(path, ("cell_id", "rank", "domain", "taxid", "name", "count"))
    counts = counts.loc[(counts["rank"] == rank) & counts["cell_id"].isin(host.obs_names)].copy()
    if counts.empty:
        raise ValueError(f"No taxon counts at {rank} rank match host cell IDs")
    meta = counts[["taxid", "name", "domain"]].drop_duplicates("taxid").set_index("taxid")
    if counts[["taxid", "domain"]].drop_duplicates()["taxid"].duplicated().any():
        raise ValueError("Each taxid at a rank must map to one domain")
    return io.sparse_counts(counts, host.obs_names, meta.index, "taxid"), meta


def fit_oof_presence(x, y, device, seed, epochs):
    output = np.empty(y.shape, np.float32)
    splits = KFold(n_splits=5, shuffle=True, random_state=0).split(x)
    for fold, (train, val) in enumerate(splits):
        scaler = StandardScaler().fit(x[train])
        train_x = torch.from_numpy(scaler.transform(x[train]).astype(np.float32)).to(device)
        val_x = torch.from_numpy(scaler.transform(x[val]).astype(np.float32)).to(device)
        train_y = torch.from_numpy(y[train]).to(device)
        torch.manual_seed(seed + fold)
        model = MLP(x.shape[1], y.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(train_x), train_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output[val] = torch.sigmoid(model(val_x)).cpu().numpy()
    return output


def cluster_bootstrap(prediction, presence, sample, n_boot):
    weighted_auc = WAUC(prediction, presence)
    clusters = pd.unique(sample)
    membership = (clusters[:, None] == sample[None, :]).astype(float)
    rng = np.random.default_rng(20260922)
    draws = np.empty((n_boot, presence.shape[1]))
    for b in range(n_boot):
        slots = rng.integers(0, len(clusters), len(clusters))
        weights = np.bincount(slots, minlength=len(clusters)).astype(float) @ membership
        draws[b] = weighted_auc.auc(weights)
    p = (1 + ((draws <= 0.5) | ~np.isfinite(draws)).sum(axis=0)) / (n_boot + 1)
    return np.nanpercentile(draws, [2.5, 97.5], axis=0), p


def rank(args):
    host = io.raw_host(args.host, args.counts_layer)
    if host.n_obs < 5:
        raise ValueError("Taxon ranking requires at least five host cells")
    groups = io.groups(host, args.group_column)
    x, features = normalized_matrix(host, args.n_hvg)
    counts, meta = taxon_matrix(args.taxon_counts, host, args.rank)
    positive = np.asarray((counts > 0).sum(axis=0)).ravel()
    use = (positive >= args.min_positive) & (positive < host.n_obs)
    if args.exclude_taxids:
        exclusion = io.frame(args.exclude_taxids, ("taxid",))
        if "rank" in exclusion:
            exclusion = exclusion[exclusion["rank"] == args.rank]
        use &= ~meta.index.isin(exclusion["taxid"])
    qualifying = meta.loc[use]
    domains = sorted(set(qualifying["domain"]) & {"Bacteria", "Fungi", "Viruses"})
    device = torch.device(args.device)
    tables = []
    for index, domain in enumerate(domains, start=1):
        if args.domain and domain != args.domain:
            continue
        mask = use & (meta["domain"].to_numpy() == domain)
        y = (counts[:, mask].toarray() > 0).astype(np.float32)
        prediction = fit_oof_presence(x, y, device, 137 * index, args.epochs)
        ci, p = cluster_bootstrap(prediction, y, groups, args.bootstraps)
        table = meta.loc[mask].reset_index()
        table["rank"] = args.rank
        table["auroc"] = auroc_cols(y, prediction)
        table["auroc_rank"] = table["auroc"].rank(method="min", ascending=False).astype(int)
        table["n_positive"] = positive[mask]
        table["ci_low"], table["ci_high"] = ci
        table["bootstrap_p"] = p
        table["bootstrap_q"] = bh_fdr_1d(p)
        tables.append(table.sort_values("auroc_rank", kind="stable"))
    if not tables:
        raise ValueError("No eligible taxa. Check the requested domain, cell IDs and positivity threshold")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(out / "taxon_ranking.tsv", sep="\t", index=False)
    io.write_rows(out / "host_features.tsv", ("gene_id",), ((gene,) for gene in features))


def nb_loss(out, counts, log_theta):
    mean = torch.exp(torch.clamp(out, -12.0, 13.0))
    size = torch.exp(torch.clamp(log_theta, -6.0, 6.0))
    epsilon = 1e-8
    return -(torch.lgamma(counts + size) - torch.lgamma(size) - torch.lgamma(counts + 1.0)
             + size * (torch.log(size + epsilon) - torch.log(size + mean + epsilon))
             + counts * (torch.log(mean + epsilon) - torch.log(size + mean + epsilon)))


def nb_importance_loss(out, counts, log_theta):
    return nb_loss(out.double(), counts.double(), log_theta.double())


def presence_loss(out, presence):
    return F.softplus(out) - presence * out


def permfit(args):
    host = io.raw_host(args.host, args.counts_layer)
    sample = io.groups(host, args.group_column)
    x, genes = normalized_matrix(host, args.n_hvg, args.host_features)
    counts, meta = taxon_matrix(args.taxon_counts, host, args.rank)
    if args.taxid not in meta.index:
        raise ValueError(f"Taxon {args.taxid} has no matched counts at {args.rank} rank")
    raw = counts[:, meta.index.get_loc(args.taxid)].toarray().ravel()
    present = raw > 0
    if not present.any() or present.all():
        raise ValueError("The focal taxon must have both positive and negative host cells")
    library = np.asarray(host.X.sum(axis=1)).ravel()
    load = np.asarray(counts.sum(axis=1)).ravel()
    design = np.ascontiguousarray(np.column_stack(
        (x, np.log1p(library), np.log1p(np.maximum(0, load - raw)))), dtype=np.float32)
    positive = raw[present]
    fit_count = (positive == 1).mean() < 0.5 and len(np.unique(positive)) >= 20
    y = (raw if fit_count else present).astype(np.float32)
    folds, assignment = within_sample_folds(sample)
    sums = np.zeros((len(genes), 1))
    squares = np.zeros_like(sums)
    oof = np.empty(len(y))
    device = torch.device(args.device)
    for fold, (train, val) in enumerate(folds):
        if not len(val):
            continue
        if not len(train):
            raise ValueError("Sample groups produce an empty training fold")
        scaler = StandardScaler().fit(design[train])
        train_x = torch.from_numpy(scaler.transform(design[train]).astype(np.float32)).to(device)
        val_x = scaler.transform(design[val]).astype(np.float32)
        train_y = torch.from_numpy(y[train]).to(device)
        seed_all(137 + fold)
        model = MLP(design.shape[1], 1).to(device)
        log_theta = torch.zeros(1, device=device, requires_grad=True)
        params = list(model.parameters()) + ([log_theta] if fit_count else [])
        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            optimizer.zero_grad()
            output = model(train_x).squeeze(1)
            loss = (nb_loss(output, train_y, log_theta).mean() if fit_count
                    else F.binary_cross_entropy_with_logits(output, train_y))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(val_x).to(device)).cpu().numpy().ravel().astype(float)
        oof[val] = (np.log1p(np.exp(np.clip(logits, -12.0, 13.0))) if fit_count
                    else 1.0 / (1.0 + np.exp(-logits)))
        loss_function = (partial(nb_importance_loss, log_theta=log_theta.detach()) if fit_count
                         else presence_loss)
        perm_seed = 20260902 + (0 if args.rank == "genus" else 1_000_000) + fold
        s, ss, _, _ = permfit_fold_sufficient_statistics(
            model, val_x, y[val, None], np.array([0]), np.arange(len(genes)),
            args.permutations, args.feature_batch, perm_seed, False, device, loss_function)
        sums += s
        squares += ss
    imp, se, z, p, q, _ = finalise_permfit(sums, squares, len(y))
    result = pd.DataFrame({"gene_id": genes, "importance": imp.ravel(), "se": se.ravel(),
                           "z": z.ravel(), "p": p.ravel(), "q": q.ravel(),
                           "selected": q.ravel() < 0.05})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "gene_importance.tsv", sep="\t", index=False)
    selected = result.loc[result["selected"]].sort_values(
        ["q", "p", "importance"], ascending=[True, True, False], kind="stable")
    selected[["gene_id"]].to_csv(out / "selected_host_genes.tsv", sep="\t", index=False)
    io.write_rows(out / "positive_cells.tsv", ("cell_id",),
                  ((cell,) for cell in host.obs_names[present]))
    pred = pd.DataFrame({"cell_id": host.obs_names, "fold": assignment, "count": raw,
                         "prediction": oof, "target": "count" if fit_count else "presence"})
    pred.to_csv(out / "out_of_fold.tsv", sep="\t", index=False)
