"""Input matrices and tabular output shared by the analysis steps."""

import csv
from collections import Counter
from pathlib import Path


def read_rows(path, required):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        yield from reader


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def read_mapping(path):
    mapping = {}
    for row in read_rows(path, ("read_id", "cell_id")):
        key = row["read_id"]
        if not key or not row["cell_id"] or key in mapping:
            raise ValueError(f"{path} requires a unique read_id and a nonempty cell_id on every row")
        mapping[key] = row
    return mapping


def count_taxa(args):
    rank_fields = ("phylum", "genus", "species")
    domain_names = {"bacteria": "Bacteria", "fungi": "Fungi", "virus": "Viruses"}
    read_map = read_mapping(args.read_map)
    counts, names = Counter(), {}
    fields = ["read_id", "domain"] + [f"pred_{rank}_taxid" for rank in rank_fields]
    for row in read_rows(args.predictions, fields):
        read = read_map.get(row["read_id"])
        domain = domain_names.get(row["domain"])
        if read is None or domain is None:
            continue
        for rank in rank_fields:
            taxid = int(row[f"pred_{rank}_taxid"] or "-1")
            if taxid > 0:
                key = (rank, domain, taxid)
                names.setdefault(key, row.get(f"pred_{rank}_name") or str(taxid))
                counts[(*key, read["cell_id"])] += 1
    if not counts:
        raise ValueError("No classified reads or fragments matched the supplied read map")
    write_rows(args.out, ("cell_id", "rank", "domain", "taxid", "name", "count"),
               ((cell, rank, domain, taxid, names[rank, domain, taxid], n)
                for (rank, domain, taxid, cell), n in counts.items()))


def frame(path, required):
    import pandas as pd

    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return table


def raw_host(path, layer):
    import anndata as ad
    import numpy as np
    from scipy.sparse import issparse

    host = ad.read_h5ad(path)
    if layer and layer not in host.layers:
        raise ValueError(f"Host AnnData has no layer named {layer!r}")
    if layer:
        host.X = host.layers[layer]
    if not host.obs_names.is_unique or not host.var_names.is_unique:
        raise ValueError("Host AnnData requires unique cell IDs and host gene IDs")
    values = host.X.data if issparse(host.X) else host.X
    if values is None or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Host AnnData must contain finite nonnegative raw counts")
    return host


def groups(host, column):
    if column not in host.obs:
        raise ValueError(f"Host AnnData .obs has no column named {column!r}")
    values = host.obs[column]
    if values.isna().any() or (values.astype(str) == "").any():
        raise ValueError(f"Grouping column {column!r} contains missing values")
    return values.astype(str).to_numpy()


def count_values(table):
    import numpy as np

    values = table["count"].astype(float).to_numpy()
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any()):
        raise ValueError("The count column must contain finite, nonnegative integers")
    return values


def sparse_counts(table, cells, items, item_column):
    from scipy.sparse import csr_matrix

    row = cells.get_indexer(table["cell_id"])
    col = items.get_indexer(table[item_column])
    counts = count_values(table)
    use = (row >= 0) & (col >= 0)
    return csr_matrix((counts[use], (row[use], col[use])), shape=(len(cells), len(items)))


def gene_ids(path):
    genes = list(dict.fromkeys(row["gene_id"] for row in read_rows(path, ("gene_id",))
                               if row["gene_id"]))
    if not genes:
        raise ValueError(f"No gene IDs were found in {path}")
    return genes
