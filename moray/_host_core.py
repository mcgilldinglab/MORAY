"""Host-prediction model and resampling helpers from the frozen analysis."""

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.model_selection import KFold

N_FOLDS = 5
KFOLD_SEED = 0


class MLP(nn.Module):
    def __init__(self, i, o, h=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


def rank_cols(M):
    return np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])


def auroc_cols(Y, P):
    Rp = rank_cols(P)
    npos = Y.sum(0); nneg = Y.shape[0] - npos
    s = (Rp * Y).sum(0)
    den = npos * nneg
    return np.where(den == 0, np.nan, (s - npos * (npos + 1) / 2) / np.where(den == 0, 1, den))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def within_sample_folds(group: np.ndarray, n_folds: int = N_FOLDS, seed: int = KFOLD_SEED):
    """KFold(shuffle) inside each sample -> every fold's train AND validation cover all samples."""
    n = len(group)
    assign = np.full(n, -1, dtype=np.int64)
    for g in pd.unique(pd.Series(group)):
        idx = np.flatnonzero(group == g)
        k = min(n_folds, len(idx))
        if k < 2:
            assign[idx] = 0
            continue
        for f, (_, te) in enumerate(KFold(n_splits=k, shuffle=True, random_state=seed).split(idx)):
            assign[idx[te]] = f % n_folds
    folds = []
    for f in range(n_folds):
        te = np.flatnonzero(assign == f)
        tr = np.flatnonzero(assign != f)
        folds.append((tr, te))
    return folds, assign


class WAUC:
    """Exact weighted Mann-Whitney AUROC over a FIXED score matrix P (N x C).

    Pre-sorts every column once and pre-computes tie-group boundaries, so one bootstrap
    resample (expressed as per-cell integer weights) costs O(N*C) vectorised work instead
    of a fresh rankdata.  Ties are handled exactly (mid-rank), so the w == 1 case is
    numerically identical to storm_baseDL.auroc_cols."""

    def __init__(self, P, Y):
        P = np.ascontiguousarray(P, dtype=np.float64)
        self.N, self.C = P.shape
        self.order = np.argsort(P, axis=0, kind="mergesort")
        Ps = np.take_along_axis(P, self.order, axis=0)
        self.ys = np.take_along_axis(np.ascontiguousarray(Y, dtype=np.float64), self.order, axis=0)
        # tie groups per column: positions of the first / last element of each run of equal values
        newgrp = np.ones_like(Ps, dtype=bool)
        newgrp[1:] = Ps[1:] != Ps[:-1]
        idx = np.arange(self.N)[:, None] * np.ones((1, self.C), dtype=int)
        # first position of the run = last j <= i with newgrp[j]
        self.grp_start = np.maximum.accumulate(np.where(newgrp, idx, 0), axis=0)
        # last position of the run = first j >= i that ENDS a run (i.e. j+1 starts a new one)
        isend = np.empty_like(newgrp)
        isend[:-1] = newgrp[1:]
        isend[-1] = True
        endmark = np.where(isend, idx, self.N - 1)
        self.grp_end = np.minimum.accumulate(endmark[::-1], axis=0)[::-1]
        self.n_tied_cols = int((~newgrp[1:]).any(0).sum())

    def auc(self, w):
        """w: per-cell non-negative weights (N,) -> AUROC per column (C,)."""
        ws = np.asarray(w, dtype=np.float64)[self.order]                   # (N,C)
        pw = ws * self.ys
        nw = ws - pw
        cum = np.cumsum(nw, axis=0)
        cum0 = np.vstack([np.zeros((1, self.C)), cum])                     # cum0[i] = sum of nw[:i]
        before = np.take_along_axis(cum0, self.grp_start, axis=0)          # negatives strictly before run
        inrun = np.take_along_axis(cum0, self.grp_end + 1, axis=0) - before
        U = (pw * (before + 0.5 * inrun)).sum(0)
        den = pw.sum(0) * nw.sum(0)
        return np.where(den > 0, U / np.where(den > 0, den, 1.0), np.nan)
