"""Permutation statistics used by the frozen single-target models."""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm
import torch


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def bh_fdr_1d(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment with finite-value filtering."""
    p_values = np.asarray(p_values, dtype=np.float64)
    out = np.full(p_values.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(p_values))
    if valid.size == 0:
        return out
    observed = p_values[valid]
    order = np.argsort(observed, kind="mergesort")
    ranked = observed[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out[valid[order]] = adjusted
    return out


@torch.inference_mode()
def permfit_fold_sufficient_statistics(
    model: torch.nn.Module,
    x_validation: np.ndarray,
    y_validation_report: np.ndarray,
    report_local: np.ndarray,
    gene_indices: np.ndarray,
    n_perm: int,
    feature_batch: int,
    permutation_seed: int,
    verify_fast_path: bool,
    device: torch.device,
    cell_loss,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return cell sums/squared sums of mean-permutation deviance increases.

    deepTL first averages the loss difference across permutations for each held-
    out observation, then estimates importance and its SE across observations.
    We retain the two sufficient statistics required for that calculation rather
    than materialising n_perm x n_cell x n_gene x n_microbe.
    """
    x_tensor = torch.from_numpy(x_validation).to(device)
    y_tensor = torch.from_numpy(y_validation_report).to(device)
    report_tensor = torch.as_tensor(report_local, dtype=torch.long, device=device)

    first = model.net[0]
    last = model.net[3]
    w1 = first.weight
    b1 = first.bias
    w2 = last.weight[report_tensor]
    b2 = last.bias[report_tensor]

    base_pre_activation = x_tensor @ w1.T + b1
    base_logits = torch.relu(base_pre_activation) @ w2.T + b2
    base_loss = cell_loss(base_logits, y_tensor)
    base_probabilities = torch.sigmoid(base_logits).cpu().numpy().astype(np.float32)

    n_genes = len(gene_indices)
    n_outputs = len(report_local)
    cell_sum = np.zeros((n_genes, n_outputs), dtype=np.float64)
    cell_sumsq = np.zeros_like(cell_sum)
    generator = torch.Generator(device=device)
    generator.manual_seed(permutation_seed)
    fast_path_max_abs = 0.0

    for start in range(0, n_genes, feature_batch):
        stop = min(start + feature_batch, n_genes)
        selected_np = gene_indices[start:stop]
        selected = torch.as_tensor(selected_np, dtype=torch.long, device=device)
        batch_size = len(selected_np)

        # Independent random permutations for every repetition and gene.
        random_keys = torch.rand(
            (n_perm, batch_size, x_tensor.shape[0]), generator=generator, device=device
        )
        permutation_indices = torch.argsort(random_keys, dim=2)
        del random_keys

        original_values = x_tensor[:, selected].T.contiguous()  # gene x cell
        repeated_values = original_values.unsqueeze(0).expand(n_perm, -1, -1)
        permuted_values = torch.gather(repeated_values, 2, permutation_indices)
        delta = permuted_values - original_values.unsqueeze(0)

        # Exact update after changing one input feature:
        # z_perm = z_original + (x_perm_j - x_j) * W1[:, j].
        selected_w1 = w1[:, selected].T.contiguous()  # gene x hidden
        permuted_pre_activation = (
            base_pre_activation.unsqueeze(0).unsqueeze(0)
            + delta.unsqueeze(-1) * selected_w1.unsqueeze(0).unsqueeze(2)
        )
        permuted_hidden = torch.relu(permuted_pre_activation)
        permuted_logits = torch.matmul(permuted_hidden, w2.T) + b2

        if verify_fast_path and start == 0:
            brute_x = x_tensor.clone()
            brute_x[:, selected[0]] = permuted_values[0, 0]
            brute_logits = model(brute_x)[:, report_tensor]
            fast_path_max_abs = float(
                torch.max(torch.abs(brute_logits - permuted_logits[0, 0])).cpu()
            )

        # Eq. 15 / deepTL log_lik_diff: permuted binary deviance minus original.
        permuted_loss = cell_loss(permuted_logits, y_tensor[None, None, :, :])
        mean_cell_difference = (permuted_loss - base_loss[None, None, :, :]).mean(dim=0)
        cell_sum[start:stop] = mean_cell_difference.sum(dim=1).cpu().numpy()
        cell_sumsq[start:stop] = mean_cell_difference.square().sum(dim=1).cpu().numpy()

        if start == 0 or stop == n_genes or stop % max(feature_batch * 25, 1) == 0:
            log(f"      genes {stop}/{n_genes}")

        del (
            permutation_indices,
            repeated_values,
            permuted_values,
            delta,
            selected_w1,
            permuted_pre_activation,
            permuted_hidden,
            permuted_logits,
            permuted_loss,
            mean_cell_difference,
        )

    return cell_sum, cell_sumsq, base_probabilities, fast_path_max_abs


def finalise_permfit(
    cell_sum: np.ndarray,
    cell_sumsq: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce deepTL importance, importance_sd and one-sided p-value."""
    importance = cell_sum / n_cells
    centred_ss = np.maximum(cell_sumsq - (cell_sum**2) / n_cells, 0.0)
    sample_variance = centred_ss / max(n_cells - 1, 1)
    importance_se = np.sqrt(sample_variance / n_cells)
    z_score = np.full_like(importance, np.nan)
    positive_se = importance_se > 0
    z_score[positive_se] = importance[positive_se] / importance_se[positive_se]
    zero_se = ~positive_se & np.isfinite(importance)
    z_score[zero_se & (importance > 0)] = np.inf
    z_score[zero_se & (importance <= 0)] = -np.inf
    p_value = norm.sf(z_score)

    # Main gate: BH across all tested human genes, separately for every microbe.
    within_microbe_fdr = np.empty_like(p_value)
    for output in range(p_value.shape[1]):
        within_microbe_fdr[:, output] = bh_fdr_1d(p_value[:, output])
    global_level_fdr = bh_fdr_1d(p_value.ravel()).reshape(p_value.shape)
    return importance, importance_se, z_score, p_value, within_microbe_fdr, global_level_fdr
