"""Cross-modal VAE and inference functions from the frozen microbe-to-host model."""

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, latent, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.z_mean = nn.Linear(hidden, latent)
        self.z_logvar = nn.Linear(hidden, latent)

    def forward(self, x):
        h = self.net(x)
        return self.z_mean(h), self.z_logvar(h)


class LNDecoder(nn.Module):
    def __init__(self, latent, out_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def standardize_by_train(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-score every feature by TRAIN mean/std (step5E standardize_train_valid_test, L110-114)."""
    mean = X[train_idx].mean(axis=0, keepdims=True)
    std = X[train_idx].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32), mean.ravel(), std.ravel()


class CrossModalVAE_LN_nocov(nn.Module):
    """encoder(x_m) -> mu/logvar -> z; host_decoder(z); microbe_decoder(z).  (v2.2 Arm B minus [c].)"""

    def __init__(self, microbe_dim, host_dim, hidden, latent, dropout):
        super().__init__()
        self.encoder = LNEncoder(microbe_dim, hidden, latent, dropout)
        self.host_decoder = LNDecoder(latent, host_dim, hidden, dropout)
        self.microbe_decoder = LNDecoder(latent, microbe_dim, hidden, dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x_m):
        mu, logvar = self.encoder(x_m)
        z = self.reparameterize(mu, logvar)
        return self.host_decoder(z), self.microbe_decoder(z), mu, logvar

    def predict_host(self, x_m):
        """Deterministic mu path (CV1:185-188 without covariates)."""
        mu, _ = self.encoder(x_m)
        return self.host_decoder(mu)


def train_fixed_latent(Z: np.ndarray, Yt: np.ndarray, s: dict) -> dict:
    """= vb_core.train_fixed with latent = s['latent'] (no min with n_in).  Everything else identical: seed
    20260819 (torch + numpy) at the start of every training, batch 128 shuffle drop_last, Adam lr 1e-3, fixed epochs,
    loss = host_w * MSE(host) + microbe_w * MSE(microbe, all input columns) + kl_w * KL; CPU float32."""
    seed = int(s["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_tr = Z.shape[0]
    bs = min(int(s["batch"]), max(2, n_tr))
    drop_last = n_tr > bs
    loader = DataLoader(TensorDataset(torch.from_numpy(Z).float(), torch.from_numpy(Yt).float()),
                        batch_size=bs, shuffle=True, drop_last=drop_last)
    n_in, G = Z.shape[1], Yt.shape[1]
    latent = int(s["latent"])                                      # the only change vs vb_core.train_fixed
    net = CrossModalVAE_LN_nocov(n_in, G, hidden=int(s["hidden"]), latent=latent, dropout=float(s["dropout"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(s["lr"]))
    mse = nn.MSELoss()
    n_nonfinite = 0
    trace = []
    t0 = time.time()
    for ep in range(1, int(s["epochs"]) + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            yhat, xhat, mu, logvar = net(xb)
            l_h = mse(yhat, yb)
            l_m = mse(xhat, xb)
            l_k = kl_divergence(mu, logvar)
            loss = float(s["host_w"]) * l_h + float(s["microbe_w"]) * l_m + float(s["kl_w"]) * l_k
            if not torch.isfinite(loss):
                n_nonfinite += 1
            loss.backward()
            opt.step()
            trace.append(torch.stack([loss.detach(), l_h.detach(), l_m.detach(), l_k.detach()]).clone())
    net.eval()
    return dict(net=net, trace=torch.stack(trace), n_nonfinite=n_nonfinite, steps_per_epoch=len(loader),
                drop_last=drop_last, batch=bs, latent=latent, seconds=time.time() - t0)


def predict64(net: nn.Module, Z: np.ndarray) -> np.ndarray:
    """mu-path host prediction in float64 (a float64 copy of the trained float32 model; = hv_vae.PermEngine)."""
    nd = copy.deepcopy(net).double().eval()
    with torch.no_grad():
        return nd.predict_host(torch.from_numpy(np.asarray(Z, np.float32).astype(np.float64))).numpy()


def insample_L(net: nn.Module, Z: np.ndarray, Yt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene in-sample host reconstruction error L(h) = sum over the 392 cells of (prediction - target)^2."""
    P = predict64(net, Z)
    L = ((P - np.asarray(Yt, np.float32).astype(np.float64)) ** 2).sum(0)
    return L, P


def counterfactual(net: nn.Module, Z: np.ndarray, col: int, delta: float = 1.0) -> np.ndarray:
    """Per cell and gene: prediction with input column `col` + delta (z units = + delta SD), all other inputs
    unchanged, minus the unchanged prediction (float64, mu path).  Returns the (n, G) matrix."""
    Zp = np.asarray(Z, np.float32).astype(np.float64).copy()
    Zp[:, col] += delta
    nd = copy.deepcopy(net).double().eval()
    with torch.no_grad():
        p1 = nd.predict_host(torch.from_numpy(Zp)).numpy()
        p0 = nd.predict_host(torch.from_numpy(np.asarray(Z, np.float32).astype(np.float64))).numpy()
    return p1 - p0
