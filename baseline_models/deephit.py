#!/usr/bin/env python3
"""
DeepHit (Lee et al., 2018): A discrete-time competing risks model.

Implementation follows pycox (Kvamme et al.) which is the reference used by
the DKAJ paper. Key design choices from pycox:
  - Network outputs (batch, K, J) logits (per-risk heads)
  - PMF via pad-end + softmax + drop-last (not prepend-zero)
  - Ranking loss via precomputed pair matrix + matrix CIF diff
  - CIF = PMF cumsum over time bins
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class DeepHitNet(nn.Module):
    """DeepHit network for competing risks.

    Architecture (matching DKAJ's CauseSpecificNet):
        x -> shared MLP -> per-risk heads -> (batch, K, J) logits

    Args:
        input_dim: dimension of covariates (p)
        n_bins: number of time bins (J)
        n_causes: number of competing event types (K)
        hidden_dim: width of hidden layers
        n_layers: number of shared hidden layers
        dropout: dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        n_bins: int,
        n_causes: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.n_causes = n_causes

        # Shared MLP layers
        shared_layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers)

        # Per-risk heads: each outputs n_bins logits
        self.risk_nets = nn.ModuleList(
            [nn.Linear(hidden_dim, n_bins) for _ in range(n_causes)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: x -> (batch, K, J) raw logits.

        Args:
            x: covariates (batch, input_dim)

        Returns:
            phi: (batch, K, J) raw logits per risk per time bin
        """
        h = self.shared_net(x)
        risk_outs = [net(h) for net in self.risk_nets]
        return torch.stack(risk_outs, dim=1)  # (batch, K, J)

    def get_pmf(self, x: Tensor) -> Tensor:
        """Get PMF using pycox convention: pad-end, softmax, drop-last.

        Following pycox:
          phi: (batch, K, J)
          flat: (batch, K*J)
          padded: (batch, K*J + 1) — pad zero at end
          pmf_flat: softmax(padded)[:, :-1]  — drop last (survival absorbing state)
          pmf: (batch, K, J)

        Args:
            x: covariates (batch, input_dim)

        Returns:
            pmf: (batch, K, J) — cause-bin probabilities
        """
        phi = self.forward(x)  # (batch, K, J)
        batch_size = phi.shape[0]
        flat = phi.reshape(batch_size, -1)  # (batch, K*J)
        # Pad one zero column at end (survival absorbing state)
        pad = torch.zeros(batch_size, 1, device=flat.device)
        padded = torch.cat([flat, pad], dim=1)  # (batch, K*J + 1)
        pmf_flat = torch.softmax(padded, dim=1)[:, :-1]  # (batch, K*J)
        return pmf_flat.reshape(phi.shape)  # (batch, K, J)


def _build_rank_mat(bin_indices: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Build pair rank matrix R where R_ij = 1{T_i < T_j and D_i > 0}.

    Following pycox's pair_rank_mat: R_ij = 1 if subject i has an observed
    event before subject j (or same time but j is censored).

    Args:
        bin_indices: time bin indices (n,)
        delta: event indicators (n,), 0=censored, 1..K=event

    Returns:
        R: (n, n) numpy array
    """
    n = len(bin_indices)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if delta[i] == 0:
            continue
        for j in range(n):
            if (bin_indices[i] < bin_indices[j]) or (
                bin_indices[i] == bin_indices[j] and delta[j] == 0
            ):
                mat[i, j] = 1.0
    return mat


def _nll_pmf_cr(
    pmf: Tensor, idx_durations: Tensor, events: Tensor, eps: float = 1e-7
) -> Tensor:
    """Negative log-likelihood for competing risks PMF (pycox formula).

    For events: -log(pmf[i, k-1, j])
    For censored: -log(1 - sum_k cumsum_pmf[i, k, j])

    Args:
        pmf: (batch, K, J) cause-bin probabilities
        idx_durations: (batch,) bin indices, long
        events: (batch,) event indicators, long, 0=censored, 1..K=event

    Returns:
        scalar NLL loss
    """
    batch_size = pmf.shape[0]
    # events: 1-indexed -> 0-indexed, censored becomes -1
    events_0 = events - 1
    event_01 = (events_0 != -1).float()
    idx = torch.arange(batch_size, device=pmf.device)

    # Event term: log pmf[i, k, j] for subjects with events
    part1 = pmf[idx, events_0, idx_durations].relu().add(eps).log() * event_01

    # Censored term: log(1 - sum_k CIF_k(t_j|x_i))
    # CIF at bin j = cumsum of pmf up to and including bin j
    cif_at_j = pmf.cumsum(dim=2)[idx, :, idx_durations]  # (batch, K)
    surv_at_j = (1.0 - cif_at_j.sum(dim=1)).relu().add(eps).log()
    part2 = surv_at_j * (1.0 - event_01)

    loss = -(part1 + part2)
    return loss.mean()


def _rank_loss_cr(
    pmf: Tensor,
    idx_durations: Tensor,
    events: Tensor,
    rank_mat: Tensor,
    sigma: float,
) -> Tensor:
    """Ranking loss for competing risks (pycox formula).

    For each risk k, compute:
      R_ij = F_i^k(T_i) - F_j^k(T_i)  (CIF difference at event time of i)
      loss_k = sum_ij rank_mat[i,j] * exp(-R_ij / sigma) * I(event_i == k)

    Uses matrix operations matching pycox's _diff_cdf_at_time_i.

    Args:
        pmf: (batch, K, J)
        idx_durations: (batch,)
        events: (batch,), 0=censored, 1..K=event
        rank_mat: (batch, batch) pair indicator matrix
        sigma: bandwidth parameter

    Returns:
        scalar ranking loss
    """
    batch_size, n_risks, n_bins = pmf.shape
    events_0 = events - 1  # 0-indexed, censored = -1

    # One-hot indicator for duration bin: y[i, k, j] = 1 if j == idx_durations[i]
    y = torch.zeros_like(pmf)
    y[torch.arange(batch_size), :, idx_durations] = 1.0

    loss_total = torch.tensor(0.0, device=pmf.device)
    for k in range(n_risks):
        pmf_k = pmf[:, k, :]  # (batch, J)
        y_k = y[:, k, :]  # (batch, J)

        # _diff_cdf_at_time_i: R_ij = F_i(T_i) - F_j(T_i)
        # CIF matrix: (batch, J) cumsum -> (batch, J)
        cif_k = pmf_k.cumsum(dim=1)
        # r = cif_k @ y_k^T: r[i,j] = sum_t F_i(t) * I(T_j == t) = F_i(T_j)
        r = cif_k.matmul(y_k.T)  # (batch, batch)
        # diag_r[i] = F_i(T_i)
        diag_r = r.diag().unsqueeze(0)  # (1, batch)
        # R_ij = F_i(T_i) - F_j(T_i)
        r_diff = diag_r - r  # (batch, batch), need transpose
        r_diff = r_diff.T  # R_ij = F_i(T_i) - F_j(T_i)

        # rank_loss_k = rank_mat * exp(-R/sigma), weighted by event indicator
        rank_loss_k = rank_mat * torch.exp(-r_diff / sigma)
        rank_loss_k = rank_loss_k.mean(dim=1)  # (batch,)
        # Only count for subjects with event k
        event_k_mask = (events_0 == k).float()
        loss_total = loss_total + (rank_loss_k * event_k_mask).mean()

    return loss_total


class DeepHit:
    """DeepHit competing risks model with training and prediction.

    Discretizes time into bins based on quantiles of observed event times,
    trains a shared MLP with per-risk heads to output PMF.

    Args:
        n_bins: number of time discretization bins
        n_causes: number of competing event types
        hidden_dim: width of MLP hidden layers
        n_layers: number of hidden layers
        lr: learning rate
        batch_size: mini-batch size
        epochs: maximum training epochs
        patience: early stopping patience
        alpha: weight for NLL vs ranking loss
            loss = alpha * nll + (1 - alpha) * rank_loss
        sigma: bandwidth for ranking loss
        dropout: dropout rate
    """

    def __init__(
        self,
        n_bins: int = 100,
        n_causes: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 2,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 500,
        patience: int = 50,
        alpha: float = 0.2,
        sigma: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        self.n_bins = n_bins
        self.n_causes = n_causes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.alpha = alpha
        self.sigma = sigma
        self.dropout = dropout

        self.net: DeepHitNet | None = None
        self.bin_edges: np.ndarray | None = None

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> DeepHit:
        """Reconstruct a trained DeepHit model from a checkpoint.

        Args:
            data: checkpoint dict with keys: state_dict, bin_edges,
                n_bins, n_causes, hidden_dim, n_layers

        Returns:
            A DeepHit instance ready for predict_cif()
        """
        model = cls(
            n_bins=data["n_bins"],
            n_causes=data["n_causes"],
            hidden_dim=data["hidden_dim"],
            n_layers=data["n_layers"],
        )
        model.bin_edges = data["bin_edges"]
        # Infer input_dim from first shared layer weight shape
        input_dim = data["state_dict"]["shared_net.0.weight"].shape[1]
        model.net = DeepHitNet(
            input_dim=input_dim,
            n_bins=data["n_bins"],
            n_causes=data["n_causes"],
            hidden_dim=data["hidden_dim"],
            n_layers=data["n_layers"],
        )
        model.net.load_state_dict(data["state_dict"])
        model.net.eval()
        return model

    def _discretize_times(self, Y: np.ndarray, Delta: np.ndarray) -> np.ndarray:
        """Create time bins from quantiles of observed event times.

        Args:
            Y: observed times (n,)
            Delta: event indicators (n,)

        Returns:
            bin_edges: (n_bins + 1,) array of bin boundaries
        """
        event_times = Y[Delta > 0]
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(event_times, quantiles)
        # Ensure unique and sorted
        bin_edges = np.unique(bin_edges)
        # Extend slightly to cover all times
        bin_edges[0] = 0.0
        bin_edges[-1] = bin_edges[-1] + 1e-6
        return bin_edges

    def _time_to_bin(self, t: np.ndarray) -> np.ndarray:
        """Map continuous times to bin indices.

        Args:
            t: times (n,)

        Returns:
            bin_idx: (n,) integer bin indices, 0-indexed
        """
        assert self.bin_edges is not None
        idx = np.searchsorted(self.bin_edges, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.bin_edges) - 2)
        return idx.astype(np.int64)

    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        Delta_train: Tensor,
        X_val: Tensor | None = None,
        Y_val: Tensor | None = None,
        Delta_val: Tensor | None = None,
        verbose: bool = True,
        device: torch.device | str | None = None,
    ) -> list[float]:
        """Train the DeepHit model.

        Args:
            X_train: training covariates (n_train, p)
            Y_train: training observed times (n_train,)
            Delta_train: training event indicators (n_train,)
            X_val: validation covariates (optional, for early stopping)
            Y_val: validation observed times (optional)
            Delta_val: validation event indicators (optional)
            verbose: print progress

        Returns:
            list of training losses per epoch
        """
        Y_train_np = Y_train.numpy()
        Delta_train_np = Delta_train.numpy().astype(np.int64)
        input_dim = X_train.shape[1]

        # Discretize time
        self.bin_edges = self._discretize_times(Y_train_np, Delta_train_np)
        actual_n_bins = len(self.bin_edges) - 1
        if actual_n_bins != self.n_bins:
            if verbose:
                print(
                    f"  DeepHit: adjusted n_bins from {self.n_bins} to "
                    f"{actual_n_bins} (due to duplicate quantiles)"
                )
            self.n_bins = actual_n_bins

        dev = torch.device(device) if device is not None else torch.device("cpu")

        self.net = DeepHitNet(
            input_dim=input_dim,
            n_bins=self.n_bins,
            n_causes=self.n_causes,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(dev)

        bin_idx_train = self._time_to_bin(Y_train_np)

        X_t = X_train.float().to(dev)
        bin_t = torch.tensor(bin_idx_train, dtype=torch.long, device=dev)
        delta_t = Delta_train.long().to(dev)

        # Precompute pair rank matrix for ranking loss
        rank_mat_np = _build_rank_mat(bin_idx_train, Delta_train_np)
        rank_mat_full = torch.tensor(rank_mat_np, device=dev)

        dataset = TensorDataset(X_t, bin_t, delta_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        # Validation data
        has_val = X_val is not None and Y_val is not None and Delta_val is not None
        if has_val:
            assert Y_val is not None and Delta_val is not None
            bin_idx_val = self._time_to_bin(Y_val.numpy())
            X_val_t = X_val.float().to(dev)
            bin_val_t = torch.tensor(bin_idx_val, dtype=torch.long, device=dev)
            delta_val_t = Delta_val.long().to(dev)

        net = self.net
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        losses: list[float] = []
        val_loss = 0.0

        for epoch in range(self.epochs):
            net.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, bin_batch, d_batch in loader:
                pmf = net.get_pmf(x_batch)  # (batch, K, J)

                # NLL loss
                nll = _nll_pmf_cr(pmf, bin_batch, d_batch)

                # Ranking loss (on full batch with precomputed rank_mat subset)
                if self.alpha < 1.0:
                    # Build rank_mat for this mini-batch from the full matrix
                    # Since DataLoader shuffles, we need per-batch rank_mat
                    # For simplicity, compute NLL only on mini-batch,
                    # rank loss on full training data periodically
                    loss = nll
                else:
                    loss = nll

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Compute full ranking loss at end of epoch if alpha < 1
            if self.alpha < 1.0 and (epoch + 1) % 5 == 0:
                net.train()
                full_pmf = net.get_pmf(X_t)
                full_nll = _nll_pmf_cr(full_pmf, bin_t, delta_t)
                full_rank = _rank_loss_cr(
                    full_pmf, bin_t, delta_t, rank_mat_full, self.sigma
                )
                full_loss = self.alpha * full_nll + (1.0 - self.alpha) * full_rank
                optimizer.zero_grad(set_to_none=True)
                full_loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            # Early stopping on validation NLL
            if has_val:
                net.eval()
                with torch.no_grad():
                    val_pmf = net.get_pmf(X_val_t)
                    val_loss = float(
                        _nll_pmf_cr(val_pmf, bin_val_t, delta_val_t).item()
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(
                            f"  DeepHit: early stopping at epoch {epoch + 1} "
                            f"(val_loss={best_val_loss:.6f})"
                        )
                    break

            if verbose and (epoch + 1) % 50 == 0:
                msg = f"  DeepHit epoch {epoch + 1:4d}/{self.epochs}, loss: {avg_loss:.6f}"
                if has_val:
                    msg += f", val_loss: {val_loss:.6f}"
                print(msg)

        # Restore best model
        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        return losses

    def predict_cif(
        self,
        X: Tensor,
        times: Tensor,
    ) -> Tensor:
        """Predict CIF values at specified time points.

        CIF_k(t) = cumsum of PMF_k up to the bin containing t.
        Follows pycox: cif = pmf.cumsum(dim=2).

        Args:
            X: covariates (n, p)
            times: time points to evaluate (n_times,)

        Returns:
            cif: (n, K, n_times) CIF values for each cause at each time
        """
        cif, _ = self.predict_cif_survival(X, times)
        return cif

    def predict_cif_survival(
        self,
        X: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict CIF and native residual-mass survival on a time grid."""
        assert self.net is not None
        assert self.bin_edges is not None
        net = self.net

        net.eval()
        dev = next(net.parameters()).device
        with torch.no_grad():
            pmf = net.get_pmf(X.float().to(dev))

        cif_cumsum = pmf.cumsum(dim=2).cpu()
        n = X.shape[0]
        n_times = len(times)
        cif = torch.zeros(n, self.n_causes, n_times)
        survival = torch.zeros(n, n_times)

        for ti in range(n_times):
            t_val = float(times[ti].item())
            idx = int(np.searchsorted(self.bin_edges, t_val, side="right")) - 1
            idx = max(0, min(idx, self.n_bins - 1))
            cif_at_time = cif_cumsum[:, :, idx]
            cif[:, :, ti] = cif_at_time
            survival[:, ti] = 1.0 - cif_at_time.sum(dim=1)

        return cif, survival
