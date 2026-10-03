"""JointSoftComp: SoftComp on the observed-state process.

As in the joint survival super learner (Munch & Gerds, 2026), censoring is a state of
its own. The network outputs K+1 logits for the observed states {at risk, observed
cause 1..K, censored}; the at-risk logit is fixed at 0. Training minimizes the
cross-entropy of the observed state at times drawn from the empirical distribution of
the training event times. Every label is observed, so the censoring distribution is
never needed. CIFs are recovered with the Aalen–Johansen identity
dLambda_k = dP_k / P_0(t-).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .crsoft import CRSoftNet


def observed_state_labels(
    t: Tensor, y: Tensor, delta: Tensor, num_causes: int
) -> Tensor:
    """Observed state at times ``t`` (n, M): 0 before Y, else Delta (K+1 if censored)."""
    state_after = torch.where(
        delta == 0, torch.full_like(delta, num_causes + 1), delta
    ).long()
    after = t >= y.unsqueeze(1)
    return torch.where(after, state_after.unsqueeze(1).expand_as(after), 0)


def aalen_johansen_from_observed(
    probs: Tensor, num_causes: int
) -> tuple[Tensor, Tensor]:
    """Map observed-state probabilities to CIFs and survival.

    Args:
        probs: (n, J, K+2) probabilities of (at risk, cause 1..K, censored) on a
            time grid whose first point is 0.
        num_causes: K.

    Returns:
        cif (n, K, J) and survival (n, J); monotone with S + sum_k F_k = 1.
    """
    n = probs.shape[0]
    p0 = probs[:, :, 0]
    pk = probs[:, :, 1 : num_causes + 1]
    increments = (pk[:, 1:] - pk[:, :-1]).clamp(min=0.0)
    d_lambda = increments / p0[:, :-1].unsqueeze(-1).clamp(min=1e-12)
    d_lambda = d_lambda / d_lambda.sum(-1, keepdim=True).clamp(min=1.0)
    ones = torch.ones(n, 1, dtype=probs.dtype, device=probs.device)
    survival = torch.cumprod(torch.cat([ones, 1.0 - d_lambda.sum(-1)], dim=1), dim=1)
    cif_increments = survival[:, :-1].unsqueeze(-1) * d_lambda
    zeros = torch.zeros(n, 1, num_causes, dtype=probs.dtype, device=probs.device)
    cif = torch.cat([zeros, cif_increments.cumsum(dim=1)], dim=1)
    return cif.permute(0, 2, 1), survival


class JointSoftComp(CRSoftNet):
    """Residual FFNN with a (K+2)-class softmax over the observed states."""

    def __init__(
        self,
        input_dim: int,
        num_causes: int = 2,
        hidden_dim: int = 32,
        num_blocks: int = 1,
        dropout: float = 0.0,
        grid_size: int = 1000,
    ) -> None:
        super().__init__(input_dim, num_causes + 1, hidden_dim, num_blocks, dropout)
        self.num_causes = num_causes
        self.grid_size = grid_size
        self.register_buffer("tau", torch.tensor(0.0))
        self._time_pool = torch.zeros(1)

    def observed_state_log_probs(self, x: Tensor, t: Tensor) -> Tensor:
        logits = self(x, t)
        zeros = torch.zeros(logits.shape[0], 1, device=logits.device)
        return nn.functional.log_softmax(torch.cat([zeros, logits], dim=1), dim=1)

    def state_nll(self, x: Tensor, y: Tensor, delta: Tensor, t: Tensor) -> Tensor:
        """Mean cross-entropy of the observed state at times ``t`` (n, M)."""
        labels = observed_state_labels(t, y, delta, self.num_causes).reshape(-1)
        x_rep = x.repeat_interleave(t.shape[1], dim=0)
        log_probs = self.observed_state_log_probs(x_rep, t.reshape(-1))
        return nn.functional.nll_loss(log_probs, labels)

    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        Delta_train: Tensor,
        *,
        epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 256,
        n_times: int = 4,
        verbose: bool = True,
        device: torch.device | str | None = None,
        **kwargs: object,
    ) -> list[float]:
        """Train with ``n_times`` times per subject and minibatch, drawn from the
        empirical distribution of the training event times."""
        with torch.no_grad():
            self.tau.fill_(float(Y_train.max()))
        events = Y_train[Delta_train > 0]
        self._time_pool = (events if len(events) else Y_train).detach().clone()
        return super().fit(
            X_train,
            Y_train,
            Delta_train,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            n_aug=n_times,
            verbose=verbose,
            device=device,
            **kwargs,
        )

    def _train_step(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        d_batch: Tensor,
        class_weights: Tensor | None,
        n_aug: int,
        aug_weight: float,
        brier_lambda: float,
        brier_n_times: int,
        brier_t_max: float,
        device: torch.device,
    ) -> Tensor:
        # The parent training loop passes fit()'s n_times as n_aug.
        pool = self._time_pool.to(device)
        shape = (x_batch.shape[0], max(n_aug, 1))
        t = pool[torch.randint(len(pool), shape, device=device)]
        return self.state_nll(x_batch, y_batch, d_batch, t)

    def _val_loss(
        self,
        val_X: Tensor,
        val_Y: Tensor,
        val_D: Tensor,
        class_weights: Tensor | None,
    ) -> float:
        self.eval()
        with torch.no_grad():
            grid = torch.linspace(0.0, float(self.tau), 32, device=val_X.device)
            t = grid.expand(val_X.shape[0], -1)
            return float(self.state_nll(val_X, val_Y, val_D, t).item())

    @torch.no_grad()
    def predict_observed_state_grid(
        self, X: Tensor, times: Tensor, chunk: int = 64
    ) -> Tensor:
        """Observed-state probabilities (n, J, K+2) at ``times``."""
        self.eval()
        device = next(self.parameters()).device
        X, times = X.to(device), times.to(device)
        n = X.shape[0]
        out = []
        for start in range(0, len(times), chunk):
            block = times[start : start + chunk]
            x_rep = X.unsqueeze(1).expand(n, len(block), -1).reshape(n * len(block), -1)
            t_rep = block.unsqueeze(0).expand(n, -1).reshape(-1)
            log_probs = self.observed_state_log_probs(x_rep, t_rep)
            out.append(log_probs.exp().reshape(n, len(block), -1))
        return torch.cat(out, dim=1).cpu()

    @torch.no_grad()
    def predict_cif_survival_grid(
        self, X: Tensor, times: Tensor
    ) -> tuple[Tensor, Tensor]:
        """CIF (n, K, len(times)) and survival via Aalen–Johansen on a fine grid."""
        t_end = max(float(times.max()), 1e-6)
        dense = torch.linspace(0.0, t_end, self.grid_size, dtype=times.dtype)
        grid = torch.unique(torch.cat([dense, times.cpu().clamp(min=0.0)]))
        cif, survival = aalen_johansen_from_observed(
            self.predict_observed_state_grid(X, grid), self.num_causes
        )
        index = torch.searchsorted(grid, times.cpu().clamp(min=0.0))
        return cif[:, :, index], survival[:, index]

    def predict_cif(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        times = t.expand(x.shape[0]) if t.dim() == 0 else t
        grid = torch.unique(times.detach().cpu())
        cif, survival = self.predict_cif_survival_grid(x, grid)
        index = torch.searchsorted(grid, times.detach().cpu()).to(cif.device)
        rows = torch.arange(x.shape[0])
        return cif[rows, :, index], survival[rows, index]

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> JointSoftComp:
        model = cls(
            input_dim=data["input_dim"],
            num_causes=data["num_causes"],
            hidden_dim=data["hidden_dim"],
            num_blocks=data["num_blocks"],
            dropout=data.get("dropout", 0.0),
            grid_size=data.get("grid_size", 1000),
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model
