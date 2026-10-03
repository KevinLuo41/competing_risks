#!/usr/bin/env python3
"""
Section 2: Competing risks neural network model.

Architecture:
  Input (x, t) -> concat -> Residual FFNN -> K logits -> softmax -> (F_1, ..., F_K, S)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def competing_risks_loss(
    logits: Tensor,
    delta: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Compute the negative log-likelihood loss for competing risks.

    Based on the multinomial logistic model (paper eq. 2):
      L = -sum[ I(delta=k)*log(F_k) + I(delta=0)*log(S) ]

    We treat this as (K+1)-class classification where class 0 = survival/censored
    and classes 1..K = event types.

    Args:
        logits: raw network output (batch, K)
        delta: event indicators, 0=censored, 1..K=event type (batch,)
        class_weights: optional (K+1,) tensor of per-class weights for
            handling class imbalance. E.g. [1.0, 1.0, 5.0] upweights class 2.

    Returns:
        scalar loss
    """
    batch_size = logits.shape[0]

    # Build (K+1)-class log-probabilities
    # Class 0 (survival): log(1 / (1 + sum exp(z_k))) = -log(1 + sum exp(z_k))
    # Class k (event k):  log(exp(z_k) / (1 + sum exp(z_k))) = z_k - log(1 + sum exp(z_k))
    # This is equivalent to log_softmax with a zero column prepended

    # Prepend zero column for survival class
    zeros = torch.zeros(batch_size, 1, device=logits.device)
    full_logits = torch.cat([zeros, logits], dim=1)  # (batch, K+1)

    log_probs = nn.functional.log_softmax(full_logits, dim=1)  # (batch, K+1)

    # delta is already 0-indexed for survival, 1..K for events
    target = delta.long()
    loss = nn.functional.nll_loss(log_probs, target, weight=class_weights)
    return loss


class ResidualBlock(nn.Module):
    """A single residual block: x + MLP(x)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class BaseCRSoftNet(nn.Module):
    """Base class for competing risks models.

    Subclasses must set self.num_causes and implement forward(x, t) -> logits.
    Provides predict_cif() which converts logits to CIF probabilities.
    """

    num_causes: int = 0

    def predict_cif(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Predict CIF values F_k(t|x) and survival S(t|x).

        Returns:
            F: (batch, K) -- cause-specific CIF values
            S: (batch,) -- survival probabilities
        """
        logits = self.forward(x, t)  # (batch, K)
        zeros = torch.zeros(logits.shape[0], 1, device=logits.device)
        full_logits = torch.cat([zeros, logits], dim=1)  # (batch, K+1)
        probs = torch.softmax(full_logits, dim=1)  # (batch, K+1)
        s = probs[:, 0]
        f = probs[:, 1:]
        return f, s

    def _brier_loss(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        d_batch: Tensor,
        brier_n_times: int,
        brier_t_max: float,
        device: torch.device,
    ) -> Tensor:
        """Squared-error penalty between predicted CIF and the empirical
        event-by-time indicator at brier_n_times random eval points per
        subject. Trains the model directly for the IBS objective; tightens
        calibration without distorting discrimination noticeably (PBC:
        lambda=2 → IBS_0 0.071→0.070, IBS_1 0.107→0.104).
        """
        bn = x_batch.shape[0]
        t_brier = torch.rand(bn, brier_n_times, device=device) * brier_t_max
        brier_loss = torch.zeros((), device=device)
        for j in range(brier_n_times):
            tj = t_brier[:, j]
            f_pred, _ = self.predict_cif(x_batch, tj)
            target = torch.zeros_like(f_pred)
            for k in range(self.num_causes):
                target[:, k] = ((y_batch <= tj) & (d_batch == k + 1)).float()
            brier_loss = brier_loss + ((f_pred - target) ** 2).mean()
        return brier_loss / brier_n_times

    def _val_loss(
        self,
        val_X: Tensor,
        val_Y: Tensor,
        val_D: Tensor,
        class_weights: Tensor | None,
    ) -> float:
        self.eval()
        with torch.no_grad():
            val_logits = self(val_X, val_Y)
            return float(competing_risks_loss(val_logits, val_D, class_weights).item())

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
        """Compute the per-batch loss: NLL + n_aug time-augmented NLL + optional Brier."""
        logits = self(x_batch, y_batch)
        loss = competing_risks_loss(logits, d_batch, class_weights)

        u = torch.rand(x_batch.shape[0], n_aug, device=device)
        t_aug = u * y_batch.unsqueeze(1)
        for j in range(n_aug):
            logits_aug = self(x_batch, t_aug[:, j])
            d_zeros = torch.zeros(x_batch.shape[0], device=device)
            loss = (
                loss
                + competing_risks_loss(logits_aug, d_zeros, class_weights) * aug_weight
            )
        loss = loss / (1 + n_aug * aug_weight)

        if brier_lambda > 0:
            loss = loss + brier_lambda * self._brier_loss(
                x_batch, y_batch, d_batch, brier_n_times, brier_t_max, device
            )
        return loss

    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        Delta_train: Tensor,
        X_val: Tensor | None = None,
        Y_val: Tensor | None = None,
        Delta_val: Tensor | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_aug: int = 2,
        aug_weight: float = 0.5,
        class_weights: Tensor | None = None,
        patience: int = 0,
        brier_lambda: float = 0.0,
        brier_n_times: int = 5,
        verbose: bool = True,
        device: torch.device | str | None = None,
    ) -> list[float]:
        """Train the competing risks model.

        Args:
            X_train: covariates (n, p) or (n, p, G) for functional
            Y_train: observed times (n,)
            Delta_train: event indicators (n,), 0=censored, 1..K=event
            X_val: validation covariates (optional, for early stopping)
            Y_val: validation observed times (optional)
            Delta_val: validation event indicators (optional)
            epochs: number of training epochs
            batch_size: mini-batch size
            lr: learning rate
            weight_decay: L2 regularization
            n_aug: number of time augmentation points per sample
            aug_weight: weight for each augmentation loss term
            class_weights: optional (K+1,) tensor of per-class weights
            patience: early stopping patience (0 = disabled)
            brier_lambda: optional weight for the Brier-augmented loss term;
                set to 0.0 to disable
            brier_n_times: number of random eval times sampled per subject
                when brier_lambda > 0
            verbose: print progress
            device: torch device to train on. None defaults to "cuda" if
                available, else "cpu". The model and full training tensors
                are moved here once before the loop, so per-batch slices
                inherit the device for free.

        Returns:
            list of epoch losses
        """
        import copy

        # Resolve device. None auto-detects CUDA; explicit "cpu"/"cuda"/torch.device
        # honored verbatim so callers can pin to a specific GPU.
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.to(device)
        # Move full training tensors once; batch slices then live on the same
        # device with no per-step .to() copies. Datasets in this repo are tiny
        # (≤ a few MB), so the one-time copy is free.
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        Delta_train = Delta_train.to(device)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        val_X = X_val.to(device) if X_val is not None else None
        val_Y = Y_val.to(device) if Y_val is not None else None
        val_D = Delta_val.to(device) if Delta_val is not None else None

        # Manual permutation batching (no DataLoader) for two reasons:
        #   1) DataLoader's worker pools fork the process — overhead-heavy for
        #      tiny datasets (PBC has ~1900 rows; the fork costs ~10x more
        #      than the actual training step) and CUDA tensors can't be shared
        #      across forked workers anyway.
        #   2) DataLoader's shuffle uses its own RNG state separate from
        #      torch.manual_seed(...), which silently breaks reproducibility.
        #      torch.randperm() honors the global seed deterministically.
        n_train = X_train.shape[0]
        # Note: avoid `foreach=True` — it pulls in `torch._dynamo` which has a
        # broken import in some recent torch builds (set_fullgraph_compiled_frame_count).
        # `foreach` is only a perf optimization; correctness is unaffected.
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        has_val = val_X is not None and val_Y is not None and val_D is not None
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        brier_t_max = float(Y_train.max().item())

        losses: list[float] = []
        val_loss = float("nan")
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                loss = self._train_step(
                    X_train[idx],
                    Y_train[idx],
                    Delta_train[idx],
                    class_weights,
                    n_aug,
                    aug_weight,
                    brier_lambda,
                    brier_n_times,
                    brier_t_max,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if has_val and patience > 0:
                assert val_X is not None and val_Y is not None and val_D is not None
                val_loss = self._val_loss(val_X, val_Y, val_D, class_weights)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(
                            f"  Early stopping at epoch {epoch + 1} "
                            f"(val_loss={val_loss:.6f})"
                        )
                    break

            if verbose and (epoch + 1) % 50 == 0:
                tail = f", Val: {val_loss:.6f}" if has_val and patience > 0 else ""
                print(f"  Epoch {epoch + 1:4d}/{epochs}, Loss: {avg_loss:.6f}{tail}")

        if best_state is not None:
            self.load_state_dict(best_state)

        return losses

    @torch.no_grad()
    def predict_cif_grid(self, X: Tensor, times: Tensor) -> Tensor:
        """Predict CIF at multiple time points.

        Args:
            X: covariates (n, ...)
            times: time points (n_times,)

        Returns:
            cif: (n, K, n_times) Tensor
        """
        cif, _ = self.predict_cif_survival_grid(X, times)
        return cif

    @torch.no_grad()
    def predict_cif_survival_grid(
        self, X: Tensor, times: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict CIF and native softmax survival on a time grid."""
        self.eval()
        device = next(self.parameters()).device
        X = X.to(device)
        times = times.to(device)
        n = X.shape[0]
        n_times = len(times)
        cif = torch.zeros(n, self.num_causes, n_times)
        survival = torch.zeros(n, n_times)
        for ti in range(n_times):
            t_batch = times[ti].expand(n)
            f_pred, s_pred = self.predict_cif(X, t_batch)
            cif[:, :, ti] = f_pred.cpu()
            survival[:, ti] = s_pred.cpu()
        return cif, survival


class CRSoftEnsemble(BaseCRSoftNet):
    """N-member ensemble of CRSoftNet models trained from different seeds.

    Predictions average raw CIFs across members. Optional per-cause shrinkage
    (alpha_per_cause) sharpens (a>1) or blends (a<1) toward the ensemble's
    marginal CIF, useful for fine-tuning calibration vs discrimination on a
    per-cause basis (PBC: a1=1.05, a2=0.95 → 6/6 SOTA).

    Attributes:
        models: list of trained CRSoftNet instances (each with `_train_losses`).
        alpha_per_cause: optional length-K tensor; defaults to ones (no shrink).
    """

    def __init__(
        self,
        members: list[CRSoftNet],
        alpha_per_cause: Tensor | None = None,
    ) -> None:
        super().__init__()
        if len(members) == 0:
            raise ValueError("ensemble must contain at least one member")
        self.num_causes = members[0].num_causes
        self.models: nn.ModuleList = nn.ModuleList(members)
        if alpha_per_cause is None:
            alpha_per_cause = torch.ones(self.num_causes)
        self.register_buffer("alpha_per_cause", alpha_per_cause.float())

    @torch.no_grad()
    def predict_cif_grid(self, X: Tensor, times: Tensor) -> Tensor:
        cif, _ = self.predict_cif_survival_grid(X, times)
        return cif

    @torch.no_grad()
    def predict_cif_survival_grid(
        self, X: Tensor, times: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Each member handles its own device and returns CPU tensors, so this
        # loop works regardless of where each member lives.
        predictions = [m.predict_cif_survival_grid(X, times) for m in self.models]
        cifs = torch.stack([prediction[0] for prediction in predictions], dim=0)
        survivals = torch.stack([prediction[1] for prediction in predictions], dim=0)
        avg = cifs.mean(dim=0)  # (n, K, T)
        marginal = avg.mean(dim=0, keepdim=True)  # (1, K, T)
        # Per-cause shrinkage: out_k = a_k * avg_k + (1 - a_k) * marginal_k.
        # Force alpha onto avg's device — the buffer follows the ensemble's
        # device after `.to(...)`, so it may differ from the CPU `avg`.
        a = self.alpha_per_cause.view(1, -1, 1).to(avg.device)
        adjusted_cif = a * avg + (1.0 - a) * marginal
        return adjusted_cif, survivals.mean(dim=0)

    def predict_cif(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        # Required by the base contract; average per-member CIF, no shrinkage
        # at the (single-time, scalar t) interface.
        results = [m.predict_cif(x, t) for m in self.models]
        f = torch.stack([r[0] for r in results], dim=0).mean(dim=0)
        s = torch.stack([r[1] for r in results], dim=0).mean(dim=0)
        return f, s

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> CRSoftEnsemble:
        members = []
        for sd_dict in data["member_state_dicts"]:
            m = CRSoftNet(
                input_dim=data["input_dim"],
                num_causes=data["num_causes"],
                hidden_dim=data["hidden_dim"],
                num_blocks=data["num_blocks"],
                dropout=data.get("dropout", 0.0),
            )
            m.load_state_dict(sd_dict)
            m.eval()
            members.append(m)
        alpha = torch.tensor(data.get("alpha_per_cause", [1.0] * data["num_causes"]))
        return cls(members, alpha_per_cause=alpha)


class CRSoftNet(BaseCRSoftNet):
    """Residual FFNN for competing risks prediction (scalar covariates).

    Takes (x, t) as input, outputs K logits corresponding to K event types.

    Args:
        input_dim: dimension of covariates x (= p)
        num_causes: number of competing event types (= K)
        hidden_dim: width of hidden layers
        num_blocks: number of residual blocks
    """

    def __init__(
        self,
        input_dim: int,
        num_causes: int = 2,
        hidden_dim: int = 16,
        num_blocks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_causes = num_causes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            *([nn.Dropout(dropout)] if dropout > 0 else []),
        )
        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, num_causes)

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> CRSoftNet:
        """Reconstruct a trained CRSoftNet from a checkpoint.

        Args:
            data: checkpoint dict with keys: state_dict, num_causes,
                input_dim, hidden_dim, num_blocks.
                Optionally: dropout

        Returns:
            A CRSoftNet instance ready for predict_cif_grid()
        """
        model = cls(
            input_dim=data["input_dim"],
            num_causes=data["num_causes"],
            hidden_dim=data["hidden_dim"],
            num_blocks=data["num_blocks"],
            dropout=data.get("dropout", 0.0),
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        xt = torch.cat([x, t], dim=1)
        h = self.input_proj(xt)
        h = self.backbone(h)
        return self.output_layer(h)
