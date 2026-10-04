#!/usr/bin/env python3
"""
NeuralFineGray (Jeanselme et al.): Neural network implementation of the
Fine-Gray subdistribution hazard model for competing risks.

Key design:
  - PositiveLinear layers ensure cumulative hazard is non-negative
  - Autograd computes the subdistribution hazard from cumulative hazard
  - Balance network learns cause-specific mixing weights
  - CIF_k(t|x) = beta_k * (1 - exp(-H_k(t|x)))

Reference: Jeanselme et al., "Neural Fine-Gray: Monotonic neural networks for
competing risks", 2022.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PositiveLinear(nn.Module):
    """Linear layer with positive weights (via squaring)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.log_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_param: nn.Parameter | None = None
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias_param, -bound, bound)
        self.log_weight.data.abs_().sqrt_()

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.log_weight**2, self.bias_param)


class NeuralFineGrayNet(nn.Module):
    """NeuralFineGray network for competing risks.

    Architecture:
      x → embed MLP → x_rep
      x_rep → balance MLP → log_beta (K) (cause mixing weights)
      (x_rep, t) → PositiveLinear outcome nets → cumulative hazard H_k

    Args:
        input_dim: number of features
        n_causes: number of competing event types
        layers: hidden layer sizes for embedding network
        layers_surv: hidden layer sizes for outcome (survival) networks
        dropout: dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        n_causes: int = 2,
        layers: list[int] | None = None,
        layers_surv: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [100, 100, 100]
        if layers_surv is None:
            layers_surv = [100]
        self.n_causes = n_causes

        # Embedding network: x → x_rep (same dim as input)
        embed_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in layers + [input_dim]:
            embed_layers.append(nn.Linear(prev_dim, hidden))
            embed_layers.append(nn.BatchNorm1d(hidden))
            if dropout > 0:
                embed_layers.append(nn.Dropout(dropout))
            embed_layers.append(nn.ReLU())
            prev_dim = hidden
        self.embed = nn.Sequential(*embed_layers)

        # Balance network: x_rep → log_beta (K)
        balance_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in layers + [n_causes]:
            balance_layers.append(nn.Linear(prev_dim, hidden))
            balance_layers.append(nn.BatchNorm1d(hidden))
            balance_layers.append(nn.ReLU())
            prev_dim = hidden
        self.balance = nn.Sequential(*balance_layers)

        # Outcome networks: (x_rep, t) → cumulative hazard per cause
        # PositiveLinear ensures monotonicity
        self.outcome = nn.ModuleList()
        for _ in range(n_causes):
            surv_layers: list[nn.Module] = []
            prev_dim = input_dim + 1  # x_rep + t
            for hidden in layers_surv:
                surv_layers.append(PositiveLinear(prev_dim, hidden, bias=True))
                if dropout > 0:
                    surv_layers.append(nn.Dropout(dropout))
                surv_layers.append(nn.Tanh())
                prev_dim = hidden
            # Last layer: output 1, with Softplus activation
            surv_layers.append(PositiveLinear(prev_dim, 1, bias=True))
            surv_layers[-1] = PositiveLinear(prev_dim, 1, bias=True)
            surv_layers.append(nn.Softplus())
            self.outcome.append(nn.Sequential(*surv_layers))

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: covariates (batch, p)
            t: time points (batch,)

        Returns:
            log_sr: log cumulative hazard (batch, K) — negative values
            log_beta: log mixing weights (batch, K)
            tau: time tensor with grad enabled (batch, 1)
        """
        x_rep = self.embed(x)
        log_beta = self.log_softmax(self.balance(x_rep))

        # Compute cumulative hazard for each cause
        tau = t.clone().detach().requires_grad_(True).unsqueeze(1)
        sr_list = []
        for outcome_net in self.outcome:
            h = tau * outcome_net(torch.cat([x_rep, tau], dim=1))
            sr_list.append(-h)  # log S_k = -H_k

        log_sr = torch.cat(sr_list, dim=1)  # (batch, K)
        return log_sr, log_beta, tau

    def gradient(self, log_sr: Tensor, tau: Tensor, events: Tensor) -> Tensor:
        """Compute hazard via autograd (gradient of cumulative hazard w.r.t. time).

        Args:
            log_sr: log survival (batch, K), from forward()
            tau: time tensor with grad (batch, 1), from forward()
            events: event indicators (batch,), 1..K

        Returns:
            hazard: (batch,) hazard values for subjects with events
        """
        # Compute per-cause grad of -log_sr w.r.t. tau to get hazard
        # Sum all cause contributions, then take grad once
        total = torch.zeros(1, device=log_sr.device, dtype=log_sr.dtype)
        for r in range(self.n_causes):
            mask = events == (r + 1)
            if mask.any():
                total = total + (-log_sr[:, r][mask]).sum()
        grads = torch.autograd.grad(
            total,
            tau,
            create_graph=True,
            allow_unused=True,
        )[0]
        if grads is None:
            grads = torch.zeros_like(tau)
        return grads.clamp(min=1e-10)[:, 0]


def _nfg_loss(model: NeuralFineGrayNet, x: Tensor, t: Tensor, e: Tensor) -> Tensor:
    """NeuralFineGray loss: Fine-Gray subdistribution log-likelihood.

    For censored (e=0): -log(sum_k beta_k * S_k(t))
    For event k (e=k): -log(beta_k * f_k(t)) = -log(beta_k) - log(h_k(t)) + H_k(t)

    Args:
        model: NeuralFineGrayNet
        x: covariates (batch, p)
        t: times (batch,)
        e: event indicators (batch,), 0=censored, 1..K=event
    """
    log_sr, log_beta, tau = model.forward(x, t)

    # log(beta_k * S_k) = log_beta_k + log_S_k
    log_balance_sr = log_beta + log_sr

    # Censored: -log(sum_k beta_k * S_k(t))
    error = torch.tensor(0.0, device=x.device)
    if (e == 0).any():
        error = error - torch.logsumexp(log_balance_sr[e == 0], dim=1).sum()

    # Events: -log(beta_k * h_k * S_k) — need hazard via gradient
    has_events = (e > 0).any()
    if has_events:
        log_hr = model.gradient(log_sr, tau, e).log()
        for k in range(model.n_causes):
            mask = e == (k + 1)
            if mask.any():
                error = error - (log_balance_sr[mask][:, k] + log_hr[mask]).sum()

    return error / len(x)


class NeuralFG:
    """NeuralFineGray competing risks model with training and prediction.

    Args:
        n_causes: number of competing event types
        layers: hidden layer sizes for embedding network
        layers_surv: hidden layer sizes for outcome networks
        lr: learning rate
        batch_size: mini-batch size
        epochs: maximum training epochs
        patience: early stopping patience (consecutive non-improvements)
        dropout: dropout rate
        weight_decay: L2 regularization
    """

    def __init__(
        self,
        n_causes: int = 2,
        layers: list[int] | None = None,
        layers_surv: list[int] | None = None,
        lr: float = 1e-3,
        batch_size: int = 100,
        epochs: int = 1000,
        patience: int = 3,
        dropout: float = 0.0,
        weight_decay: float = 0.001,
    ) -> None:
        self.n_causes = n_causes
        self.layers = layers or [100, 100, 100]
        self.layers_surv = layers_surv or [100]
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.net: NeuralFineGrayNet | None = None
        self._time_max: float = 1.0

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> NeuralFG:
        """Reconstruct a trained NeuralFineGray model from a checkpoint.

        Args:
            data: checkpoint dict with keys: state_dict, n_causes,
                  layers, layers_surv, _time_max

        Returns:
            A NeuralFG instance ready for predict_cif()
        """
        model = cls(
            n_causes=data["n_causes"],
            layers=data["layers"],
            layers_surv=data["layers_surv"],
        )
        model._time_max = data["_time_max"]
        # Infer input_dim from the first embedding layer weight shape
        input_dim = data["state_dict"]["embed.0.weight"].shape[1]
        model.net = NeuralFineGrayNet(
            input_dim=input_dim,
            n_causes=data["n_causes"],
            layers=data["layers"],
            layers_surv=data["layers_surv"],
        ).double()
        model.net.load_state_dict(data["state_dict"])
        model.net.eval()
        return model

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
        """Train the NeuralFineGray model.

        Args:
            X_train: training covariates (n, p)
            Y_train: training observed times (n,)
            Delta_train: training event indicators (n,)
            X_val: validation covariates (optional)
            Y_val: validation observed times (optional)
            Delta_val: validation event indicators (optional)
            verbose: print progress

        Returns:
            list of training losses per epoch
        """
        input_dim = X_train.shape[1]

        dev = torch.device(device) if device is not None else torch.device("cpu")

        # Normalize time to [0, 1] range for stability
        self._time_max = float(Y_train.max().item()) + 1.0
        t_train = ((Y_train.double() + 1.0) / self._time_max).to(dev)
        x_train = X_train.double().to(dev)
        e_train = Delta_train.long().to(dev)

        has_val = X_val is not None and Y_val is not None and Delta_val is not None
        if has_val:
            assert Y_val is not None and Delta_val is not None
            t_val = ((Y_val.double() + 1.0) / self._time_max).to(dev)
            x_val = X_val.double().to(dev)
            e_val = Delta_val.long().to(dev)

        self.net = (
            NeuralFineGrayNet(
                input_dim=input_dim,
                n_causes=self.n_causes,
                layers=self.layers,
                layers_surv=self.layers_surv,
                dropout=self.dropout,
            )
            .double()
            .to(dev)
        )

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_loss = float("inf")
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        patience_counter = 0
        prev_val_loss = float("inf")
        losses: list[float] = []

        n = x_train.shape[0]
        n_batches = (n + self.batch_size - 1) // self.batch_size

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0

            for j in range(n_batches):
                idx = perm[j * self.batch_size : (j + 1) * self.batch_size]
                if len(idx) == 0:
                    continue

                xb = x_train[idx]
                tb = t_train[idx]
                eb = e_train[idx]

                optimizer.zero_grad()
                loss = _nfg_loss(self.net, xb, tb, eb)
                if loss.grad_fn is not None:
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            # Validation-based early stopping
            if has_val:
                self.net.eval()
                # NOTE: no torch.no_grad() here — NFG needs autograd for hazard
                val_loss = _nfg_loss(self.net, x_val, t_val, e_val).item()

                if val_loss < prev_val_loss:
                    patience_counter = 0
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {
                            k: v.clone() for k, v in self.net.state_dict().items()
                        }
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(
                            f"  NeuralFG: early stopping at epoch {epoch + 1} "
                            f"(val_loss={best_val_loss:.6f})"
                        )
                    break
                prev_val_loss = val_loss

            if verbose and (epoch + 1) % 50 == 0:
                msg = f"  NeuralFG epoch {epoch + 1:4d}/{self.epochs}, loss: {avg_loss:.6f}"
                if has_val:
                    msg += f", val_loss: {val_loss:.6f}"
                print(msg)

        self.net.load_state_dict(best_state)
        self.net.eval()
        return losses

    def predict_cif(self, X: Tensor, times: Tensor) -> Tensor:
        """Predict CIF values at specified time points.

        CIF_k(t|x) = beta_k * (1 - exp(log_S_k(t|x)))

        Args:
            X: covariates (n, p)
            times: time points (n_times,)

        Returns:
            cif: (n, K, n_times) CIF values
        """
        cif, _ = self.predict_cif_survival(X, times)
        return cif

    def predict_cif_survival(
        self,
        X: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict CIF and the model's mixture survival."""
        assert self.net is not None
        self.net.eval()
        dev = next(self.net.parameters()).device
        x = X.double().to(dev)
        n = x.shape[0]
        n_times = len(times)
        cif = torch.zeros(n, self.n_causes, n_times)
        survival = torch.zeros(n, n_times)

        with torch.no_grad():
            for ti in range(n_times):
                t_val = (float(times[ti].item()) + 1.0) / self._time_max
                t_batch = torch.full((n,), t_val, dtype=torch.float64, device=dev)
                log_sr, log_beta, _ = self.net(x, t_batch)
                beta = log_beta.exp()
                cause_survival = log_sr.exp()
                cif[:, :, ti] = (beta * (1.0 - cause_survival)).float().cpu()
                survival[:, ti] = (beta * cause_survival).sum(dim=1).float().cpu()

        return cif, survival
