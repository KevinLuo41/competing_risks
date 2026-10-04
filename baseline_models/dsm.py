#!/usr/bin/env python3
"""
DSM (Nagpal et al., 2020): Deep Survival Machines models event times as a
mixture of K parametric distributions (Weibull or LogNormal) with
covariate-dependent shape/scale parameters and mixture weights.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class DSMNet(nn.Module):
    """Deep Survival Machines network for competing risks.

    For each risk k, models the event time distribution as a mixture of M
    parametric distributions (Weibull or LogNormal). The shape/scale parameters
    and mixture weights are covariate-dependent via a shared MLP representation.

    Architecture:
        x -> shared MLP (representation) -> per-risk heads:
          - shape_head[k]: linear -> activation + base_shape
          - scale_head[k]: linear -> activation + base_scale
          - gate_head[k]:  linear -> softmax (mixture weights)

    Args:
        input_dim: dimension of covariates (p)
        n_causes: number of competing event types (K)
        k: number of mixture components (M)
        layers: list of hidden layer widths for the shared representation
        distribution: "Weibull" or "LogNormal"
        temp: temperature for gate logits (default 1000)
    """

    def __init__(
        self,
        input_dim: int,
        n_causes: int = 2,
        k: int = 6,
        layers: list[int] | None = None,
        distribution: str = "Weibull",
        temp: float = 1000.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.n_causes = n_causes
        self.dist = distribution
        self.temp = temp

        if layers is None:
            layers = [64, 64]

        # Shared representation MLP
        modules: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in layers:
            modules.append(nn.Linear(prev_dim, hidden, bias=False))
            modules.append(nn.ReLU6())
            prev_dim = hidden
        self.embedding = nn.Sequential(*modules)
        last_dim = prev_dim

        # Per-risk distribution parameters
        if distribution == "Weibull":
            self.act = nn.SELU()
            init_val = -1.0
        elif distribution == "LogNormal":
            self.act = nn.Tanh()
            init_val = 1.0
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Base shape/scale parameters (learned, shared across samples)
        self.base_shape = nn.ParameterDict(
            {
                str(r + 1): nn.Parameter(torch.full((k,), init_val))
                for r in range(n_causes)
            }
        )
        self.base_scale = nn.ParameterDict(
            {
                str(r + 1): nn.Parameter(torch.full((k,), init_val))
                for r in range(n_causes)
            }
        )

        # Covariate-dependent adjustments
        self.shape_head = nn.ModuleDict(
            {str(r + 1): nn.Linear(last_dim, k, bias=True) for r in range(n_causes)}
        )
        self.scale_head = nn.ModuleDict(
            {str(r + 1): nn.Linear(last_dim, k, bias=True) for r in range(n_causes)}
        )
        self.gate_head = nn.ModuleDict(
            {str(r + 1): nn.Linear(last_dim, k, bias=False) for r in range(n_causes)}
        )

    def forward(self, x: Tensor, risk: str = "1") -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for a given risk.

        Args:
            x: covariates (batch, input_dim)
            risk: risk index as string ("1", "2", ...)

        Returns:
            shape: (batch, k) shape parameters
            scale: (batch, k) scale parameters
            logits: (batch, k) gate logits (pre-softmax)
        """
        xrep = self.embedding(x)
        n = x.shape[0]
        shape = self.act(self.shape_head[risk](xrep)) + self.base_shape[risk].expand(
            n, -1
        )
        scale = self.act(self.scale_head[risk](xrep)) + self.base_scale[risk].expand(
            n, -1
        )
        logits = self.gate_head[risk](xrep) / self.temp
        return shape, scale, logits

    def forward_all(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return distribution parameters for every cause with one embedding pass."""
        xrep = self.embedding(x)
        risks = [str(cause) for cause in range(1, self.n_causes + 1)]

        shape_weights = torch.stack(
            [self.shape_head[risk].weight for risk in risks], dim=0
        )
        shape_biases = torch.stack(
            [self.shape_head[risk].bias for risk in risks], dim=0
        )
        base_shapes = torch.stack([self.base_shape[risk] for risk in risks], dim=0)
        shapes = (
            self.act(torch.einsum("bd,kmd->bkm", xrep, shape_weights) + shape_biases)
            + base_shapes
        )

        scale_weights = torch.stack(
            [self.scale_head[risk].weight for risk in risks], dim=0
        )
        scale_biases = torch.stack(
            [self.scale_head[risk].bias for risk in risks], dim=0
        )
        base_scales = torch.stack([self.base_scale[risk] for risk in risks], dim=0)
        scales = (
            self.act(torch.einsum("bd,kmd->bkm", xrep, scale_weights) + scale_biases)
            + base_scales
        )

        gate_weights = torch.stack(
            [self.gate_head[risk].weight for risk in risks], dim=0
        )
        logits = torch.einsum("bd,kmd->bkm", xrep, gate_weights) / self.temp
        return shapes, scales, logits


def _dsm_conditional_loss(
    model: DSMNet,
    x: Tensor,
    t: Tensor,
    e: Tensor,
    elbo: bool = True,
    risk: str = "1",
) -> Tensor:
    """Conditional loss for DSM (Weibull or LogNormal).

    For uncensored (event=risk): log f_k(t|x) (log-density)
    For censored (event!=risk): log S_k(t|x) (log-survival)

    Both are computed as mixtures weighted by the gate.

    Args:
        model: DSMNet instance
        x: covariates (batch, p)
        t: event times (batch,)
        e: event indicators (batch,), 0=censored, 1..K=event
        elbo: if True, use ELBO (expectation); else use log-sum-exp
        risk: risk index as string

    Returns:
        scalar negative log-likelihood loss
    """
    shape, scale, logits = model.forward(x, risk)

    time = t.clamp(min=1e-10).unsqueeze(1)
    log_time = torch.log(time)
    if model.dist == "Weibull":
        exp_shape = torch.exp(shape)
        log_scaled_time = scale + log_time
        losss = -torch.pow(torch.exp(scale) * time, exp_shape)
        lossf = shape + scale + (exp_shape - 1.0) * log_scaled_time + losss
    elif model.dist == "LogNormal":
        lossf = -scale - 0.5 * float(np.log(2 * np.pi))
        lossf = lossf - torch.div(
            # pyre-ignore[58]: Tensor supports ** operator
            (log_time - shape) ** 2,
            2.0 * torch.exp(2 * scale),
        )
        standardized = torch.div(
            log_time - shape,
            torch.exp(scale) * float(np.sqrt(2)),
        )
        losss = torch.log((0.5 - 0.5 * torch.erf(standardized)).clamp(min=1e-10))
    else:
        raise ValueError(f"Unknown distribution: {model.dist}")

    if elbo:
        # ELBO: E_q[log p] using softmax weights
        lossg = nn.Softmax(dim=1)(logits)
        losss = (lossg * losss).sum(dim=1)
        lossf = (lossg * lossf).sum(dim=1)
    else:
        # Exact: log sum_m w_m * p_m via log-sum-exp
        lossg = nn.LogSoftmax(dim=1)(logits)
        losss = torch.logsumexp(lossg + losss, dim=1)
        lossf = torch.logsumexp(lossg + lossf, dim=1)

    uncensored = e == int(risk)
    ll = lossf[uncensored].sum() + losss[~uncensored].sum()
    return -ll / float(e.numel())


def _dsm_all_causes_loss(
    model: DSMNet,
    x: Tensor,
    t: Tensor,
    e: Tensor,
    elbo: bool,
) -> Tensor:
    shape, scale, logits = model.forward_all(x)
    time = t.clamp(min=1e-10).view(-1, 1, 1)
    log_time = torch.log(time)
    if model.dist == "Weibull":
        exp_shape = torch.exp(shape)
        log_scaled_time = scale + log_time
        log_survival = -torch.pow(torch.exp(scale) * time, exp_shape)
        log_density = shape + scale + (exp_shape - 1.0) * log_scaled_time + log_survival
    elif model.dist == "LogNormal":
        log_density = -scale - 0.5 * float(np.log(2 * np.pi))
        log_density = log_density - torch.div(
            # pyre-ignore[58]: Tensor supports ** operator
            (log_time - shape) ** 2,
            2.0 * torch.exp(2 * scale),
        )
        standardized = torch.div(
            log_time - shape,
            torch.exp(scale) * float(np.sqrt(2)),
        )
        log_survival = torch.log((0.5 - 0.5 * torch.erf(standardized)).clamp(min=1e-10))
    else:
        raise ValueError(f"Unknown distribution: {model.dist}")

    if elbo:
        weights = torch.softmax(logits, dim=2)
        log_survival = (weights * log_survival).sum(dim=2)
        log_density = (weights * log_density).sum(dim=2)
    else:
        log_weights = torch.log_softmax(logits, dim=2)
        log_survival = torch.logsumexp(log_weights + log_survival, dim=2)
        log_density = torch.logsumexp(log_weights + log_density, dim=2)

    causes = torch.arange(
        1,
        model.n_causes + 1,
        dtype=e.dtype,
        device=e.device,
    ).view(1, -1)
    uncensored = e.view(-1, 1) == causes
    log_likelihood = torch.where(uncensored, log_density, log_survival)
    return -log_likelihood.sum() / float(e.numel())


def _dsm_predict_cdf(
    model: DSMNet,
    x: Tensor,
    t_horizon: list[float],
    risk: str = "1",
) -> Tensor:
    """Predict log-survival values at given time horizons for a risk.

    Returns log S_k(t|x) for all requested times.

    Args:
        model: DSMNet instance
        x: covariates (batch, p)
        t_horizon: list of time points
        risk: risk index as string

    Returns:
        tensor with shape (batch, n_times), containing log S_k(t|x)
    """
    shape, scale, logits = model.forward(x, risk)
    log_weights = nn.LogSoftmax(dim=1)(logits)

    times = torch.tensor(t_horizon, dtype=x.dtype, device=x.device)
    times = times.clamp(min=1e-10).view(1, 1, -1)
    if model.dist == "Weibull":
        component_log_survival = -torch.pow(
            torch.exp(scale).unsqueeze(2) * times,
            torch.exp(shape).unsqueeze(2),
        )
    elif model.dist == "LogNormal":
        standardized = torch.div(
            torch.log(times) - shape.unsqueeze(2),
            torch.exp(scale).unsqueeze(2) * float(np.sqrt(2)),
        )
        component_log_survival = torch.log(
            (0.5 - 0.5 * torch.erf(standardized)).clamp(min=1e-10)
        )
    else:
        raise ValueError(f"Unknown distribution: {model.dist}")
    return torch.logsumexp(
        component_log_survival + log_weights.unsqueeze(2),
        dim=1,
    )


def _dsm_predict_all_cause_survival(
    model: DSMNet,
    x: Tensor,
    times: Tensor,
    batch_size: int = 512,
) -> Tensor:
    output = torch.empty(x.shape[0], model.n_causes, len(times))
    time_grid = times.to(dtype=x.dtype, device=x.device).clamp(min=1e-10)
    time_grid = time_grid.view(1, 1, 1, -1)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        shape, scale, logits = model.forward_all(x[start:end])
        if model.dist == "Weibull":
            component_log_survival = -torch.pow(
                torch.exp(scale).unsqueeze(3) * time_grid,
                torch.exp(shape).unsqueeze(3),
            )
        elif model.dist == "LogNormal":
            standardized = torch.div(
                torch.log(time_grid) - shape.unsqueeze(3),
                torch.exp(scale).unsqueeze(3) * float(np.sqrt(2)),
            )
            component_log_survival = torch.log(
                (0.5 - 0.5 * torch.erf(standardized)).clamp(min=1e-10)
            )
        else:
            raise ValueError(f"Unknown distribution: {model.dist}")
        log_weights = torch.log_softmax(logits, dim=2).unsqueeze(3)
        output[start:end] = (
            torch.exp(torch.logsumexp(log_weights + component_log_survival, dim=2))
            .float()
            .cpu()
        )
    return output


class DSM:
    """Deep Survival Machines for competing risks.

    Models each cause-specific event time distribution as a mixture of M
    parametric distributions (Weibull or LogNormal). Parameters and mixture
    weights are covariate-dependent via a neural network.

    CIF is computed as: CIF_k(t|x) = 1 - S_k(t|x), where S_k is the
    cause-specific survival function from the DSM mixture model for risk k.

    Note: In the single-risk DSM, predict_risk returns 1 - S(t|x). For
    competing risks, each risk's model gives a cause-specific CDF.

    Args:
        n_causes: number of competing event types (K)
        n_features: number of input features (p)
        k: number of mixture components per cause
        layers: hidden layer widths for shared representation
        distribution: "Weibull" or "LogNormal"
    """

    def __init__(
        self,
        n_causes: int,
        n_features: int,
        k: int = 6,
        layers: list[int] | None = None,
        distribution: str = "Weibull",
    ) -> None:
        self.n_causes = n_causes
        self.n_features = n_features
        self.k = k
        self.layers = layers if layers is not None else [64, 64]
        self.distribution = distribution
        self.net: DSMNet | None = None

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> DSM:
        """Reconstruct a trained DSM model from a checkpoint.

        Args:
            data: checkpoint dict with keys: state_dict, n_causes,
                  n_features, k, layers, distribution

        Returns:
            A DSM instance ready for predict_cif()
        """
        model = cls(
            n_causes=data["n_causes"],
            n_features=data["n_features"],
            k=data["k"],
            layers=data["layers"],
            distribution=data["distribution"],
        )
        model.net = DSMNet(
            input_dim=data["n_features"],
            n_causes=data["n_causes"],
            k=data["k"],
            layers=data["layers"],
            distribution=data["distribution"],
        ).double()
        model.net.load_state_dict(data["state_dict"])
        model.net.eval()
        return model

    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        Delta_train: Tensor,
        X_val: Tensor,
        Y_val: Tensor,
        Delta_val: Tensor,
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 256,
        patience: int = 50,
        device: torch.device | str | None = None,
    ) -> list[float]:
        """Train the DSM model.

        Args:
            X_train: training covariates (n_train, p)
            Y_train: training observed times (n_train,)
            Delta_train: training event indicators (n_train,), 0=censored, 1..K=event
            X_val: validation covariates
            Y_val: validation observed times
            Delta_val: validation event indicators
            epochs: maximum training epochs
            lr: learning rate
            batch_size: mini-batch size
            patience: early stopping patience

        Returns:
            list of training losses per epoch
        """
        dev = torch.device(device) if device is not None else torch.device("cpu")

        self.net = (
            DSMNet(
                input_dim=self.n_features,
                n_causes=self.n_causes,
                k=self.k,
                layers=self.layers,
                distribution=self.distribution,
            )
            .double()
            .to(dev)
        )

        assert self.net is not None
        net = self.net

        x_t = X_train.double().to(dev)
        t_t = Y_train.double().to(dev)
        e_t = Delta_train.double().to(dev)

        x_v = X_val.double().to(dev)
        t_v = Y_val.double().to(dev)
        e_v = Delta_val.double().to(dev)

        dataset = TensorDataset(x_t, t_t, e_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        losses: list[float] = []

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, t_batch, e_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _dsm_all_causes_loss(
                    net,
                    x_batch,
                    t_batch,
                    e_batch,
                    elbo=True,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            # Validation loss for early stopping
            net.eval()
            with torch.no_grad():
                val_loss = _dsm_all_causes_loss(
                    net,
                    x_v,
                    t_v,
                    e_v,
                    elbo=False,
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"  DSM: early stopping at epoch {epoch + 1} "
                    f"(val_loss={best_val_loss:.6f})"
                )
                break

            if (epoch + 1) % 50 == 0:
                print(
                    f"  DSM epoch {epoch + 1:4d}/{epochs}, "
                    f"loss: {avg_loss:.6f}, val_loss: {val_loss:.6f}"
                )

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

        CIF_k(t|x) = 1 - S_k(t|x), where S_k is the cause-specific survival
        function from the DSM mixture model.

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
        """Predict CIF and latent-independent overall survival."""
        assert self.net is not None
        net = self.net

        net.eval()
        dev = next(net.parameters()).device
        x_t = X.double().to(dev)
        with torch.no_grad():
            cause_survival = _dsm_predict_all_cause_survival(net, x_t, times)

        cif = 1.0 - cause_survival
        survival = cause_survival.prod(dim=1)
        return cif, survival
