#!/usr/bin/env python3
"""Case III v5 data generation with folded routing and time interactions.

Let ``X ~ N_4(0, I_4)``.  Cause allocation follows a folded angular
multinomial router,

    s_k(x) = d_k + delta |v_k^T (x_2, x_3)|,
    pi_k(x) = softmax_k(s(x)),

where the three ``v_k`` are equally spaced unit vectors.  Each absolute
projection is exactly representable as ``ReLU(z) + ReLU(-z)``.  Unlike the
quadratic and pairwise static score in Case II, this router is a single folded
angular layer and affects only conditional cause allocation.  Event timing is
controlled separately by the cumulative hazard

    q(x) = min(|x_4|, 2),
    u = t / tau,
    H(t, x) = exp{a x_1 + gamma [q(x) - q_0]}
              [c_1(q) u + c_2(q) u^2],
    c_1(q) = c_0 + b q,
    c_2(q) = d_0 + d_1 [q_max - q].

The true probabilities are

    F_k(t | x) = pi_k(x) [1 - exp{-H(t, x)}],
    S(t | x) = exp{-H(t, x)}.

The default parameters make both time coefficients strictly positive.  Hence
``H(0, x) = 0`` and

    dH(t, x) / dt
      = exp{a x_1 + gamma [q(x) - q_0]}
        [c_1(q) + 2 c_2(q) u] / tau > 0,

so every CIF is monotone and the probabilities sum to one.  Excluding the
optional persistent acceleration, the folded contribution to ``dH/dt`` is
proportional to ``b - 2 d_1 u`` and changes sign at
``t = tau b / (2 d_1)``.  Its contribution to ``H`` is proportional to
``u (b - d_1 u)`` and changes sign later, at ``t = tau b / d_1``.  The
optional ``gamma`` term adds a persistent folded acceleration at every time
while preserving positivity and monotonicity; it defaults to zero.  Both
effects act through event timing, isolating the dynamic, non-proportional
interaction from the folded cause-allocation model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .utils import assert_monotone_cif, assign_causes_and_censor, solve_inverse_cdf

K = 3
P = 4
T_MAX = 150.0
INTERACTION_EARLY_TIME = 4.0
INTERACTION_TRANSITION_TIME = 16.0 * 0.78 / (2.0 * 1.18)
CUMULATIVE_HAZARD_CROSSING_TIME = 16.0 * 0.78 / 1.18
INTERACTION_LATE_TIME = 18.0


def compute_static_logits(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compute the folded angular logits of the cause-allocation router."""
    router_features = x[:, 1:3]
    return (
        params["router_intercept"].unsqueeze(0)
        + params["router_weight"]
        * (router_features @ params["router_directions"].T).abs()
    )


def compute_cause_probabilities(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compute the time-invariant conditional cause probabilities."""
    return torch.softmax(compute_static_logits(x, params), dim=1)


def compute_folded_projection(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compute ``q(x) = min(|x_4|, projection_cap)``."""
    return x[:, 3].abs().clamp(max=params["projection_cap"])


def compute_folded_index(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compatibility alias for :func:`compute_folded_projection`."""
    return compute_folded_projection(x, params)


def compute_time_coefficients(
    x: Tensor,
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Return the positive linear and quadratic time coefficients."""
    folded_index = compute_folded_projection(x, params)
    linear = (
        params["linear_coefficient_floor"]
        + params["linear_coefficient_weight"] * folded_index
    )
    quadratic = params["quadratic_coefficient_floor"] + params[
        "quadratic_coefficient_weight"
    ] * (params["projection_cap"] - folded_index)
    return linear, quadratic


def compute_hazard_crossing_time(params: dict[str, Tensor]) -> Tensor:
    """Return where the time-coefficient effect on ``dH/dt`` changes sign."""
    return (
        params["time_scale"]
        * params["linear_coefficient_weight"]
        / (2.0 * params["quadratic_coefficient_weight"])
    )


def compute_cumulative_hazard_crossing_time(
    params: dict[str, Tensor],
) -> Tensor:
    """Return where the time-coefficient effect on ``H`` changes sign."""
    return (
        params["time_scale"]
        * params["linear_coefficient_weight"]
        / params["quadratic_coefficient_weight"]
    )


def compute_acceleration(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compute the positive shared and persistent folded acceleration."""
    folded_projection = compute_folded_projection(x, params)
    log_acceleration = params["acceleration_weight"] * x[:, 0] + params[
        "folded_acceleration_weight"
    ] * (folded_projection - params["folded_acceleration_center"])
    return torch.exp(log_acceleration)


def _expand_time(t: Tensor, n_subjects: int) -> Tensor:
    if t.dim() == 0:
        return t.expand(n_subjects)
    if t.dim() != 1 or len(t) != n_subjects:
        raise ValueError("t must be scalar or contain one value per subject")
    return t


def compute_cumulative_hazard(
    x: Tensor,
    t: Tensor,
    params: dict[str, Tensor],
) -> Tensor:
    """Compute the nonnegative cumulative event hazard ``H(t, x)``."""
    expanded_time = _expand_time(t, x.shape[0]).clamp(min=0.0)
    scaled_time = expanded_time / params["time_scale"]
    linear, quadratic = compute_time_coefficients(x, params)
    acceleration = compute_acceleration(x, params)
    return acceleration * (linear * scaled_time + quadratic * scaled_time.square())


def compute_cumulative_hazard_derivative(
    x: Tensor,
    t: Tensor,
    params: dict[str, Tensor],
) -> Tensor:
    """Compute the strictly positive right derivative of ``H`` in time."""
    expanded_time = _expand_time(t, x.shape[0]).clamp(min=0.0)
    scaled_time = expanded_time / params["time_scale"]
    linear, quadratic = compute_time_coefficients(x, params)
    acceleration = compute_acceleration(x, params)
    return (
        acceleration * (linear + 2.0 * quadratic * scaled_time) / params["time_scale"]
    )


def compute_cif(
    x: Tensor,
    t: Tensor,
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute the Case III v5 CIF and survival probabilities."""
    cumulative_hazard = compute_cumulative_hazard(x, t, params)
    survival = torch.exp(-cumulative_hazard)
    total_event_probability = -torch.expm1(-cumulative_hazard)
    cif = compute_cause_probabilities(x, params) * total_event_probability.unsqueeze(1)
    return cif, survival


def generate_parameters(
    K: int = K,
    p: int = P,
    seed: int = 42,
    router_weight: float = 2.0,
    acceleration_weight: float = 0.20,
    folded_acceleration_weight: float = 0.0,
    folded_acceleration_center: float = 0.75,
    time_scale: float = 16.0,
    projection_cap: float = 2.0,
    linear_coefficient_floor: float = 0.02,
    linear_coefficient_weight: float = 0.78,
    quadratic_coefficient_floor: float = 0.04,
    quadratic_coefficient_weight: float = 1.18,
) -> dict[str, Tensor]:
    """Return the current pre-registration candidate parameters.

    ``seed`` is accepted for protocol compatibility.  The truth is fully
    deterministic and contains no randomly drawn coefficients.
    """
    if (K, p) != (3, 4):
        raise ValueError("Case III v5 requires K=3 and p=4")
    if router_weight <= 0 or acceleration_weight <= 0:
        raise ValueError("router and acceleration weights must be positive")
    if not math.isfinite(folded_acceleration_weight):
        raise ValueError("folded acceleration weight must be finite")
    if folded_acceleration_weight < 0:
        raise ValueError("folded acceleration weight must be non-negative")
    if time_scale <= 0 or projection_cap <= 0:
        raise ValueError("time scale and projection cap must be positive")
    if not 0 <= folded_acceleration_center <= projection_cap:
        raise ValueError("folded acceleration center must lie within the projection")
    if linear_coefficient_floor <= 0 or linear_coefficient_weight <= 0:
        raise ValueError("time coefficients must be positive")
    if quadratic_coefficient_floor <= 0 or quadratic_coefficient_weight <= 0:
        raise ValueError("quadratic time coefficients must be positive")

    del seed
    angles = torch.arange(K, dtype=torch.float32) * (2.0 * math.pi / K)
    return {
        "router_intercept": torch.zeros(K),
        "router_directions": torch.stack([torch.cos(angles), torch.sin(angles)], dim=1),
        "router_weight": torch.tensor(router_weight),
        "acceleration_weight": torch.tensor(acceleration_weight),
        "folded_acceleration_weight": torch.tensor(folded_acceleration_weight),
        "folded_acceleration_center": torch.tensor(folded_acceleration_center),
        "time_scale": torch.tensor(time_scale),
        "projection_cap": torch.tensor(projection_cap),
        "linear_coefficient_floor": torch.tensor(linear_coefficient_floor),
        "linear_coefficient_weight": torch.tensor(linear_coefficient_weight),
        "quadratic_coefficient_floor": torch.tensor(quadratic_coefficient_floor),
        "quadratic_coefficient_weight": torch.tensor(quadratic_coefficient_weight),
    }


def generate_data(
    n: int = 5000,
    K: int = K,
    p: int = P,
    censor_rate: float = 0.5,
    seed: int = 42,
    params: dict[str, Tensor] | None = None,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Generate one Case III v5 subject sample and interaction diagnostics."""
    torch.manual_seed(seed)
    if params is None:
        params = generate_parameters(K=K, p=p, seed=seed)
    if (K, p) != (3, 4):
        raise ValueError("Case III v5 requires K=3 and p=4")
    if params["router_intercept"].shape != (K,):
        raise ValueError("router_intercept has the wrong shape")
    if params["router_directions"].shape != (K, 2):
        raise ValueError("router_directions has the wrong shape")

    x = torch.randn(n, p)
    assert_monotone_cif(
        lambda values, times: compute_cif(values, times, params),
        x[:32],
        torch.linspace(0.0, T_MAX, 151),
    )
    t_event = solve_inverse_cdf(
        x,
        torch.rand(n),
        lambda values, times: compute_cif(values, times, params),
        t_max=T_MAX,
    )
    y, delta, epsilon = assign_causes_and_censor(
        x,
        t_event,
        lambda values, times: compute_cif(values, times, params),
        censor_rate,
    )
    return {
        "X": x,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": epsilon,
        "cause_probabilities": compute_cause_probabilities(x, params),
        "acceleration": compute_acceleration(x, params),
        "folded_projection": compute_folded_projection(x, params),
        "folded_index": compute_folded_projection(x, params),
        "hazard_rate_early": compute_cumulative_hazard_derivative(
            x, torch.tensor(INTERACTION_EARLY_TIME), params
        ),
        "hazard_rate_transition": compute_cumulative_hazard_derivative(
            x, torch.tensor(INTERACTION_TRANSITION_TIME), params
        ),
        "hazard_rate_late": compute_cumulative_hazard_derivative(
            x, torch.tensor(INTERACTION_LATE_TIME), params
        ),
        "params": params,
    }
