"""Population minimizers of SoftComp's current training losses vs. the true CIF.

Stdlib only; run with `python3 population_targets.py`.

Censoring is defined relative to the fixed horizon TAU: exponential censoring at rate
lam_c with P(C < min(T, TAU)) = rho, and follow-up ends at TAU, so Y = min(T, C, TAU).
For a network that is flexible in t, each risk is minimized pointwise in t < TAU:
  CE at Y with label Delta:            pi_k(t) = lam_k / (lam + lam_c)
  CE + augmentation (U ~ Unif(0, Y)):  pi_k(t) = lam_k / (lam + lam_c + aug * E[1/Y | Y > t])
  unweighted Brier:                    pi_k(t) = P(Y <= t, Delta = k) = int_0^t G dF_k
Mapping the observed-state probabilities through the Aalen-Johansen identity
returns the true F_k for every censoring level (JointSoftComp).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

Fn = Callable[[float], float]
TAU = 20.0
RHOS = (0.0, 0.2, 0.5, 0.8)


def simpson(f: Fn, a: float, b: float, n: int = 4000) -> float:
    if b <= a:
        return 0.0
    n += n % 2
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def censoring_rate(censored_fraction: Fn, rho: float) -> float:
    """lam_c with censored_fraction(lam_c) = rho; censored_fraction increases in lam_c."""
    if rho == 0.0:
        return 0.0
    lo, hi = 1e-8, 100.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if censored_fraction(mid) < rho else (lo, mid)
    return math.sqrt(lo * hi)


def loss_targets(
    cum_haz: Fn,
    haz: Fn,
    share: float,
    lam_c: float,
    t: float,
    aug: float,
) -> dict[str, float]:
    def obs_surv(s: float) -> float:
        return math.exp(-cum_haz(s) - lam_c * s)

    def obs_dens(s: float) -> float:
        return (haz(s) + lam_c) * obs_surv(s)

    # Y has a density on (0, TAU) and an atom of size obs_surv(TAU) at TAU.
    tail = simpson(lambda y: obs_dens(y) / y, t, TAU) + obs_surv(TAU) / TAU
    inv_y = tail / obs_surv(t)
    return {
        "true": share * (1 - math.exp(-cum_haz(t))),
        "ce": share * haz(t) / (haz(t) + lam_c),
        "ce_aug": share * haz(t) / (haz(t) + lam_c + aug * inv_y),
        "brier": simpson(lambda s: share * haz(s) * obs_surv(s), 0.0, t),
    }


def aalen_johansen_from_observed(
    cum_haz: Fn, haz: Fn, share: float, lam_c: float, t: float, n_grid: int = 4000
) -> float:
    """Map observed-state probabilities (P_0, P_k) to F_k via dLambda_k = dP_k / P_0(-)."""
    h = t / n_grid
    surv, cif, p_k_prev, p0_prev = 1.0, 0.0, 0.0, 1.0
    for j in range(1, n_grid + 1):
        s0, s1 = (j - 1) * h, j * h
        p_k = p_k_prev + simpson(
            lambda s: share * haz(s) * math.exp(-cum_haz(s) - lam_c * s), s0, s1, 8
        )
        p0 = math.exp(-cum_haz(s1) - lam_c * s1)
        d_lam_k = (p_k - p_k_prev) / p0_prev
        cif += surv * d_lam_k
        surv *= 1 - d_lam_k / share
        p_k_prev, p0_prev = p_k, p0
    return cif


def print_table(
    title: str,
    hazards: tuple[Fn, Fn],
    share: float,
    lam_cs: dict[str, float],
    times: list[float],
    aug: float,
) -> None:
    cum_haz, haz = hazards
    print(f"\n=== {title}  (lambda_aug * M = {aug:g}, tau = {TAU:g})")
    print(
        f"{'rho':>5} {'t':>5} {'true':>7} {'CE':>7} {'CE+aug':>7} {'Brier':>7} {'A(AJ)':>7}"
    )
    for label, lam_c in lam_cs.items():
        for t in times:
            r = loss_targets(cum_haz, haz, share, lam_c, t, aug)
            aj = aalen_johansen_from_observed(cum_haz, haz, share, lam_c, t)
            print(
                f"{label:>5} {t:5.1f} {r['true']:7.3f} {r['ce']:7.3f} "
                f"{r['ce_aug']:7.3f} {r['brier']:7.3f} {aj:7.3f}"
            )


def constant_hazard_example() -> None:
    lam1, lam2 = 0.10, 0.05
    lam = lam1 + lam2

    def censored_fraction(lam_c: float) -> float:
        return lam_c / (lam + lam_c) * (1 - math.exp(-(lam + lam_c) * TAU))

    lam_cs = {f"{p:.0%}": censoring_rate(censored_fraction, p) for p in RHOS}
    print(f"\nConstant hazards: lambda_c = {lam_cs}")
    for aug in (1.0, 4.0):
        print_table(
            "Constant hazards (lam1=0.10, lam2=0.05), cause 1",
            (lambda s: lam * s, lambda s: lam),
            lam1 / lam,
            lam_cs,
            [1.0, 5.0, 10.0, 15.0, 20.0],
            aug,
        )


def case3_hazards(x: list[float]) -> tuple[Fn, Fn]:
    q = min(abs(x[3]), 2.0)
    c1, c2 = 0.02 + 0.78 * q, 0.04 + 1.18 * (2 - q)
    scale = math.exp(0.20 * x[0])

    def cum_haz(t: float) -> float:
        u = t / 16
        return scale * (c1 * u + c2 * u * u)

    def haz(t: float) -> float:
        return scale * (c1 + 2 * c2 * t / 16) / 16

    return cum_haz, haz


def case3_router(x: list[float]) -> list[float]:
    dirs = [(1.0, 0.0), (-0.5, math.sqrt(3) / 2), (-0.5, -math.sqrt(3) / 2)]
    s = [2 * abs(v[0] * x[1] + v[1] * x[2]) for v in dirs]
    e = [math.exp(v - max(s)) for v in s]
    return [v / sum(e) for v in e]


def case3_follow_up(n: int = 50_000, seed: int = 1) -> list[float]:
    """min(T, TAU) for a sample from the Case III covariate and event-time distribution."""
    rng = random.Random(seed)
    follow_up = []
    for _ in range(n):
        x = [rng.gauss(0, 1) for _ in range(4)]
        q = min(abs(x[3]), 2.0)
        c1, c2 = 0.02 + 0.78 * q, 0.04 + 1.18 * (2 - q)
        e = rng.expovariate(1.0) / math.exp(0.20 * x[0])
        t = 16 * (-c1 + math.sqrt(c1 * c1 + 4 * c2 * e)) / (2 * c2)
        follow_up.append(min(t, TAU))
    return follow_up


def case3_example() -> None:
    follow_up = case3_follow_up()

    def censored_fraction(lam_c: float) -> float:
        return sum(1 - math.exp(-lam_c * u) for u in follow_up) / len(follow_up)

    lam_cs = {f"{p:.0%}": censoring_rate(censored_fraction, p) for p in RHOS}
    print(f"\nCase III: lambda_c = {lam_cs}")
    for x in ([0.0, 1.0, 0.2, 0.3], [0.0, 1.0, 0.2, 1.7]):
        shares = case3_router(x)
        k = max(range(3), key=lambda i: shares[i])
        print_table(
            f"Case III x={x}, cause {k + 1} (pi_k={shares[k]:.3f})",
            case3_hazards(x),
            shares[k],
            lam_cs,
            [2.0, 5.0, 10.0, 15.0, 20.0],
            1.0,
        )


def main() -> None:
    constant_hazard_example()
    case3_example()


if __name__ == "__main__":
    main()
