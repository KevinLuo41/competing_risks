#!/usr/bin/env python3
"""
Case I (Linear) — evaluate all 5 models against the known true CIF.

mu_k(x, t) = intercept_k + beta_k^T x + alpha * t   (alpha shared across causes)

Two-tier caching for fast iteration: see ../runner.py.
CRSoft is the iteration target (eval cache never read).

CLI:
  --retrain MODEL [MODEL ...]    force retrain (also invalidates eval cache)
  --reeval  MODEL [MODEL ...]    force re-eval (re-predict + re-evaluate)
  --list                        show cache state and exit

To change CRSoft hyperparameters, edit `_train_crsoft` below and rerun
with `--retrain crsoft`.
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

from ...baseline_models import CsCox, DeepHit, DSM, NeuralFG
from ...crsoft_model import CRSoftNet
from ...data.case1 import compute_cif, generate_data
from ...data.utils import generate_test_observations
from ...evaluation import (
    build_evaluation_time_grid,
    compute_mse_accuracy,
    evaluate_cif_metrics,
    get_output_dir,
    isotonic_project_cif,
    plot_cif_comparison,
    plot_event_time_distribution,
    plot_training_loss,
    print_data_summary,
    print_example_predictions,
    save_results_txt,
)
from ..runner import (
    delete_cache,
    ModelSpec,
    parse_args,
    print_cache_state,
    print_plan,
    print_results,
    resolve_models,
    run_or_load,
)

OUTPUT_DIR = get_output_dir("case1")
CASE_NAME = "case1"
CASE_LABEL = "Case I (Linear)"

N_TRAIN = 5000
N_TEST = 1000
K = 3
P = 5
SEED_TRAIN = 42
SEED_TEST = 123


# =====================================================================
# Data prep (case-specific)
# =====================================================================
def _prepare_data() -> tuple:
    print(f"\nGenerating training data (n={N_TRAIN})...")
    train_data = generate_data(n=N_TRAIN, K=K, p=P, seed=SEED_TRAIN)

    X_train = train_data["X"]
    Y_train = train_data["Y"]
    Delta_train = train_data["Delta"]

    n_val = int(N_TRAIN * 0.1)
    X_val, Y_val, Delta_val = X_train[:n_val], Y_train[:n_val], Delta_train[:n_val]
    X_train_fit = X_train[n_val:]
    Y_train_fit = Y_train[n_val:]
    Delta_train_fit = Delta_train[n_val:]

    true_cif_fn = functools.partial(
        compute_cif,
        beta=train_data["beta"],
        alpha=train_data["alpha"],
        intercept=train_data["intercept"],
    )

    print(f"Generating test data (n={N_TEST})...")
    X_test, Y_test, Delta_test = generate_test_observations(
        N_TEST, P, true_cif_fn, censor_rate=0.5, seed=SEED_TEST
    )

    print_data_summary(train_data, K, P)
    print("\nPlotting data distribution...")
    plot_event_time_distribution(train_data, K, output_dir=OUTPUT_DIR)

    eval_times = build_evaluation_time_grid(Y_test, Delta_test, n_grid=100)
    print(
        f"\n  Evaluation time grid: {len(eval_times)} points, "
        f"range [{eval_times[0]:.2f}, {eval_times[-1]:.2f}]"
    )
    return (
        train_data,
        X_train_fit,
        Y_train_fit,
        Delta_train_fit,
        X_val,
        Y_val,
        Delta_val,
        X_test,
        Y_test,
        Delta_test,
        eval_times,
        true_cif_fn,
    )


# =====================================================================
# Build per-model specs (case-specific: hyperparams, train data wiring)
# =====================================================================
def _build_specs(
    X_train_fit: Tensor,
    Y_train_fit: Tensor,
    Delta_train_fit: Tensor,
    X_val: Tensor,
    Y_val: Tensor,
    Delta_val: Tensor,
    X_train_full: Tensor,
    Y_train_full: Tensor,
    Delta_train_full: Tensor,
    eval_times: Tensor,
    device: torch.device | str | None = None,
) -> list[ModelSpec]:
    def _train_deephit() -> DeepHit:
        print("Training DeepHit...")
        m = DeepHit(
            n_bins=100,
            n_causes=K,
            hidden_dim=64,
            n_layers=2,
            lr=1e-3,
            batch_size=256,
            epochs=500,
            patience=50,
            alpha=0.2,
            sigma=0.1,
        )
        m.fit(
            X_train_fit,
            Y_train_fit,
            Delta_train_fit,
            X_val,
            Y_val,
            Delta_val,
            verbose=True,
            device=device,
        )
        return m

    def _train_dsm() -> DSM:
        print("Training DSM...")
        m = DSM(n_causes=K, n_features=P, k=6, layers=[64, 64], distribution="Weibull")
        m.fit(
            X_train_fit,
            Y_train_fit,
            Delta_train_fit,
            X_val,
            Y_val,
            Delta_val,
            epochs=500,
            lr=1e-3,
            batch_size=256,
            patience=50,
            device=device,
        )
        return m

    def _train_cscox() -> CsCox:
        print("Training cs-Cox...")
        m = CsCox(n_causes=K)
        m.fit(X_train_fit, Y_train_fit, Delta_train_fit)
        return m

    def _train_nfg() -> NeuralFG:
        print("Training NeuralFineGray...")
        m = NeuralFG(
            n_causes=K,
            layers=[100, 100, 100],
            layers_surv=[100],
            lr=1e-3,
            batch_size=100,
            epochs=1000,
            patience=3,
            weight_decay=0.001,
        )
        m.fit(
            X_train_fit,
            Y_train_fit,
            Delta_train_fit,
            X_val,
            Y_val,
            Delta_val,
            verbose=True,
            device=device,
        )
        return m

    def _train_crsoft() -> CRSoftNet:
        # ---- CRSoft hyperparameters (tuned for new case1 data) ----
        # n_aug=1 picked by sweep: MSE 0.0083 -> 0.0027 (3x), beats NeuralFG (2nd place)
        hidden_dim, num_blocks = 16, 1
        epochs, lr, weight_decay = 1000, 1e-3, 3e-3
        n_aug, aug_weight = 1, 0.5
        # ---------------------------------------------------------------------
        print(
            f"Training CRSoft (h{hidden_dim}_b{num_blocks}, n_aug={n_aug}, "
            f"wd={weight_decay:.0e})..."
        )
        torch.manual_seed(0)
        m = CRSoftNet(
            input_dim=P,
            num_causes=K,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        losses = m.fit(
            X_train_full,
            Y_train_full,
            Delta_train_full,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            n_aug=n_aug,
            aug_weight=aug_weight,
            verbose=True,
            device=device,
        )
        m._train_losses = losses  # pyre-ignore[16]
        m._hyperparams = {
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
        }  # pyre-ignore[16]
        return m

    return [
        ModelSpec(
            name="DeepHit",
            cache_key="deephit",
            cli_aliases=["deephit"],
            train=_train_deephit,
            serialize=lambda m: {
                "state_dict": m.net.state_dict(),
                "bin_edges": m.bin_edges,
                "n_bins": m.n_bins,
                "n_causes": m.n_causes,
                "hidden_dim": m.hidden_dim,
                "n_layers": m.n_layers,
            },
            load_ckpt=DeepHit.load_from_checkpoint,
            predict=lambda m, X, t: m.predict_cif(X, t),
            predict_survival=lambda m, X, t: m.predict_cif_survival(X, t),
        ),
        ModelSpec(
            name="DSM",
            cache_key="dsm",
            cli_aliases=["dsm"],
            train=_train_dsm,
            serialize=lambda m: {
                "state_dict": m.net.state_dict(),
                "n_causes": m.n_causes,
                "n_features": m.n_features,
                "k": m.k,
                "layers": m.layers,
                "distribution": m.distribution,
            },
            load_ckpt=DSM.load_from_checkpoint,
            predict=lambda m, X, t: m.predict_cif(X, t),
            predict_survival=lambda m, X, t: m.predict_cif_survival(X, t),
        ),
        ModelSpec(
            name="cs-Cox",
            cache_key="cs_cox",
            cli_aliases=["cscox", "cox"],
            train=_train_cscox,
            serialize=lambda m: {
                "models": m.models,
                "baseline_hazards": m.baseline_hazards,
                "n_causes": m.n_causes,
            },
            load_ckpt=CsCox.load_from_checkpoint,
            predict=lambda m, X, t: m.predict_cif(X, t),
            predict_survival=lambda m, X, t: m.predict_cif_survival(X, t),
        ),
        ModelSpec(
            name="NeuralFG",
            cache_key="neural_fg",
            cli_aliases=["neuralfg", "nfg"],
            train=_train_nfg,
            serialize=lambda m: {
                "state_dict": m.net.state_dict(),
                "n_causes": m.n_causes,
                "layers": m.layers,
                "layers_surv": m.layers_surv,
                "_time_max": m._time_max,
            },
            load_ckpt=NeuralFG.load_from_checkpoint,
            predict=lambda m, X, t: m.predict_cif(X, t),
            predict_survival=lambda m, X, t: m.predict_cif_survival(X, t),
        ),
        ModelSpec(
            name="CRSoft",
            cache_key="crsoft",
            cli_aliases=["crsoft", "softcomp"],
            train=_train_crsoft,
            serialize=lambda m: {
                "state_dict": m.state_dict(),
                "num_causes": m.num_causes,
                "input_dim": P,
                "hidden_dim": m._hyperparams["hidden_dim"],
                "num_blocks": m._hyperparams["num_blocks"],
            },
            load_ckpt=CRSoftNet.load_from_checkpoint,
            predict=lambda m, X, t: m.predict_cif_grid(X, t),
            predict_survival=lambda m, X, t: m.predict_cif_survival_grid(X, t),
            post_process=lambda c: isotonic_project_cif(c, eval_times),
            is_iteration_target=False,  # settled — cache eval like other models
        ),
    ]


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    print("=" * 70)
    print(CASE_LABEL)
    print("=" * 70)

    # Build a stub spec list for CLI token validation (cheap; no data needed)
    stub_specs = _build_specs(*[torch.empty(0)] * 9, torch.empty(0))
    args = parse_args(prog_desc=f"{CASE_LABEL} run controller")

    if args.list:
        print_cache_state(CASE_NAME, stub_specs)
        return

    retrain_set = resolve_models(args.retrain, stub_specs)
    reeval_set = resolve_models(args.reeval, stub_specs)

    for ck in retrain_set:
        print(f"--retrain {ck}: clearing caches")
        delete_cache(CASE_NAME, ck, ("model", "eval"))
    for ck in reeval_set:
        print(f"--reeval {ck}: clearing eval cache")
        delete_cache(CASE_NAME, ck, ("eval",))

    print_plan(CASE_NAME, stub_specs)

    # ---- Prepare data + real specs ----
    (
        train_data,
        X_train_fit,
        Y_train_fit,
        Delta_train_fit,
        X_val,
        Y_val,
        Delta_val,
        X_test,
        Y_test,
        Delta_test,
        eval_times,
        true_cif_fn,
    ) = _prepare_data()

    specs = _build_specs(
        X_train_fit,
        Y_train_fit,
        Delta_train_fit,
        X_val,
        Y_val,
        Delta_val,
        train_data["X"],
        train_data["Y"],
        train_data["Delta"],
        eval_times,
    )

    # eval_fn closes over test/train data + true_cif_fn (sim-case includes MSE+accuracy)
    Y_train, Delta_train = train_data["Y"], train_data["Delta"]

    def eval_fn(cif: Tensor) -> dict[str, float]:
        return {
            **compute_mse_accuracy(cif, X_test, eval_times, true_cif_fn, K),
            **evaluate_cif_metrics(
                cif, Y_test, Delta_test, Y_train, Delta_train, eval_times, K
            ),
        }

    all_results: list[tuple[str, dict[str, float]]] = []
    cifs: dict[str, Tensor] = {}
    crsoft_model: CRSoftNet | None = None

    for spec in specs:
        force_reeval = spec.cache_key in reeval_set or spec.cache_key in retrain_set
        model, cif, metrics = run_or_load(
            CASE_NAME,
            spec,
            X_test,
            eval_times,
            eval_fn,
            force_reeval=force_reeval,
        )
        all_results.append((spec.name, metrics))
        cifs[spec.name] = cif
        if spec.name == "CRSoft" and isinstance(model, CRSoftNet):
            crsoft_model = model

    # ---- CRSoft-specific extras ----
    if crsoft_model is not None:
        losses = getattr(crsoft_model, "_train_losses", None)
        if losses:
            print("\nPlotting CRSoft training loss...")
            plot_training_loss(losses, output_dir=OUTPUT_DIR)
        print("\nCRSoft example predictions:")
        print_example_predictions(crsoft_model, X_test, true_cif_fn, K)

    # ---- Print + save ----
    print_results(CASE_LABEL, K, all_results)
    save_results_txt(CASE_NAME, all_results)

    # ---- CIF comparison plot ----
    print("\nPlotting CIF comparison (all models + true)...")
    n_test = X_test.shape[0]
    cif_true = torch.zeros(n_test, K, len(eval_times))
    with torch.no_grad():
        for ti in range(len(eval_times)):
            f_true, _ = true_cif_fn(X_test, eval_times[ti].expand(n_test))
            cif_true[:, :, ti] = f_true
    model_cifs = {"True": cif_true, **cifs}
    plot_cif_comparison(model_cifs, eval_times, K, output_dir=OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
