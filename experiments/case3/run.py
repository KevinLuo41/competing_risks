#!/usr/bin/env python3
"""
Case III (Functional Covariates) — evaluate CRSoft against the known true CIF.

mu_k(X, t) = intercept_k + sum_j <X_j, beta_{k,j}> + alpha * t   (alpha shared)
where X_j(s) are functional covariates and beta_{k,j}(s) are functional effects.

Only CRSoft (FunctionalCRSoftNet) is used since baselines don't support
functional covariates. Two-tier caching for fast iteration: see ../runner.py.

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

from ...crsoft_model import FunctionalCRSoftNet
from ...data.case3 import (
    _generate_functional_covariates,
    compute_cif,
    generate_data,
    N_BASIS,
)
from ...data.utils import assign_causes_and_censor, solve_inverse_cdf
from ...evaluation import (
    build_evaluation_time_grid,
    compute_mse_accuracy,
    evaluate_cif_metrics,
    get_output_dir,
    isotonic_project_cif,
    plot_cif_comparison,
    plot_event_time_distribution,
    plot_functional_pipeline,
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

OUTPUT_DIR = get_output_dir("case3")
CASE_NAME = "case3"
CASE_LABEL = "Case III (Functional Covariates)"

N_TRAIN = 5000
N_TEST = 1000
K = 2  # number of competing causes
P = 3  # number of functional covariates
N_GRID = 50  # grid points per functional covariate
SEED_TRAIN = 42
SEED_TEST = 123


# =====================================================================
# Data prep (case-specific): functional covariates have their own
# generators (not the scalar generate_test_observations helper).
# =====================================================================
def _prepare_data() -> tuple:
    print(f"\nGenerating training data (n={N_TRAIN})...")
    train_data = generate_data(n=N_TRAIN, K=K, p=P, n_grid=N_GRID, seed=SEED_TRAIN)

    beta_func = train_data["beta_func"]
    alpha_true = train_data["alpha"]
    intercept_true = train_data["intercept"]
    true_cif_fn = functools.partial(
        compute_cif,
        beta_func=beta_func,
        alpha=alpha_true,
        intercept=intercept_true,
        n_grid=N_GRID,
    )

    print(f"Generating test data (n={N_TEST})...")
    torch.manual_seed(SEED_TEST)
    X_test = _generate_functional_covariates(N_TEST, P, N_GRID, N_BASIS)
    Y_test, Delta_test, _ = assign_causes_and_censor(
        X_test,
        solve_inverse_cdf(X_test, torch.rand(N_TEST), true_cif_fn),
        true_cif_fn,
        censor_rate=0.3,
    )

    print_data_summary(train_data, K, P)
    print(f"\n  Grid points:         {N_GRID}")
    print(f"  X_func shape:        {train_data['X_func'].shape}")

    print("\nPlotting data distribution...")
    plot_event_time_distribution(train_data, K, output_dir=OUTPUT_DIR)

    eval_times = build_evaluation_time_grid(
        train_data["Y"], train_data["Delta"], n_grid=100
    )
    print(
        f"\n  Evaluation time grid: {len(eval_times)} points, "
        f"range [{eval_times[0]:.2f}, {eval_times[-1]:.2f}]"
    )
    return (
        train_data,
        X_test,
        Y_test,
        Delta_test,
        eval_times,
        true_cif_fn,
    )


# =====================================================================
# Build per-model specs (case-specific: only CRSoft for functional covariates)
# =====================================================================
def _build_specs(
    X_train_full: Tensor,
    Y_train_full: Tensor,
    Delta_train_full: Tensor,
    eval_times: Tensor,
) -> list[ModelSpec]:
    def _train_crsoft() -> FunctionalCRSoftNet:
        # ---- CRSoft hyperparameters (tuned via experiments/case3/sweep.py) ----
        # n_aug=1 + wd=3e-3 picked by sweep: MSE 0.078 -> 0.021 (3.7x improvement
        # over the prior h32_b2 lr5e-3 wd1e-4 baseline). Same dominant axes as
        # cases 1-2: stronger weight decay and a single-anchor time augmentation.
        embed_dim = 8
        hidden_dim, num_blocks = 32, 1
        epochs, lr, weight_decay = 500, 1e-3, 3e-3
        n_aug, aug_weight = 1, 0.5
        # ----------------------------------------------------------------------
        print(
            f"Training CRSoft (h{hidden_dim}_b{num_blocks}_e{embed_dim}, "
            f"n_aug={n_aug}, wd={weight_decay:.0e})..."
        )
        torch.manual_seed(0)
        m = FunctionalCRSoftNet(
            num_covariates=P,
            n_grid=N_GRID,
            embed_dim=embed_dim,
            num_causes=K,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  Model parameters: {n_params:,}")

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
        )
        m._train_losses = losses  # pyre-ignore[16]
        m._hyperparams = {  # pyre-ignore[16]
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "embed_dim": embed_dim,
        }
        return m

    return [
        ModelSpec(
            name="CRSoft",
            cache_key="crsoft",
            cli_aliases=["crsoft"],
            train=_train_crsoft,
            serialize=lambda m: {
                "state_dict": m.state_dict(),
                "num_causes": m.num_causes,
                "num_covariates": P,
                "n_grid": N_GRID,
                "embed_dim": m._hyperparams["embed_dim"],
                "hidden_dim": m._hyperparams["hidden_dim"],
                "num_blocks": m._hyperparams["num_blocks"],
            },
            load_ckpt=FunctionalCRSoftNet.load_from_checkpoint,
            predict=lambda m, X, t: m.predict_cif_grid(X, t),
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

    # Stub spec list for CLI token validation (cheap; no data needed)
    stub_specs = _build_specs(
        torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    )
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
        X_test,
        Y_test,
        Delta_test,
        eval_times,
        true_cif_fn,
    ) = _prepare_data()

    specs = _build_specs(
        train_data["X_func"],
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
    crsoft_model: FunctionalCRSoftNet | None = None

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
        if spec.name == "CRSoft" and isinstance(model, FunctionalCRSoftNet):
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

    # ---- CIF comparison plot (per-cause × samples grid, same layout as case1/2) ----
    print("\nPlotting CIF comparison (CRSoft + true)...")
    n_test = X_test.shape[0]
    cif_true = torch.zeros(n_test, K, len(eval_times))
    with torch.no_grad():
        for ti in range(len(eval_times)):
            f_true, _ = true_cif_fn(X_test, eval_times[ti].expand(n_test))
            cif_true[:, :, ti] = f_true
    model_cifs = {"True": cif_true, **cifs}
    plot_cif_comparison(model_cifs, eval_times, K, output_dir=OUTPUT_DIR)

    # ---- Functional pipeline figure (X_j(s) → β_{k,j}(s) → F_k(t|X)) ----
    # Pass the post-processed (isotonic-projected) CIF tensor so the predicted
    # curves match cif_comparison.png and remain monotone in t.
    if crsoft_model is not None and "CRSoft" in cifs:
        print("\nPlotting functional pipeline (X(s) → β(s) → CIF)...")
        plot_functional_pipeline(
            cifs["CRSoft"],
            X_test,
            train_data["beta_func"],
            true_cif_fn,
            eval_times,
            K,
            output_dir=OUTPUT_DIR,
        )

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
