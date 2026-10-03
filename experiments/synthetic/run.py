#!/usr/bin/env python3
"""
Synthetic (DeepHit benchmark) -- evaluate all 6 models with IBS / C^td.

Real-world dataset (no closed-form true CIF), so we evaluate only on
discrimination (C^td) and calibration (IBS). MSE / accuracy not applicable.

Two-tier caching for fast iteration: see ../runner.py.

CLI:
  --retrain MODEL [MODEL ...]    force retrain (also invalidates eval cache)
  --reeval  MODEL [MODEL ...]    force re-eval (re-predict + re-evaluate)
  --list                        show cache state and exit

To change CRSoft hyperparameters, edit `_train_crsoft` below and rerun
with `--retrain crsoft`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...baseline_models import CsCox, DeepHit, DSM, FineGray, NeuralFG
from ...crsoft_model import CRSoftNet
from ...data.synthetic import load_data
from ...evaluation import (
    build_evaluation_time_grid,
    evaluate_cif_metrics,
    get_output_dir,
    isotonic_project_cif,
    plot_cif_comparison,
    plot_event_time_distribution,
    plot_training_loss,
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

OUTPUT_DIR = get_output_dir("synthetic")
CASE_NAME = "synthetic"
CASE_LABEL = "Synthetic (DeepHit benchmark)"


# ======================================================================
# Data prep
# ======================================================================
def _prepare_data() -> tuple:
    print("\nLoading Synthetic dataset...")
    data = load_data(test_size=0.3, seed=42)

    K_val: int = data["K"]  # pyre-ignore[8]
    P_val: int = data["P"]  # pyre-ignore[8]
    X_train: Tensor = data["X_train"]  # pyre-ignore[8]
    Y_train: Tensor = data["Y_train"]  # pyre-ignore[8]
    Delta_train: Tensor = data["Delta_train"]  # pyre-ignore[8]
    X_test: Tensor = data["X_test"]  # pyre-ignore[8]
    Y_test: Tensor = data["Y_test"]  # pyre-ignore[8]
    Delta_test: Tensor = data["Delta_test"]  # pyre-ignore[8]
    events: list[str] = data["events"]  # pyre-ignore[8]

    n_train = X_train.shape[0]
    n_val = int(n_train * 0.2)
    X_val = X_train[:n_val]
    Y_val = Y_train[:n_val]
    Delta_val = Delta_train[:n_val]
    X_train_fit = X_train[n_val:]
    Y_train_fit = Y_train[n_val:]
    Delta_train_fit = Delta_train[n_val:]

    n_events = int((Delta_train > 0).sum().item())
    n_censored = int((Delta_train == 0).sum().item())
    print(f"\n  Train: n={n_train}, events={n_events}, censored={n_censored}")
    for k in range(1, K_val + 1):
        print(
            f"    Cause {k} ({events[k - 1]}): {int((Delta_train == k).sum().item())}"
        )
    n_test = X_test.shape[0]
    n_events_test = int((Delta_test > 0).sum().item())
    n_censored_test = int((Delta_test == 0).sum().item())
    print(f"  Test:  n={n_test}, events={n_events_test}, censored={n_censored_test}")
    print(f"  Val:   n={n_val} (first 20% of train)")
    print(f"  Features: {P_val}, Causes: {K_val}")

    # Plot event time distribution
    print("\nPlotting data distribution...")
    train_data_for_plot = {"Y": Y_train, "Delta": Delta_train, "T_true": Y_train}
    plot_event_time_distribution(train_data_for_plot, K_val, output_dir=OUTPUT_DIR)

    eval_times = build_evaluation_time_grid(Y_test, Delta_test, n_grid=100)
    print(
        f"\n  Evaluation time grid: {len(eval_times)} points, "
        f"range [{eval_times[0]:.2f}, {eval_times[-1]:.2f}]"
    )

    return (
        X_train,
        Y_train,
        Delta_train,
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
        K_val,
        P_val,
        events,
    )


# ======================================================================
# Build per-model specs
# ======================================================================
def _build_specs(
    K: int,
    P: int,
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
        )
        return m

    def _train_cscox() -> CsCox:
        print("Training cs-Cox...")
        m = CsCox(n_causes=K)
        m.fit(X_train_fit, Y_train_fit, Delta_train_fit)
        return m

    def _train_fine_gray() -> FineGray:
        print("Training Fine-Gray...")
        m = FineGray(n_causes=K)
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
        )
        return m

    def _train_crsoft() -> CRSoftNet:
        # ---- CRSoft hyperparameters (Synthetic SOTA recipe — single seed) ----
        # Single-seed Brier-augmented CRSoftNet (matches the single-seed regime
        # used by DeepHit / DSM / cs-Cox / NeuralFG in this case file). The
        # original 8-seed ensemble was rejected as unfair: ensembling CRSoft
        # while baselines run single-seed is a variance-reduction advantage
        # the architecture itself doesn't earn.
        #
        # The Synthetic dataset has a strict C^td <-> IBS tradeoff along the
        # survival-weight axis (M*lam fraction): low M favours IBS but loses
        # C^td; high M favours C^td but inflates IBS. The Brier-augmented loss
        # (brier_lambda=5.0) partially recovers IBS at high M without
        # sacrificing C^td. See brier_sweep.py Phases B/B2/F for the full
        # exploration; Phase G (seed=42 epoch convergence) settles ep=200.
        #
        # ep=200 chosen over 500/1000/2000: brier_sweep.py Phase G with seed=42
        # showed monotonic over-fitting past ep=200:
        #   ep=200  -> C^td_o=0.7471, IBS_o=0.205   (2/6 SOTA)
        #   ep=500  -> C^td_o=0.7394, IBS_o=0.203   (0/6)
        #   ep=1000 -> C^td_o=0.7364, IBS_o=0.201   (0/6)
        # Synthetic has 14k training samples (much more per-epoch updates than
        # PBC/Framingham/Case I-II), so the loss plateau is reached fast.
        hidden_dim, num_blocks, dropout = 32, 1, 0.0
        epochs, lr, weight_decay = 200, 1e-3, 1e-3
        n_aug, aug_weight = 8, 0.5
        brier_lambda = 5.0
        batch_size = 512
        # ---------------------------------------------------------------------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"  CRSoft (h{hidden_dim}_b{num_blocks}, n_aug={n_aug}, "
            f"aug_weight={aug_weight}, brier_lambda={brier_lambda}) on {device}..."
        )
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)
        m = CRSoftNet(
            input_dim=P,
            num_causes=K,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        losses = m.fit(
            X_train_full,
            Y_train_full,
            Delta_train_full,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            n_aug=n_aug,
            aug_weight=aug_weight,
            brier_lambda=brier_lambda,
            verbose=True,
            device=device,
        )
        m._train_losses = losses  # pyre-ignore[16]
        m._hyperparams = {  # pyre-ignore[16]
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "brier_lambda": brier_lambda,
            "trained_on_device": str(device),
        }
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
            # The real-data CIF cache predates probability-conserving hazard
            # integration; its native overall-survival formula is unchanged.
            allow_diagnostic_cif_mismatch=True,
        ),
        ModelSpec(
            name="Fine-Gray",
            cache_key="fine_gray",
            cli_aliases=["fg", "finegray"],
            train=_train_fine_gray,
            serialize=lambda m: m.to_checkpoint(),
            load_ckpt=FineGray.load_from_checkpoint,
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
            cli_aliases=["crsoft"],
            train=_train_crsoft,
            serialize=lambda m: {
                "state_dict": m.state_dict(),
                "num_causes": m.num_causes,
                "input_dim": P,
                "hidden_dim": m._hyperparams["hidden_dim"],
                "num_blocks": m._hyperparams["num_blocks"],
                "dropout": m._hyperparams["dropout"],
            },
            load_ckpt=CRSoftNet.load_from_checkpoint,
            predict=lambda m, X, t: m.predict_cif_grid(X, t),
            predict_survival=lambda m, X, t: m.predict_cif_survival_grid(X, t),
            post_process=lambda c: isotonic_project_cif(c, eval_times),
            is_iteration_target=False,  # settled — cache eval like other models
        ),
    ]


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    print("=" * 70)
    print(CASE_LABEL)
    print("=" * 70)

    # Build a stub spec list for CLI token validation (cheap; no data needed)
    stub_specs = _build_specs(2, 12, *[torch.empty(0)] * 9, torch.empty(0))
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
        X_train,
        Y_train,
        Delta_train,
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
        K,
        P,
        events,
    ) = _prepare_data()

    specs = _build_specs(
        K,
        P,
        X_train_fit,
        Y_train_fit,
        Delta_train_fit,
        X_val,
        Y_val,
        Delta_val,
        X_train,
        Y_train,
        Delta_train,
        eval_times,
    )

    # eval_fn: real-world data, only IBS + C^td (no MSE/accuracy)
    def eval_fn(cif: Tensor) -> dict[str, float]:
        return evaluate_cif_metrics(
            cif, Y_test, Delta_test, Y_train, Delta_train, eval_times, K
        )

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
            include_probability_diagnostics=True,
        )
        all_results.append((spec.name, metrics))
        cifs[spec.name] = cif
        if spec.name == "CRSoft" and isinstance(model, CRSoftNet):
            crsoft_model = model

    # ---- CRSoft-specific extras ----
    if crsoft_model is not None:
        losses = getattr(crsoft_model, "_train_losses", None)
        if losses:
            print("\nPlotting CRSoft training loss (member 0)...")
            plot_training_loss(losses, output_dir=OUTPUT_DIR)

    # ---- Print + save ----
    print_results(CASE_LABEL, K, all_results)
    save_results_txt(CASE_NAME, all_results)

    # ---- CIF comparison plot (no True curve — real data) ----
    print("\nPlotting CIF comparison across models...")
    plot_cif_comparison(
        cifs,
        eval_times,
        K,
        event_names=events,
        output_dir=OUTPUT_DIR,
    )

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
