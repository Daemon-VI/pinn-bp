"""Command line interface.

    pinnbp train      --config configs/pinn.yaml
    pinnbp experiment --config configs/experiment.yaml    # the whole comparison
    pinnbp sweep      --run reports/pinn-default
    pinnbp figures    --run reports/pinn-default
    pinnbp gallery                                        # artifact examples
    pinnbp info

``experiment`` is the one that matters: it trains all three models on identical data, runs
the robustness sweep, checks parameter recovery, draws every figure and writes
``docs/RESULTS.md`` from the numbers it just measured. Having one command produce the whole
report is what stops the report and the code drifting apart -- a results file that is
regenerated rather than edited cannot quietly keep a number the code no longer produces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import Config, load_config

__all__ = ["main"]


def _load(args) -> Config:
    cfg = load_config(args.config) if args.config else Config()
    if getattr(args, "name", None):
        cfg.name = args.name
    if getattr(args, "model", None):
        cfg.model_kind = args.model
    if getattr(args, "epochs", None):
        cfg.train.epochs = args.epochs
    if getattr(args, "subjects", None):
        cfg.data.n_subjects = args.subjects
    if getattr(args, "seed", None) is not None:
        cfg.data.seed = args.seed
        cfg.train.seed = args.seed
    if getattr(args, "split", None):
        cfg.data.split = args.split
    return cfg


def cmd_train(args) -> int:
    from .train import train_model

    cfg = _load(args)
    print(f"=== train {cfg.model_kind} :: {cfg.name} ===")
    res = train_model(cfg)
    print(f"[done] {res.seconds:.1f}s -> {res.out_dir}")
    return 0


def cmd_sweep(args) -> int:
    from .evaluate import robustness_sweep, save_sweep

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    model = _restore(run, cfg)
    sweep = robustness_sweep({cfg.model_kind: model}, cfg)
    save_sweep(sweep, run / "sweep.json")
    print(f"[done] -> {run / 'sweep.json'}")
    return 0


def _restore(run: Path, cfg: Config):
    """Reload a trained model from a run directory."""
    import torch

    from .train import build_model

    model = build_model(cfg)
    ckpt = run / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def cmd_figures(args) -> int:
    import torch

    from . import plots
    from .data.build import build_dataset, training_mean

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    fig_dir = run / "figures"

    hist_path = run / "history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        if history:
            plots.training_history(history, fig_dir / "training.png")

    pred_path = run / "test_predictions.npz"
    if pred_path.exists():
        with np.load(pred_path) as z:
            y_true, y_pred = z["y_true"], z["y_pred"]
        prepared, splits = build_dataset(cfg, verbose=False)
        tmean = training_mean(prepared, splits)
        plots.bland_altman(y_true, y_pred, fig_dir / "bland_altman.png", title=cfg.name)
        plots.prediction_scatter(y_true, y_pred, fig_dir / "scatter.png",
                                 title=cfg.name, train_mean=tmean)

    sweep_path = run / "sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        plots.robustness_curve(sweep["sweep"], fig_dir / "robustness.png")
        if sweep.get("per_mechanism"):
            plots.mechanism_bars(sweep["per_mechanism"], fig_dir / "mechanisms.png")

    if cfg.model_kind == "pinn" and (run / "best.pt").exists():
        model = _restore(run, cfg)
        prepared, splits = build_dataset(cfg, verbose=False)
        idx = splits.test[: min(4, len(splits.test))]
        with torch.no_grad():
            x = torch.from_numpy(prepared["channels"][idx])
            out = model(x)
        for k, j in enumerate(idx):
            plots.waveform_panel(
                prepared["channels"][j, 0], out.waveform[k].numpy(), prepared["fs"],
                fig_dir / f"waveform_{k}.png",
                sbp=float(out.sbp[k]), dbp=float(out.dbp[k]),
                true_sbp=float(prepared["sbp"][j]), true_dbp=float(prepared["dbp"][j]),
                params={
                    "R": float(out.params.R[k]), "C": float(out.params.C[k]),
                    "SV": float(out.params.SV[k]), "HR": float(out.params.HR[k]),
                },
            )

    print(f"[done] figures -> {fig_dir}")
    return 0


def cmd_gallery(args) -> int:
    """Draw one window under each artifact mechanism, with its SQI."""
    from . import plots
    from .data.artifacts import ARTIFACT_KINDS, apply_artifacts
    from .data.synthetic import CohortConfig, generate_cohort, sample_subject, simulate_subject
    from .dsp.preprocess import bandpass, normalize_window
    from .dsp.sqi import signal_quality

    fs = 125.0
    rng = np.random.default_rng(7)
    p = sample_subject(rng, 0, hypertensive=False)
    rec = simulate_subject(p, 12.0, fs, rng)
    raw = rec["ppg"][: int(8 * fs)]

    sigs: dict[str, np.ndarray] = {}
    sqis: dict[str, float] = {}
    clean = normalize_window(bandpass(raw, fs), "robust")
    sigs["clean"] = clean
    sqis["clean"] = signal_quality(clean, fs).overall

    for kind in ARTIFACT_KINDS:
        r = np.random.default_rng(11)
        corrupted = apply_artifacts(raw, fs, float(args.severity), r, kinds=(kind,), n_kinds=1)
        s = normalize_window(bandpass(corrupted, fs), "robust")
        sigs[kind] = s
        sqis[kind] = signal_quality(s, fs).overall

    out = Path(args.out)
    plots.signal_gallery(sigs, fs, out, sqi=sqis)
    print(f"[done] gallery -> {out}")
    for k, v in sqis.items():
        print(f"  {k:17s} SQI {v:.3f}")

    # A quick reminder that the cohort behind the gallery is the same one used everywhere.
    c = generate_cohort(CohortConfig(n_subjects=4, duration_s=15.0, seed=7))
    print(f"[info] cohort check: {len(c['sbp'])} windows, "
          f"SBP {c['sbp'].mean():.0f}/{c['dbp'].mean():.0f} mmHg")
    return 0


def cmd_info(args) -> int:
    """Print environment, dataset availability, and model sizes."""
    import torch

    from .data.real import DATASET_SOURCES
    from .models.baselines import DataDrivenCNN
    from .models.pinn import PINNBP

    print(f"python  {sys.version.split()[0]}")
    print(f"torch   {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"threads={torch.get_num_threads()}")
    print(f"PINN    {PINNBP().n_parameters():,} parameters")
    print(f"CNN     {DataDrivenCNN().n_parameters():,} parameters")
    print("\ndatasets:")
    print("  synthetic  always available (simulator; pipeline validation only)")
    for key, src in DATASET_SOURCES.items():
        print(f"  {key:10s} {src['name']}")
        print(f"             {src['url']}  ({src['size']})")
    return 0


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render rows as a Markdown table. ``columns`` is [(key, header)]."""
    head = "| " + " | ".join(h for _, h in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for r in rows:
        cells = []
        for k, _ in columns:
            v = r.get(k, "")
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def cmd_experiment(args) -> int:
    """Train every model, sweep, and write docs/RESULTS.md from the measured numbers."""
    from . import plots
    from .data.build import build_dataset, training_mean
    from .evaluate import parameter_recovery, robustness_sweep, save_sweep, sqi_stratified_report
    from .train import train_model

    base = _load(args)
    out_root = Path(base.out_dir)
    results: dict[str, dict] = {}
    models: dict[str, object] = {}
    trained: dict[str, object] = {}

    for kind in ("pinn", "cnn", "ridge"):
        cfg = load_config(args.config) if args.config else Config()
        cfg.model_kind = kind
        cfg.name = f"{args.tag}-{kind}" if args.tag else kind
        if args.epochs:
            cfg.train.epochs = args.epochs
        if args.subjects:
            cfg.data.n_subjects = args.subjects
        if args.seed is not None:
            cfg.data.seed = args.seed
            cfg.train.seed = args.seed

        print(f"\n{'=' * 68}\n=== {kind.upper()} ===\n{'=' * 68}")
        res = train_model(cfg)
        results[kind] = res.test_report
        models[kind] = res.model
        trained[kind] = res

    sweep_cfg = load_config(args.config) if args.config else Config()
    if args.subjects:
        sweep_cfg.data.n_subjects = args.subjects
    if args.seed is not None:
        sweep_cfg.data.seed = args.seed

    print(f"\n{'=' * 68}\n=== ROBUSTNESS SWEEP ===\n{'=' * 68}")
    sweep = robustness_sweep(models, sweep_cfg)
    save_sweep(sweep, out_root / "sweep.json")

    print(f"\n{'=' * 68}\n=== PARAMETER RECOVERY ===\n{'=' * 68}")
    recovery = parameter_recovery(models["pinn"], sweep_cfg, severity=0.0)
    for k, v in recovery.items():
        if isinstance(v, dict) and "pearson_r" in v:
            print(f"  {k:5s} r {v['pearson_r']:+.3f}  rho {v['spearman_r']:+.3f}")
    recovery_noisy = parameter_recovery(models["pinn"], sweep_cfg, severity=0.6)

    strat = sqi_stratified_report(models["pinn"], sweep_cfg, severity=0.6)
    print(f"\n[SQI gating] coverage {strat['coverage'] * 100:.0f}%  "
          f"accepted MAE {strat.get('accepted', {}).get('sbp_mae', float('nan')):.2f}  "
          f"rejected MAE {strat.get('rejected', {}).get('sbp_mae', float('nan')):.2f}")

    fig_dir = out_root / "figures"
    plots.robustness_curve(sweep["sweep"], fig_dir / "robustness_sbp.png", metric="sbp_mae")
    plots.robustness_curve(sweep["sweep"], fig_dir / "robustness_dbp.png", metric="dbp_mae")
    if sweep["per_mechanism"]:
        plots.mechanism_bars(sweep["per_mechanism"], fig_dir / "mechanisms.png")

    prepared, splits = build_dataset(sweep_cfg, verbose=False)
    tmean = training_mean(prepared, splits)
    for kind, res in trained.items():
        pred_path = res.out_dir / "test_predictions.npz"
        if pred_path.exists():
            with np.load(pred_path) as z:
                yt, yp = z["y_true"], z["y_pred"]
            plots.bland_altman(yt, yp, fig_dir / f"bland_altman_{kind}.png", title=kind)
            plots.prediction_scatter(yt, yp, fig_dir / f"scatter_{kind}.png",
                                     title=kind, train_mean=tmean)
        if res.history:
            plots.training_history(res.history, fig_dir / f"training_{kind}.png")

    payload = {
        "results": results,
        "sweep": sweep,
        "parameter_recovery_clean": recovery,
        "parameter_recovery_severity_0.6": recovery_noisy,
        "sqi_stratified": strat,
        "config": sweep_cfg.as_dict(),
    }
    (out_root / "experiment.json").write_text(json.dumps(payload, indent=2, default=str),
                                              encoding="utf-8")

    _write_results_md(payload, Path("docs/RESULTS.md"))
    print(f"\n[done] experiment.json + docs/RESULTS.md + figures -> {out_root}")
    return 0


def _write_results_md(payload: dict, path: Path) -> None:
    """Generate docs/RESULTS.md from measured numbers.

    Generated, never hand-edited -- the same rule this workspace applies to resumes and
    generated decks. Every number below traces to ``reports/experiment.json``, so a claim
    here cannot survive the code that produced it changing.
    """
    res = payload["results"]
    sweep = payload["sweep"]
    cfg = payload["config"]

    rows = []
    for kind in ("pinn", "cnn", "ridge"):
        if kind not in res:
            continue
        r = res[kind]
        rows.append({
            "model": kind,
            "sbp_mae": r["sbp"]["mae"], "sbp_me": r["sbp"]["me"], "sbp_sd": r["sbp"]["sd"],
            "sbp_bhs": r["sbp"]["bhs"]["grade"], "sbp_aami": "pass" if r["sbp"]["aami"]["passes"] else "fail",
            "dbp_mae": r["dbp"]["mae"], "dbp_me": r["dbp"]["me"], "dbp_sd": r["dbp"]["sd"],
            "dbp_bhs": r["dbp"]["bhs"]["grade"], "dbp_aami": "pass" if r["dbp"]["aami"]["passes"] else "fail",
        })
    if rows and "mean_baseline" in res.get("pinn", {}):
        mb = res["pinn"]["mean_baseline"]
        rows.append({
            "model": "mean predictor", "sbp_mae": mb["sbp"]["mae"], "sbp_me": mb["sbp"]["me"],
            "sbp_sd": mb["sbp"]["sd"], "sbp_bhs": "-", "sbp_aami": "-",
            "dbp_mae": mb["dbp"]["mae"], "dbp_me": mb["dbp"]["me"], "dbp_sd": mb["dbp"]["sd"],
            "dbp_bhs": "-", "dbp_aami": "-",
        })

    main_tbl = _markdown_table(rows, [
        ("model", "model"), ("sbp_mae", "SBP MAE"), ("sbp_me", "SBP ME"), ("sbp_sd", "SBP SD"),
        ("sbp_bhs", "SBP BHS"), ("dbp_mae", "DBP MAE"), ("dbp_me", "DBP ME"),
        ("dbp_sd", "DBP SD"), ("dbp_bhs", "DBP BHS"),
    ])

    sweep_tbl = _markdown_table(
        sorted(sweep["sweep"], key=lambda r: (r["severity"], r["model"])),
        [("severity", "severity"), ("model", "model"), ("sbp_mae", "SBP MAE"),
         ("dbp_mae", "DBP MAE"), ("sqi_pass_pct", "% SQI ok")],
    )

    mech_tbl = _markdown_table(
        sorted(sweep.get("per_mechanism", []), key=lambda r: (r["mechanism"], r["model"])),
        [("mechanism", "mechanism"), ("model", "model"), ("sbp_mae", "SBP MAE"),
         ("dbp_mae", "DBP MAE"), ("sqi_mean", "mean SQI")],
    )

    rec = payload["parameter_recovery_clean"]
    rec_rows = [
        {"param": k, "pearson_r": v["pearson_r"], "spearman_r": v["spearman_r"],
         "pred_sd": v.get("pred_sd", float("nan")), "true_sd": v.get("true_sd", float("nan"))}
        for k, v in rec.items() if isinstance(v, dict) and "pearson_r" in v
    ]
    rec_tbl = _markdown_table(rec_rows, [
        ("param", "parameter"), ("pearson_r", "Pearson r"), ("spearman_r", "Spearman rho"),
        ("pred_sd", "predicted SD"), ("true_sd", "true SD"),
    ]) if rec_rows else "_not available_"

    strat = payload["sqi_stratified"]
    d = cfg["data"]

    text = f"""# Results

**Generated by `pinnbp experiment`. Do not hand-edit** -- rerun the command instead. Every
number here is read from `reports/experiment.json`, which is written by the same run.

## What was measured, and on what

| | |
|---|---|
| dataset | `{d['dataset']}` |
| subjects | {d['n_subjects']} |
| window | {d['window_s']} s at {d['fs']} Hz, stride {d['stride_s']} s |
| split | **{d['split']}-disjoint**, ratios {tuple(d['ratios'])}, seed {d['seed']} |
| training artifact severity | {d['train_artifact_severity']} (applied to {d['train_artifact_prob'] * 100:.0f}% of windows) |

> **These are simulated data.** The cohort comes from the physiological simulator in
> `pinnbp.data.synthetic`, not from people. What follows validates that the pipeline,
> physics, losses, splits and metrics behave correctly and that the model can recover what
> is in principle recoverable. It is **not** clinical accuracy, and no number here should be
> quoted as performance on human subjects. See `docs/DATA.md` for running the same pipeline
> on PPG-BP or the UCI set.

## Headline comparison (clean test split)

{main_tbl}

ME and SD are the AAMI quantities; MAE is included because it is what most papers report.
The mean-predictor row is the floor: a model that does not beat it comfortably has learned
the cohort average and nothing about the individual.

## Robustness to motion artifact

Models were trained once, at severity {d['train_artifact_severity']}, and evaluated across the
whole ladder without retraining. Physiology and artifacts are drawn from independent random
streams, so labels and subjects are identical at every severity and only the signals change.

{sweep_tbl}

### By artifact mechanism (severity {sweep.get('per_mechanism_at')})

{mech_tbl}

The mechanisms are not interchangeable. `amplitude_drift` and `clipping` destroy amplitude
while preserving pulse timing, which is the regime where physics constraints should help.
`contact_loss` removes the signal entirely and should defeat every model -- a model that
looked robust to it would indicate a bug, not a result.

## Physical parameter recovery (PINN, clean signals)

Correlation between the parameters the network predicted and the simulator's true values.
This is only checkable on simulated data, and it is the strongest available test of whether
the physics is doing real work or acting as an elaborate regulariser.

{rec_tbl}

## Signal quality gating

At severity 0.6, gating on SQI >= {strat['threshold']} keeps {strat['coverage'] * 100:.0f}% of windows
({strat['n_accepted']} accepted, {strat['n_rejected']} rejected).

| set | SBP MAE | DBP MAE | n |
|---|---|---|---|
| accepted | {strat.get('accepted', {}).get('sbp_mae', float('nan')):.2f} | {strat.get('accepted', {}).get('dbp_mae', float('nan')):.2f} | {strat.get('accepted', {}).get('n', 0)} |
| rejected | {strat.get('rejected', {}).get('sbp_mae', float('nan')):.2f} | {strat.get('rejected', {}).get('dbp_mae', float('nan')):.2f} | {strat.get('rejected', {}).get('n', 0)} |

If accepted-window error is materially lower, refusing to report a value on rejected windows
is a real safety gain and the coverage figure is its price.

## Figures

`reports/figures/` -- robustness curves, per-mechanism bars, Bland-Altman and scatter plots
per model, training curves, and reconstructed pressure waveforms.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pinnbp",
        description="Physics-informed cuffless blood pressure estimation from PPG.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", type=str, default=None, help="YAML config path")
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--subjects", type=int, default=None)
        p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("train", help="train one model")
    common(p)
    p.add_argument("--model", choices=["pinn", "cnn", "ridge"], default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--split", choices=["subject", "random"], default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("experiment", help="train all models, sweep, write docs/RESULTS.md")
    common(p)
    p.add_argument("--tag", type=str, default=None, help="prefix for run directory names")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("sweep", help="robustness sweep for one trained run")
    p.add_argument("--run", type=str, required=True)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("figures", help="draw figures for one trained run")
    p.add_argument("--run", type=str, required=True)
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("gallery", help="plot one window under each artifact mechanism")
    p.add_argument("--severity", type=float, default=0.8)
    p.add_argument("--out", type=str, default="reports/figures/artifact_gallery.png")
    p.set_defaults(func=cmd_gallery)

    p = sub.add_parser("info", help="environment and dataset availability")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
