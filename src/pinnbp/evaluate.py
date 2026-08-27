"""Evaluation: clinical metrics, the robustness sweep, and parameter recovery.

The robustness sweep is the project's headline experiment, so it is worth being explicit
about what makes it a fair test rather than a demonstration.

*The cohort is held fixed.* Physiology and artifacts are drawn from independent random
streams (see :func:`pinnbp.data.synthetic.generate_cohort`), so raising the severity changes
the signals and nothing else. Labels, subject parameters and the train/test partition are
bit-identical across every point on the curve. Without that, a degradation curve measures
corruption *and* cohort drift together and cannot separate them.

*The test subjects never move.* The split is computed from the same seed at every severity,
so the same people are being tested throughout.

*Models are trained once, at one severity, then evaluated across all of them.* Retraining at
each level would measure something different and much easier -- adaptation to a known noise
level, rather than robustness to an unexpected one. A wearable meets whatever the wrist
gives it.

*Per-mechanism results are reported, not just the aggregate.* The mechanisms damage
different information, and the aggregate curve hides which kind of damage a model actually
tolerates. That breakdown is where a physics-informed model should distinguish itself: it
should hold up under corruptions that preserve pulse timing while destroying amplitude, and
it should fail like everything else under contact loss, which removes the signal entirely.
A model that appeared robust to contact loss would be evidence of a bug, not a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.artifacts import ARTIFACT_KINDS
from .data.datasets import BPWindows, build_windows, collate, subject_split
from .metrics import regression_report
from .train import predict

__all__ = [
    "corrupted_test_set",
    "robustness_sweep",
    "parameter_recovery",
    "sqi_stratified_report",
    "DEFAULT_SEVERITIES",
]

DEFAULT_SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def corrupted_test_set(
    cfg: Config, severity: float, kinds: tuple[str, ...] | None = None, verbose: bool = False
) -> tuple[dict, np.ndarray]:
    """Rebuild the test split with a controlled level of corruption.

    Regenerates the cohort at the given severity with ``artifact_prob=1.0`` so that severity
    is the only variable -- with a probability below one, a low-severity point would be a
    mixture of clean and corrupted windows and the curve would measure the mixture rather
    than the severity.

    Only the test subjects are prepared. ``build_windows`` is the slow step (per-beat
    exponential fits and the classical feature matrix), and preparing the ~60% of windows
    that belong to training subjects and then discarding them would triple the sweep's cost
    for nothing.

    Args:
        cfg: the config the model was trained with. Its data seed and split settings are
            reused unchanged, which is what keeps the test subjects fixed.
        severity: 0 to 1.
        kinds: restrict to specific artifact mechanisms; ``None`` uses the default pool.
        verbose: print the resulting quality statistics.

    Returns:
        ``(prepared, indices)`` where ``indices`` addresses every row of ``prepared``.
    """
    if cfg.data.dataset != "synthetic":
        raise ValueError(
            "the robustness sweep regenerates signals from the simulator, so it needs "
            f"dataset=synthetic (got {cfg.data.dataset!r}). For a real dataset, corrupt the "
            "loaded windows directly with pinnbp.data.artifacts.apply_artifacts."
        )

    from .data.synthetic import CohortConfig, generate_cohort

    d = cfg.data
    cohort = generate_cohort(
        CohortConfig(
            n_subjects=d.n_subjects,
            duration_s=d.duration_s,
            fs=d.fs,
            window_s=d.window_s,
            stride_s=d.stride_s,
            hypertensive_fraction=d.hypertensive_fraction,
            seed=d.seed,
            artifact_severity=severity,
            artifact_prob=1.0,
            extra={"artifact_kinds": list(kinds)} if kinds else {},
        )
    )

    # Same seed, same ratios, same mode -> the same subjects land in test as during training.
    splits = subject_split(
        cohort["subject"], ratios=tuple(d.ratios), seed=d.seed, mode=d.split
    )
    sel = splits.test

    prepared = build_windows(
        ppg=cohort["ppg"][sel],
        sbp=cohort["sbp"][sel],
        dbp=cohort["dbp"][sel],
        subject=cohort["subject"][sel],
        fs=d.fs,
        compute_features=True,
        sqi=cohort["sqi"][sel],
    )
    prepared["true_params"] = cohort["params"][sel]

    if verbose:
        q = prepared["sqi"]
        print(f"  severity {severity:.2f}: {len(sel)} windows, SQI {q.mean():.3f}, "
              f"{(q >= 0.5).mean() * 100:.0f}% acceptable")

    return prepared, np.arange(len(sel))


def _predict_any(model, prepared: dict, idx: np.ndarray, batch_size: int = 64):
    """Run any of the three model kinds over a prepared set.

    Ridge takes the classical feature matrix and the neural models take the channel tensor,
    so the branch is unavoidable; it is isolated here so that nothing else in the evaluation
    path has to know which kind of model it holds.
    """
    ds = BPWindows(prepared, idx)
    if hasattr(model, "predict") and not isinstance(model, torch.nn.Module):
        return ds.targets, np.asarray(model.predict(ds.features)), {"sqi": prepared["sqi"][idx]}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return predict(model, loader)


def robustness_sweep(
    models: dict[str, object],
    cfg: Config,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
    per_mechanism_at: float | None = 0.8,
    verbose: bool = True,
) -> dict:
    """Evaluate every model across the severity ladder and per artifact mechanism.

    Args:
        models: ``{name: model}``. All are evaluated on identical corrupted data, generated
            once per severity level and shared, so no model gets a luckier draw.
        cfg: the training config, reused for cohort and split settings.
        severities: severity levels to sweep.
        per_mechanism_at: severity at which to additionally break results down by mechanism.
            ``None`` skips that pass.
        verbose: progress output.

    Returns:
        ``{"sweep": [...], "per_mechanism": [...], "severities": [...]}`` where each row is
        a flat dict ready to become a DataFrame or a Markdown table.
    """
    rows: list[dict] = []
    mech_rows: list[dict] = []

    for sev in severities:
        if verbose:
            print(f"[sweep] severity {sev:.2f}")
        prepared, idx = corrupted_test_set(cfg, sev, verbose=verbose)

        for name, model in models.items():
            y_true, y_pred, extras = _predict_any(model, prepared, idx)
            rep = regression_report(y_true, y_pred)
            rows.append({
                "model": name,
                "severity": sev,
                "sqi_mean": float(prepared["sqi"].mean()),
                "sqi_pass_pct": float((prepared["sqi"] >= 0.5).mean() * 100),
                "sbp_mae": rep["sbp"]["mae"],
                "dbp_mae": rep["dbp"]["mae"],
                "sbp_me": rep["sbp"]["me"],
                "dbp_me": rep["dbp"]["me"],
                "sbp_sd": rep["sbp"]["sd"],
                "dbp_sd": rep["dbp"]["sd"],
                "sbp_r2": rep["sbp"]["r2"],
                "sbp_bhs": rep["sbp"]["bhs"]["grade"],
                "dbp_bhs": rep["dbp"]["bhs"]["grade"],
                "n": rep["sbp"]["n"],
            })
            if verbose:
                print(f"    {name:12s} SBP MAE {rep['sbp']['mae']:6.2f}  "
                      f"DBP MAE {rep['dbp']['mae']:6.2f}")

    if per_mechanism_at is not None:
        for kind in ARTIFACT_KINDS:
            if verbose:
                print(f"[mechanism] {kind} @ severity {per_mechanism_at}")
            prepared, idx = corrupted_test_set(cfg, per_mechanism_at, kinds=(kind,))
            for name, model in models.items():
                y_true, y_pred, _ = _predict_any(model, prepared, idx)
                rep = regression_report(y_true, y_pred)
                mech_rows.append({
                    "model": name,
                    "mechanism": kind,
                    "severity": per_mechanism_at,
                    "sqi_mean": float(prepared["sqi"].mean()),
                    "sbp_mae": rep["sbp"]["mae"],
                    "dbp_mae": rep["dbp"]["mae"],
                })

    return {
        "sweep": rows,
        "per_mechanism": mech_rows,
        "severities": list(severities),
        "per_mechanism_at": per_mechanism_at,
    }


def parameter_recovery(model, cfg: Config, severity: float = 0.0) -> dict:
    """Ask whether the PINN recovered the true physical parameters.

    Only answerable on simulated data, and that is precisely why it is worth doing: on real
    data there is no ground-truth compliance to check against, so a model could produce
    entirely fictitious parameters that happen to compose into the right pressure and nobody
    would ever know. Here the true values exist.

    This is the strongest available test of whether the physics is doing real work or is
    merely an elaborate regulariser. A model that predicts pressure well but recovers R and
    C no better than chance has not learned the mechanism -- and its interpretability claim,
    which is a large part of what the abstract promises, would be unsupported.

    Correlation rather than error, because the parameters have different units and because
    the identifiable quantity from a single PPG site is the *ranking* of subjects by
    compliance rather than its absolute value in mL/mmHg.

    Returns:
        ``{param: {"pearson_r": r, "spearman_r": rho, "pred_mean": .., "true_mean": ..}}``,
        or a ``note`` when the model has no parameter head.
    """
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "param_head"):
        return {"note": "model has no physical parameter head; recovery is not defined"}

    prepared, idx = corrupted_test_set(cfg, severity)
    if "true_params" not in prepared:
        return {"note": "no ground-truth parameters available for this dataset"}

    _, _, extras = _predict_any(model, prepared, idx)
    if "params" not in extras:
        return {"note": "model did not return parameters"}

    pred = extras["params"]           # (n, 5) R, C, SV, HR, Tsys
    true = prepared["true_params"]    # (n, 5) same order

    from scipy.stats import spearmanr

    names = ["R", "C", "SV", "HR", "Tsys"]
    out: dict = {}
    for i, nm in enumerate(names):
        p, t = pred[:, i], true[:, i]
        if np.std(t) < 1e-9 or np.std(p) < 1e-9:
            out[nm] = {"pearson_r": float("nan"), "spearman_r": float("nan")}
            continue
        out[nm] = {
            "pearson_r": float(np.corrcoef(p, t)[0, 1]),
            "spearman_r": float(spearmanr(p, t).statistic),
            "pred_mean": float(p.mean()),
            "true_mean": float(t.mean()),
            "pred_sd": float(p.std()),
            "true_sd": float(t.std()),
        }
    return out


def sqi_stratified_report(
    model, cfg: Config, severity: float = 0.6, threshold: float = 0.5
) -> dict:
    """Split test performance by whether the SQI accepted the window.

    This is what turns the quality index from a number into a usable safety mechanism. If
    error on accepted windows is materially lower than on rejected ones, then refusing to
    report a value on rejected windows is a real improvement in safety rather than an
    excuse for discarding inconvenient data -- and the ``coverage`` figure states the price
    of that refusal, which is the fraction of readings the user does not get.
    """
    prepared, idx = corrupted_test_set(cfg, severity)
    y_true, y_pred, _ = _predict_any(model, prepared, idx)
    sqi = prepared["sqi"]

    keep = sqi >= threshold
    out: dict = {
        "severity": severity,
        "threshold": threshold,
        "coverage": float(keep.mean()),
        "n_accepted": int(keep.sum()),
        "n_rejected": int((~keep).sum()),
    }
    for label, mask in (("accepted", keep), ("rejected", ~keep)):
        if mask.sum() >= 2:
            rep = regression_report(y_true[mask], y_pred[mask])
            out[label] = {"sbp_mae": rep["sbp"]["mae"], "dbp_mae": rep["dbp"]["mae"],
                          "n": int(mask.sum())}
        else:
            out[label] = {"note": f"only {int(mask.sum())} windows", "n": int(mask.sum())}
    return out


def save_sweep(sweep: dict, path: str | Path) -> None:
    """Persist a sweep result as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
