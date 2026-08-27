"""Figures.

Matplotlib with the Agg backend, chosen at import: this runs headless from a CLI and from
the Makefile, and the default interactive backend would try to open a window and fail.

Every figure here answers a question the results tables cannot:

- :func:`bland_altman` -- is the error a constant bias, or does it grow with pressure? The
  latter is the characteristic failure of PPG-based estimation and shows up as a slope, not
  as a worse MAE.
- :func:`prediction_scatter` -- is the model tracking individuals, or regressing everyone
  toward the cohort mean? Mean-reversion is visible instantly as a cloud flatter than the
  identity line, and it is the single most common way a good-looking MAE misleads.
- :func:`robustness_curve` -- how does error grow with corruption, and do the models
  separate as it does?
- :func:`mechanism_bars` -- *which* corruption does each model tolerate?
- :func:`waveform_panel` -- did the network produce a physically sensible pressure wave, or
  a curve that happens to have the right maximum and minimum?

The palette is Okabe-Ito, which stays distinguishable under the common forms of colour
vision deficiency and survives greyscale printing -- both relevant for a document that ends
up photocopied.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "PALETTE",
    "bland_altman",
    "prediction_scatter",
    "robustness_curve",
    "mechanism_bars",
    "training_history",
    "waveform_panel",
    "signal_gallery",
]

# Okabe-Ito, colour-vision-deficiency safe.
PALETTE = {
    "pinn": "#0072B2",
    "cnn": "#D55E00",
    "ridge": "#009E73",
    "mean": "#999999",
    "accent": "#CC79A7",
    "truth": "#000000",
}
_ORDER = ["pinn", "cnn", "ridge", "mean"]


def _colour(name: str) -> str:
    key = name.lower()
    for k in PALETTE:
        if k in key:
            return PALETTE[k]
    return PALETTE["accent"]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, title: str = "") -> Path:
    """Bland-Altman plots for SBP and DBP side by side.

    The dashed lines are the 95% limits of agreement. A fitted trend is drawn across the
    points because proportional bias -- error growing with pressure -- is the specific
    failure mode of cuffless estimation and is invisible in any single summary number.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for ax, i, name in zip(axes, (0, 1), ("SBP", "DBP"), strict=True):
        t, p = y_true[:, i], y_pred[:, i]
        mean = (t + p) / 2
        diff = p - t
        bias, sd = diff.mean(), diff.std(ddof=1)

        ax.scatter(mean, diff, s=7, alpha=0.35, color=PALETTE["pinn"], edgecolors="none")
        ax.axhline(bias, color=PALETTE["cnn"], lw=1.4, label=f"bias {bias:+.1f}")
        for k, ls in ((1.96, "--"), (-1.96, "--")):
            ax.axhline(bias + k * sd, color=PALETTE["cnn"], lw=1.0, ls=ls, alpha=0.7)
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)

        if len(mean) > 2 and np.std(mean) > 1e-9:
            slope, intercept = np.polyfit(mean, diff, 1)
            xs = np.linspace(mean.min(), mean.max(), 50)
            ax.plot(xs, slope * xs + intercept, color=PALETTE["accent"], lw=1.2,
                    label=f"trend {slope:+.2f} mmHg/mmHg")

        ax.set_xlabel(f"mean of reference and estimate ({name}, mmHg)")
        ax.set_ylabel("estimate - reference (mmHg)")
        ax.set_title(f"{name}   LoA [{bias - 1.96 * sd:.1f}, {bias + 1.96 * sd:.1f}]")
        ax.legend(loc="upper right", fontsize=7)

    if title:
        fig.suptitle(title, y=1.02)
    return _save(fig, path)


def prediction_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, title: str = "",
    train_mean: np.ndarray | None = None,
) -> Path:
    """Predicted against reference, with the identity line and the mean-predictor line.

    The horizontal grey line is what a model that learned nothing but the cohort average
    would produce. The distance of the point cloud from that line, rather than from the
    identity, is the honest visual measure of how much the model actually knows.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    for ax, i, name in zip(axes, (0, 1), ("SBP", "DBP"), strict=True):
        t, p = y_true[:, i], y_pred[:, i]
        lo = float(min(t.min(), p.min())) - 5
        hi = float(max(t.max(), p.max())) + 5

        ax.scatter(t, p, s=7, alpha=0.35, color=PALETTE["pinn"], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, ls="--", label="identity")
        if train_mean is not None:
            ax.axhline(train_mean[i], color=PALETTE["mean"], lw=1.2,
                       label=f"mean predictor ({train_mean[i]:.0f})")

        r = np.corrcoef(t, p)[0, 1] if t.std() > 1e-9 and p.std() > 1e-9 else float("nan")
        mae = float(np.abs(p - t).mean())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"reference {name} (mmHg)")
        ax.set_ylabel(f"estimated {name} (mmHg)")
        ax.set_title(f"{name}   MAE {mae:.2f}   r {r:.2f}")
        ax.legend(loc="upper left", fontsize=7)

    if title:
        fig.suptitle(title, y=1.00)
    return _save(fig, path)


def robustness_curve(sweep_rows: list[dict], path: str | Path, metric: str = "sbp_mae") -> Path:
    """Error against artifact severity, one line per model.

    The right-hand axis carries the fraction of windows the SQI would accept, which is what
    makes the curve interpretable: an error rise at a severity where most windows are
    already being rejected means something quite different from one where they are all still
    being reported to the user.
    """
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    models = sorted({r["model"] for r in sweep_rows},
                    key=lambda m: _ORDER.index(m.lower()) if m.lower() in _ORDER else 99)
    for m in models:
        rows = sorted([r for r in sweep_rows if r["model"] == m], key=lambda r: r["severity"])
        xs = [r["severity"] for r in rows]
        ys = [r[metric] for r in rows]
        ax.plot(xs, ys, marker="o", ms=4.5, lw=1.8, color=_colour(m), label=m)

    ax.set_xlabel("artifact severity")
    ax.set_ylabel(f"{metric.replace('_', ' ').upper()} (mmHg)")
    ax.set_title("Degradation under increasing motion artifact")
    ax.legend(title="model", fontsize=8)

    any_model = models[0] if models else None
    if any_model:
        rows = sorted([r for r in sweep_rows if r["model"] == any_model],
                      key=lambda r: r["severity"])
        ax2 = ax.twinx()
        ax2.plot([r["severity"] for r in rows], [r["sqi_pass_pct"] for r in rows],
                 color=PALETTE["mean"], ls=":", lw=1.4)
        ax2.set_ylabel("% windows accepted by SQI", color=PALETTE["mean"])
        ax2.tick_params(axis="y", colors=PALETTE["mean"])
        ax2.grid(False)
        ax2.set_ylim(0, 105)

    return _save(fig, path)


def mechanism_bars(mech_rows: list[dict], path: str | Path, metric: str = "sbp_mae") -> Path:
    """Grouped bars: error per artifact mechanism, per model.

    The most diagnostic figure in the set. Mechanisms are ordered by how damaging they are
    on average, so the reader sees at a glance which corruptions separate the models and
    which defeat all of them equally.
    """
    _style()
    if not mech_rows:
        raise ValueError("no per-mechanism rows to plot")

    mechs = sorted({r["mechanism"] for r in mech_rows},
                   key=lambda k: np.mean([r[metric] for r in mech_rows if r["mechanism"] == k]))
    models = sorted({r["model"] for r in mech_rows},
                    key=lambda m: _ORDER.index(m.lower()) if m.lower() in _ORDER else 99)

    x = np.arange(len(mechs))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for j, m in enumerate(models):
        vals = []
        for k in mechs:
            hit = [r[metric] for r in mech_rows if r["model"] == m and r["mechanism"] == k]
            vals.append(hit[0] if hit else np.nan)
        ax.bar(x + j * width - 0.4 + width / 2, vals, width, label=m, color=_colour(m))

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in mechs], fontsize=8)
    ax.set_ylabel(f"{metric.replace('_', ' ').upper()} (mmHg)")
    sev = mech_rows[0].get("severity", "")
    ax.set_title(f"Error by artifact mechanism (severity {sev})")
    ax.legend(title="model", fontsize=8)
    return _save(fig, path)


def training_history(history: list[dict], path: str | Path) -> Path:
    """Validation MAE and the individual physics loss terms over training.

    The physics terms are plotted on a log scale in their own panel because they differ by
    orders of magnitude and because the thing worth seeing is whether each is *falling*.
    A physics term that flatlines immediately is either weighted into irrelevance or already
    satisfied, and either way it is not contributing what the method claims.
    """
    _style()
    have_physics = any("ode" in h for h in history)
    fig, axes = plt.subplots(1, 2 if have_physics else 1, figsize=(9.5 if have_physics else 5.2, 3.6))
    axes = np.atleast_1d(axes)

    ep = [h["epoch"] for h in history]
    axes[0].plot(ep, [h["val_mae_sbp"] for h in history], color=PALETTE["pinn"], label="val SBP MAE")
    axes[0].plot(ep, [h["val_mae_dbp"] for h in history], color=PALETTE["cnn"], label="val DBP MAE")
    best = int(np.argmin([h["val_mae"] for h in history]))
    axes[0].axvline(ep[best], color=PALETTE["mean"], ls=":", lw=1.2,
                    label=f"best epoch {ep[best]}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MAE (mmHg)")
    axes[0].set_title("Validation error")
    axes[0].legend(fontsize=8)

    if have_physics:
        for key, lbl in (("data", "data"), ("ode", "ODE residual"), ("tau", "tau"),
                         ("prior", "prior"), ("map", "MAP rule"), ("per", "periodicity")):
            if key in history[0]:
                vals = np.array([h.get(key, np.nan) for h in history], dtype=float)
                vals = np.where(vals > 0, vals, np.nan)
                axes[1].plot(ep, vals, lw=1.4, label=lbl)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("loss term (log)")
        axes[1].set_title("Loss components")
        axes[1].legend(fontsize=7, ncol=2)

    return _save(fig, path)


def waveform_panel(
    ppg: np.ndarray, waveform: np.ndarray, fs: float, path: str | Path,
    sbp: float | None = None, dbp: float | None = None,
    true_sbp: float | None = None, true_dbp: float | None = None,
    params: dict | None = None,
) -> Path:
    """Input PPG beside the pressure waveform the model reconstructed.

    This is the interpretability claim made visible. The right panel is not a plot of two
    numbers; it is the pressure wave the network believes produced the PPG on the left, and
    SBP and DBP are simply its maximum and minimum. If that wave has a plausible systolic
    upstroke, a dicrotic notch and an exponential diastolic decay, the physics constraints
    are doing their job. If it is a sine wave with the right extremes, they are not.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

    t_ppg = np.arange(len(ppg)) / fs
    axes[0].plot(t_ppg, ppg, color=PALETTE["pinn"], lw=1.0)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("PPG (normalised)")
    axes[0].set_title("Input PPG window")

    u = np.linspace(0, 1, len(waveform), endpoint=False)
    axes[1].plot(u, waveform, color=PALETTE["cnn"], lw=1.8, label="reconstructed P(t)")
    if sbp is not None:
        axes[1].axhline(sbp, color=PALETTE["cnn"], ls="--", lw=0.9, alpha=0.7)
        axes[1].axhline(dbp, color=PALETTE["cnn"], ls="--", lw=0.9, alpha=0.7)
    if true_sbp is not None:
        axes[1].axhline(true_sbp, color=PALETTE["truth"], ls=":", lw=1.1,
                        label=f"reference {true_sbp:.0f}/{true_dbp:.0f}")
        axes[1].axhline(true_dbp, color=PALETTE["truth"], ls=":", lw=1.1)
    axes[1].set_xlabel("cardiac phase (fraction of cycle)")
    axes[1].set_ylabel("pressure (mmHg)")
    ttl = "Reconstructed pressure"
    if sbp is not None:
        ttl += f"   estimate {sbp:.0f}/{dbp:.0f}"
    axes[1].set_title(ttl)
    axes[1].legend(fontsize=7, loc="upper right")

    if params:
        txt = "  ".join(f"{k}={v:.2f}" for k, v in params.items())
        fig.text(0.5, -0.04, txt, ha="center", fontsize=7.5, color="#444444")

    return _save(fig, path)


def signal_gallery(
    signals: dict[str, np.ndarray], fs: float, path: str | Path, sqi: dict[str, float] | None = None
) -> Path:
    """A grid of one window under each artifact mechanism.

    Included because a reader should be able to judge for themselves whether the simulated
    corruptions look like real motion artifact. A robustness result rests entirely on that
    judgement, and asking the reader to take the corruption model on trust would be asking
    them to take the result on trust.
    """
    _style()
    n = len(signals)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 1.9 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, sig) in zip(axes, signals.items(), strict=False):
        t = np.arange(len(sig)) / fs
        ax.plot(t, sig, lw=0.9, color=PALETTE["pinn"] if name == "clean" else PALETTE["cnn"])
        title = name.replace("_", " ")
        if sqi and name in sqi:
            title += f"   SQI {sqi[name]:.2f}"
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("a.u.", fontsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncol):n]:
        ax.set_xlabel("time (s)")

    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
