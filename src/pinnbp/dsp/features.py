"""Pulse-wave-analysis features.

Two consumers, with quite different requirements:

1. The **ridge baseline** needs a fixed-length descriptor of a window. If a 30-year-old
   handcrafted feature set plus linear regression already matches the deep models, that is
   the finding, and the project should report it rather than bury it. This baseline exists
   to make that check honest.

2. The **tau physics term** needs one specific measurement: the exponential decay constant
   of the diastolic limb. That is not a statistical feature, it is an estimate of a
   physical parameter, and it comes with a goodness-of-fit that the loss uses to decide
   whether to trust it.

An honest caveat on tau, stated here because a physics loss depends on it. The decay
constant measured from a band-passed PPG is **not** the arterial R*C. It is a monotone
proxy for it, on a different scale, for two compounding reasons:

1. The PPG measures blood *volume*, not pressure, through a nonlinear compliance curve, so
   the recovered decay is biased -- most at high pressures, where the curve saturates.
2. More severely, the 0.5 Hz high-pass edge sits well above the frequency at which a 2 s
   decay lives (1/(2*pi*2.3) ~ 0.07 Hz). The filter removes most of the exponential and
   what survives decays far faster than the underlying pressure does.

Measured on this project's simulator across 30 subjects, the PPG-derived value correlates
with the true R*C at r ~ 0.58 while sitting roughly four times too low in absolute terms
(median 0.56 s against a true 2.26 s). Fitting the same estimator to the *true* pressure
waveform recovers 2.41 s against 2.26 s, which confirms the estimator itself is sound and
the loss is in the transduction and the filter.

The consequence is a design constraint, not a footnote: ``pinnbp.physics.losses.tau_consistency``
must use this as a **relative, scale-free** constraint (standardised within a batch), never
as an absolute target. Matching predicted R*C to it directly would drive compliance roughly
four times too low.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .preprocess import derivatives, find_systolic_peaks, segment_beats

__all__ = [
    "FEATURE_NAMES",
    "diastolic_tau",
    "beat_features",
    "window_features",
    "heart_rate",
]

FEATURE_NAMES: list[str] = [
    "hr_bpm",
    "hrv_sd_ms",
    "sys_amplitude",
    "sys_time_s",
    "dia_time_s",
    "sys_dia_ratio",
    "pw25",
    "pw50",
    "pw75",
    "augmentation_index",
    "notch_rel_time",
    "notch_rel_height",
    "tau_s",
    "tau_fit_r2",
    "apg_b_over_a",
    "vpg_peak",
    "skewness",
    "kurtosis",
    "area_under_pulse",
    "pulse_width_ratio",
]


def heart_rate(x: np.ndarray, fs: float) -> tuple[float, float]:
    """Heart rate and beat-to-beat variability from peak intervals.

    Returns:
        ``(hr_bpm, sd_ms)``. HR is NaN when fewer than two peaks were found -- NaN rather
        than a plausible default, so that a window with no detectable pulse is visibly
        unusable downstream instead of quietly contributing a fabricated 70 bpm.
    """
    peaks = find_systolic_peaks(x, fs)
    if len(peaks) < 2:
        return float("nan"), float("nan")
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr)
    sd = float(np.std(rr) * 1000.0) if len(rr) > 1 else 0.0
    return float(hr), sd


def diastolic_tau(beat: np.ndarray, fs: float, start_frac: float = 0.3) -> tuple[float, float]:
    """Estimate the diastolic decay constant and return ``(tau, r2)``.

    Method: over the late diastolic limb, regress the signal's time derivative against the
    signal itself. For any first-order decay toward an unknown baseline,

        y(t) = A exp(-t/tau) + c     =>     dy/dt = -(1/tau) * (y - c)

    so a straight-line fit of dy/dt against y has slope -1/tau, and the baseline c falls out
    of the intercept without ever needing to be estimated.

    That last point is the reason for this formulation. The obvious approach -- subtract a
    floor and fit a line to the logarithm -- requires knowing the asymptote, and the natural
    guess for it, the segment minimum, is not the asymptote at all but simply the last
    sample of a still-decaying curve. Subtracting it flattens the tail and biases tau
    severely: on a synthetic decay with tau = 1.4 s, the log-fit-after-min-subtraction
    approach returns 0.16 s. That is not a small error, and because tau feeds a physics
    loss it would not fail loudly -- it would quietly teach the network a compliance almost
    an order of magnitude wrong.

    Where the fit window starts is a real trade-off. Early diastole still contains the
    reflected wave and the dicrotic notch, which the 2-element Windkessel does not describe,
    so starting later reduces that contamination. But tau is 1.5-2.5 s while a whole
    diastole is only ~0.4 s, so the signal decays by little over the window and starting too
    late leaves a nearly straight line from which no rate can be recovered. Measured over 30
    simulated subjects, starting at 60% yields a usable fit on 4 of them; starting at 30%
    yields 21, at no cost in correlation with the true R*C. Hence the 0.3 default.

    Args:
        beat: one foot-to-foot beat.
        fs: sampling rate in Hz.
        start_frac: where to begin the fit, as a fraction of the distance from the systolic
            peak to the end of the beat.

    Returns:
        ``(tau_seconds, r_squared)`` where r2 is the fit quality of the derivative
        regression -- how well the segment behaves like a first-order decay at all. The
        returned tau is in seconds but is **not** calibrated to arterial R*C; see the module
        docstring. ``(nan, 0.0)`` when the beat is too short or is not decaying; the caller
        must gate on ``r2``.
    """
    n = len(beat)
    if n < 12:
        return float("nan"), 0.0

    peak_idx = int(np.argmax(beat))
    start = peak_idx + int(start_frac * (n - peak_idx))
    seg = np.asarray(beat[start:], dtype=np.float64)
    if len(seg) < 8:
        return float("nan"), 0.0

    if seg.std() < 1e-9:
        return float("nan"), 0.0

    # Smooth derivative: differencing raw samples amplifies noise by fs and would dominate
    # the regression. Fall back to plain gradients when the segment is too short for a fit.
    win = min(len(seg) if len(seg) % 2 == 1 else len(seg) - 1, 11)
    if win >= 5:
        dy = savgol_filter(seg, win, polyorder=2, deriv=1, delta=1.0 / fs)
    else:
        dy = np.gradient(seg, 1.0 / fs)

    slope, intercept = np.polyfit(seg, dy, 1)
    if slope >= 0:
        # Not decaying: this is not a diastolic limb. Refuse rather than return a negative
        # time constant that the loss would then try to match in log space.
        return float("nan"), 0.0

    pred = slope * seg + intercept
    ss_res = float(np.sum((dy - pred) ** 2))
    ss_tot = float(np.sum((dy - dy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return float(-1.0 / slope), float(max(0.0, r2))


def _pulse_width_at(beat: np.ndarray, frac: float, fs: float) -> float:
    """Width of the pulse at a given fraction of its height, in seconds."""
    lo, hi = beat.min(), beat.max()
    if hi - lo <= 1e-12:
        return float("nan")
    level = lo + frac * (hi - lo)
    above = np.where(beat >= level)[0]
    if len(above) < 2:
        return float("nan")
    return float((above[-1] - above[0]) / fs)


def _dicrotic_notch(beat: np.ndarray, fs: float) -> tuple[float, float]:
    """Locate the dicrotic notch, returned as (relative time, relative height).

    Found as the strongest local minimum of the second derivative after the systolic peak,
    which is more reliable than looking for a minimum in the signal itself -- in a stiff
    artery the notch flattens into an inflection with no local minimum at all, and the
    curvature still marks it.
    """
    n = len(beat)
    peak = int(np.argmax(beat))
    if n - peak < 8:
        return float("nan"), float("nan")

    _, apg = derivatives(beat, fs)
    search = apg[peak:]
    idx = peak + int(np.argmax(search))

    lo, hi = beat.min(), beat.max()
    rel_h = (beat[idx] - lo) / (hi - lo) if hi - lo > 1e-12 else float("nan")
    return float(idx / n), float(rel_h)


def beat_features(beat: np.ndarray, fs: float) -> dict[str, float]:
    """Morphological features of a single beat."""
    n = len(beat)
    out: dict[str, float] = {}

    lo, hi = float(beat.min()), float(beat.max())
    peak = int(np.argmax(beat))
    amp = hi - lo

    out["sys_amplitude"] = amp
    out["sys_time_s"] = peak / fs
    out["dia_time_s"] = (n - peak) / fs
    out["sys_dia_ratio"] = peak / max(n - peak, 1)

    out["pw25"] = _pulse_width_at(beat, 0.25, fs)
    out["pw50"] = _pulse_width_at(beat, 0.50, fs)
    out["pw75"] = _pulse_width_at(beat, 0.75, fs)
    out["pulse_width_ratio"] = (
        out["pw75"] / out["pw25"] if out["pw25"] and out["pw25"] > 0 else float("nan")
    )

    notch_t, notch_h = _dicrotic_notch(beat, fs)
    out["notch_rel_time"] = notch_t
    out["notch_rel_height"] = notch_h

    # Augmentation index: height of the reflected (late systolic) wave relative to the
    # pulse height. A recognised marker of arterial stiffness, and therefore a direct
    # observable counterpart to the compliance the model predicts.
    out["augmentation_index"] = (
        (beat[int(notch_t * n)] - lo) / amp if amp > 1e-12 and not np.isnan(notch_t) else float("nan")
    )

    vpg, apg = derivatives(beat, fs)
    out["vpg_peak"] = float(vpg.max())
    a_wave = float(apg.max())
    b_wave = float(apg.min())
    out["apg_b_over_a"] = b_wave / a_wave if abs(a_wave) > 1e-12 else float("nan")

    centred = beat - beat.mean()
    sd = centred.std()
    out["skewness"] = float((centred**3).mean() / sd**3) if sd > 1e-12 else float("nan")
    out["kurtosis"] = float((centred**4).mean() / sd**4) if sd > 1e-12 else float("nan")
    out["area_under_pulse"] = float(np.trapezoid(beat - lo) / fs)

    tau, r2 = diastolic_tau(beat, fs)
    out["tau_s"] = tau
    out["tau_fit_r2"] = r2

    return out


def window_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Aggregate beat features across a window into one fixed-length vector.

    Beats are summarised by their **median**, not their mean. A window that survived
    quality control can still contain one corrupted beat, and the median simply ignores it
    where a mean would let it move every feature.

    Returns:
        Vector of length ``len(FEATURE_NAMES)``, ordered to match it. NaN entries are
        possible and are the caller's problem to impute -- silently zero-filling here would
        make an unmeasurable feature indistinguishable from one that measured zero.
    """
    hr, hrv = heart_rate(x, fs)

    beats = segment_beats(x, fs)
    per_beat: list[dict[str, float]] = []
    for s, e in beats:
        seg = x[s:e]
        if len(seg) >= 12:
            per_beat.append(beat_features(seg, fs))

    feats: dict[str, float] = {"hr_bpm": hr, "hrv_sd_ms": hrv}
    if per_beat:
        keys = [k for k in FEATURE_NAMES if k not in ("hr_bpm", "hrv_sd_ms")]
        for k in keys:
            vals = np.array([b.get(k, np.nan) for b in per_beat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            feats[k] = float(np.median(vals)) if len(vals) else float("nan")
    else:
        for k in FEATURE_NAMES:
            feats.setdefault(k, float("nan"))

    return np.array([feats.get(k, float("nan")) for k in FEATURE_NAMES], dtype=np.float64)
