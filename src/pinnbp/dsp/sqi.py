"""Signal quality indexing for PPG.

The abstract promises robustness "in real-world conditions where PPG signals may be
affected by motion artifacts". Robustness has two halves, and they are not the same thing:

1. **Knowing when the signal is bad.** That is this module. A cuffless monitor that reports
   118/76 from a window of pure wrist motion is more dangerous than one that reports
   nothing, because the number looks the same as a real one.
2. **Degrading gracefully when it is bad anyway.** That is the physics constraints.

The SQI here is a weighted combination of four established indices rather than a single
one, because each fails differently: skewness is fooled by a clipped signal, perfusion by
a large-amplitude artifact, template matching by a genuine arrhythmia. Their combination is
harder to fool than any of them alone, and each component stays individually inspectable so
a rejection can be explained rather than merely asserted.

References for the individual indices: Elgendi (2016) on skewness-based PPG SQI; Orphanidou
et al. (2015) on template-matching quality for wearable recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import find_systolic_peaks, segment_beats

__all__ = ["SQIResult", "signal_quality", "skewness_sqi", "perfusion_index", "template_match_sqi"]


@dataclass
class SQIResult:
    """A quality verdict with its components kept visible.

    ``overall`` is what gates the pipeline, but a rejection is only actionable if you can
    see *which* index objected -- a low ``template`` with a healthy ``perfusion`` means an
    irregular rhythm, while both low together means the sensor lost contact.
    """

    overall: float
    skewness: float
    perfusion: float
    template: float
    beat_regularity: float
    n_beats: int

    @property
    def acceptable(self) -> bool:
        """Whether the window is fit for BP estimation.

        The 0.5 threshold is a convention adopted here, not a validated clinical cut-off;
        the robustness sweep reports results across the whole SQI range so the choice can
        be revisited against evidence rather than inherited.
        """
        return self.overall >= 0.5


def skewness_sqi(x: np.ndarray) -> float:
    """Skewness-based quality, mapped to [0, 1].

    A clean PPG pulse is asymmetric -- a fast systolic upstroke and a slow diastolic decay
    -- which gives it a consistently positive skew, empirically near 0.5-1.5. Noise is
    closer to symmetric and pushes the skew toward zero. The mapping is a Gaussian centred
    on 1.0, so both an unskewed (noisy) and an implausibly skewed (spiking) window score
    low.
    """
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    if sd < 1e-12:
        return 0.0
    s = float(((x - x.mean()) ** 3).mean() / sd**3)
    return float(np.exp(-((s - 1.0) ** 2) / (2 * 0.9**2)))


def perfusion_index(x: np.ndarray) -> float:
    """Pulsatile-to-static ratio, mapped to [0, 1].

    Perfusion index is the clinical measure of how much of the signal is actually
    pulsatile. Because windows reaching this function have usually been amplitude
    normalised, it is computed as the interquartile range relative to the total range,
    which is scale-free: a window dominated by one motion excursion has a large total range
    and a small IQR, and scores low.
    """
    x = np.asarray(x, dtype=np.float64)
    rng = float(x.max() - x.min())
    if rng < 1e-12:
        return 0.0
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    ratio = iqr / rng
    # A clean pulse train sits near 0.35-0.55; normalise so that band maps to ~1.0.
    return float(np.clip(ratio / 0.45, 0.0, 1.0))


def template_match_sqi(x: np.ndarray, fs: float) -> tuple[float, float, int]:
    """Beat-to-beat template correlation and interval regularity.

    Individual beats are resampled to a common length, averaged into a template, and each
    beat is correlated against it. Consistency across beats is strong evidence of a real
    pulse: motion artifact may look pulse-like for one beat but rarely repeats the same
    shape several times running.

    Resampling to a common length before averaging is essential -- averaging beats of
    different durations directly smears the template and makes every correlation look bad,
    including the good ones.

    Returns:
        ``(template_correlation, interval_regularity, n_beats)``, correlations in [0, 1].
    """
    beats = segment_beats(x, fs)
    if len(beats) < 3:
        return 0.0, 0.0, len(beats)

    L = 100
    resampled: list[np.ndarray] = []
    for s, e in beats:
        seg = x[s:e]
        if len(seg) < 8:
            continue
        idx = np.linspace(0, len(seg) - 1, L)
        r = np.interp(idx, np.arange(len(seg)), seg)
        sd = r.std()
        if sd > 1e-12:
            resampled.append((r - r.mean()) / sd)

    if len(resampled) < 3:
        return 0.0, 0.0, len(resampled)

    M = np.stack(resampled)
    template = M.mean(axis=0)
    tsd = template.std()
    if tsd < 1e-12:
        return 0.0, 0.0, len(resampled)
    template = (template - template.mean()) / tsd

    corrs = (M @ template) / L
    tmpl = float(np.clip(np.median(corrs), 0.0, 1.0))

    # Interval regularity: coefficient of variation of beat durations. Genuine sinus
    # arrhythmia gives a CV around 0.05-0.10; artifact-driven false beats give much more.
    durations = np.array([e - s for s, e in beats], dtype=np.float64)
    cv = float(durations.std() / durations.mean()) if durations.mean() > 0 else 1.0
    regularity = float(np.exp(-((cv / 0.15) ** 2)))

    return tmpl, regularity, len(resampled)


def signal_quality(x: np.ndarray, fs: float) -> SQIResult:
    """Combine the individual indices into one verdict.

    The weights favour template matching because it is the hardest of the four to fool: a
    corrupted window has to reproduce a consistent pulse shape several beats in a row to
    pass it, whereas skewness or perfusion can be satisfied by a single well-shaped
    excursion.

    A window with fewer than three detectable beats is capped at 0.3 regardless of what the
    other indices say. With one or two beats there is no repetition to check, so the
    strongest evidence is simply unavailable and the score should reflect that rather than
    defaulting to whatever the amplitude statistics happen to give.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16 or not np.all(np.isfinite(x)):
        return SQIResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    skew = skewness_sqi(x)
    perf = perfusion_index(x)
    tmpl, reg, n_beats = template_match_sqi(x, fs)

    overall = 0.20 * skew + 0.20 * perf + 0.40 * tmpl + 0.20 * reg

    if n_beats < 3:
        overall = min(overall, 0.3)

    peaks = find_systolic_peaks(x, fs)
    if len(peaks) == 0:
        overall = 0.0

    return SQIResult(
        overall=float(np.clip(overall, 0.0, 1.0)),
        skewness=skew,
        perfusion=perf,
        template=tmpl,
        beat_regularity=reg,
        n_beats=n_beats,
    )
