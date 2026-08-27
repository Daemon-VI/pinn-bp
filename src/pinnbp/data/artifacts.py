"""Motion artifact and noise models.

The abstract's central claim is robustness "in real-world conditions where PPG signals may
be affected by motion artifacts, noise, or changes in the user's physiological state". That
claim is only testable if the corruptions are (a) realistic in kind and (b) controllable in
degree, so this module provides six mechanisms on a shared 0-1 severity scale and the
robustness sweep in ``pinnbp.evaluate`` walks that scale.

The six are not interchangeable, and picking a mix matters more than picking a severity.
They damage *different* information:

===================  ==========================================================
Mechanism            What it destroys
===================  ==========================================================
baseline wander      low-frequency content; leaves pulse morphology intact
motion burst         whole beats; leaves the rest of the window clean
amplitude drift      relative amplitude; leaves timing intact
clipping             peak morphology; leaves the diastolic limb intact
additive noise       fine morphology, especially the second derivative
contact loss         everything, for a stretch
===================  ==========================================================

That table is the experiment. Amplitude drift and clipping preserve *timing* while
destroying *amplitude*, which is precisely the regime where the physics constraints should
help -- compliance stays recoverable through the pulse-transit and reflection route even
after the amplitude route is gone. Contact loss destroys both and should defeat every model
equally; if the PINN appeared to survive that, it would be evidence of a bug or of label
leakage, not of robustness.

A deliberate choice: corruption is applied to the **raw** window, before filtering and
normalisation, exactly as it would occur at the sensor. Applying it afterwards would let
the preprocessing chain, which never saw the artifact, appear far more effective than it is.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

__all__ = [
    "ARTIFACT_KINDS",
    "DEFAULT_ARTIFACT_KINDS",
    "apply_artifacts",
    "baseline_wander",
    "motion_burst",
    "amplitude_drift",
    "clipping",
    "additive_noise",
    "contact_loss",
]

ARTIFACT_KINDS = (
    "baseline_wander",
    "motion_burst",
    "amplitude_drift",
    "clipping",
    "additive_noise",
    "contact_loss",
)

# What a random draw actually selects from. Contact loss is deliberately absent: it removes
# the information rather than degrading it, so mixing it into the general pool would
# contaminate the graceful-degradation curve with windows that carry no recoverable signal.
# It stays available through an explicit ``kinds=("contact_loss",)`` for the per-mechanism
# breakdown, where it is the control that should defeat every model equally.
DEFAULT_ARTIFACT_KINDS = tuple(k for k in ARTIFACT_KINDS if k != "contact_loss")


def _scale(x: np.ndarray) -> float:
    """Robust amplitude scale of a window, used to size artifacts relative to the signal.

    IQR rather than std or peak-to-peak: the whole point is to size the artifact against
    the *pulse*, and a window that already contains an excursion would give a misleading
    std. Corrupting proportionally keeps severity comparable across subjects with different
    perfusion.
    """
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    return iqr if iqr > 1e-9 else float(np.std(x)) or 1.0


def baseline_wander(x: np.ndarray, fs: float, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency drift from limb movement and venous pooling.

    A sum of three components between 0.05 and 0.5 Hz, which overlaps the lower edge of the
    0.5 Hz bandpass on purpose. An artifact entirely outside the passband would be removed
    perfectly by the filter and would test nothing.
    """
    n = len(x)
    t = np.arange(n) / fs
    amp = 2.5 * severity * _scale(x)
    out = np.zeros(n)
    for _ in range(3):
        f = rng.uniform(0.05, 0.5)
        out += rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return x + amp * out / 3.0


def motion_burst(x: np.ndarray, fs: float, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Short high-amplitude transients from a hand or wrist movement.

    Each burst is a windowed chirp: real motion artifact sweeps in frequency as the limb
    accelerates and decelerates, so a fixed-frequency burst is both easier to filter and
    less representative. The Tukey envelope prevents step discontinuities at the burst
    edges, which would otherwise ring through the filter and dominate everything.
    """
    n = len(x)
    out = x.copy()
    n_bursts = int(rng.integers(1, max(2, int(1 + 3 * severity))))
    amp = 3.0 * severity * _scale(x)

    for _ in range(n_bursts):
        dur = rng.uniform(0.15, 0.6)
        length = min(int(dur * fs), n)
        if length < 4:
            continue
        start = int(rng.integers(0, max(1, n - length)))
        tt = np.arange(length) / fs
        f0, f1 = rng.uniform(0.5, 3.0), rng.uniform(3.0, 12.0)
        burst = sps.chirp(tt, f0=f0, t1=tt[-1] if tt[-1] > 0 else 1.0, f1=f1)
        burst = burst * sps.windows.tukey(length, alpha=0.5)
        out[start : start + length] += amp * rng.uniform(0.5, 1.0) * burst

    return out


def amplitude_drift(x: np.ndarray, fs: float, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Slow multiplicative gain change from varying sensor contact pressure.

    Multiplicative, not additive, and this is the interesting one: it scales pulse height
    while leaving every peak, foot and notch exactly where it was. A model that reads BP off
    amplitude degrades immediately; a model that reads it off timing and shape does not.
    """
    n = len(x)
    t = np.arange(n) / fs
    f = rng.uniform(0.02, 0.25)
    depth = 0.75 * severity
    gain = 1.0 + depth * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    gain = np.clip(gain, 0.15, None)
    centre = np.median(x)
    return centre + (x - centre) * gain


def clipping(x: np.ndarray, fs: float, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Saturation of the photodetector or its amplifier.

    Clips the top of the pulse, which removes the systolic peak while leaving the diastolic
    decay -- and therefore tau -- measurable. The complement of ``amplitude_drift``: it
    destroys the peak that a naive model keys on and preserves the decay that the physics
    term uses.
    """
    if severity <= 0:
        return x.copy()
    lo_q = 100.0 - 45.0 * severity
    hi = float(np.percentile(x, np.clip(lo_q, 50.0, 100.0)))
    return np.minimum(x, hi)


def additive_noise(x: np.ndarray, fs: float, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Broadband sensor and quantisation noise, with a pink component.

    Pure white noise is unrealistically easy to filter. Real optical front ends are
    dominated by 1/f noise at the low end, which sits inside the PPG passband and cannot be
    removed. The mix here is roughly two-thirds white, one-third pink.
    """
    n = len(x)
    amp = 0.9 * severity * _scale(x)

    white = rng.normal(0, 1, n)

    # Pink noise by spectral shaping: 1/sqrt(f) magnitude on a white spectrum.
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    pink = np.fft.irfft(spec / np.sqrt(freqs), n=n)
    pink = pink / (pink.std() + 1e-12)

    return x + amp * (0.66 * white + 0.34 * pink)


def contact_loss(x: np.ndarray, fs: float, severity: float, rng: np.random.Generator) -> np.ndarray:
    """The sensor lifts off the skin: the pulse vanishes into flat, noisy baseline.

    The hard case, and included precisely because it *should* defeat every model. A method
    that claims robustness has to be shown failing somewhere; if it did not, the sweep would
    only be measuring how gentle the corruptions were. The correct behaviour here is for the
    SQI to reject the window, not for the model to guess well.
    """
    n = len(x)
    frac = 0.55 * severity
    length = int(frac * n)
    if length < 2:
        return x.copy()
    start = int(rng.integers(0, max(1, n - length)))
    out = x.copy()
    level = float(np.median(x))
    out[start : start + length] = level + rng.normal(0, 0.06 * _scale(x), length)
    return out


_DISPATCH = {
    "baseline_wander": baseline_wander,
    "motion_burst": motion_burst,
    "amplitude_drift": amplitude_drift,
    "clipping": clipping,
    "additive_noise": additive_noise,
    "contact_loss": contact_loss,
}


def apply_artifacts(
    x: np.ndarray,
    fs: float,
    severity: float,
    rng: np.random.Generator,
    kinds: tuple[str, ...] | None = None,
    n_kinds: int | None = None,
) -> np.ndarray:
    """Apply a random subset of artifact mechanisms at a given severity.

    Severity 0 returns the signal untouched, so a sweep can include a clean control without
    special-casing it.

    ``contact_loss`` is excluded from the default random draw and must be requested
    explicitly. It is qualitatively different from the others -- it removes the information
    rather than degrading it -- so mixing it into the general pool would contaminate the
    graceful-degradation curve with windows that carry no recoverable signal at all, and the
    resulting average would describe neither regime.

    Args:
        x: raw window, before filtering.
        fs: sampling rate.
        severity: 0 (clean) to 1 (severe).
        rng: seeded generator.
        kinds: restrict to these mechanisms; defaults to all but contact loss.
        n_kinds: how many to combine. Defaults to 1-3 scaled by severity, since real
            corrupted recordings usually suffer several mechanisms at once.

    Returns:
        The corrupted window.
    """
    if severity <= 0:
        return np.asarray(x, dtype=np.float64).copy()

    pool = kinds if kinds is not None else DEFAULT_ARTIFACT_KINDS
    if n_kinds is None:
        n_kinds = int(np.clip(1 + round(2 * severity), 1, len(pool)))
    n_kinds = min(n_kinds, len(pool))

    chosen = rng.choice(np.array(pool, dtype=object), size=n_kinds, replace=False)

    out = np.asarray(x, dtype=np.float64).copy()
    for kind in chosen:
        out = _DISPATCH[str(kind)](out, fs, severity, rng)
    return out
