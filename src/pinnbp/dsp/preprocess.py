"""PPG preprocessing: filtering, derivatives, beat detection, windowing.

Nothing here is novel, and that is the point -- this is the standard pipeline the cuffless
BP literature uses, implemented once so that every model in the project sees exactly the
same input. When the PINN is compared against the data-only CNN and the ridge baseline, any
difference has to come from the model, not from one of them getting a quietly better
filter.

Two decisions worth defending:

*Zero-phase filtering.* ``filtfilt`` rather than ``lfilter``. A causal filter imposes a
group delay that varies with frequency, which shifts the systolic peak relative to the
dicrotic notch -- and pulse *timing* is one of the features the physics terms rely on. In a
real-time wearable this would be a genuine constraint and a causal filter with delay
compensation would be needed; offline, there is no reason to accept the distortion.

*Savitzky-Golay derivatives.* The first and second derivatives of the PPG (VPG and APG) are
standard morphological inputs, but plain finite differencing amplifies high-frequency noise
by a factor of fs per differentiation -- at 125 Hz that turns a clean signal into noise.
Savitzky-Golay differentiates a local polynomial fit instead, which is what makes the APG
usable at all.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

__all__ = [
    "bandpass",
    "remove_baseline",
    "normalize_window",
    "derivatives",
    "find_systolic_peaks",
    "segment_beats",
    "sliding_windows",
    "resample_signal",
]


def bandpass(
    x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 8.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass.

    The 0.5-8 Hz band is the usual choice for PPG. The low edge removes baseline wander
    from respiration and slow motion without touching the cardiac fundamental (0.75-3 Hz
    for 45-180 bpm). The high edge keeps enough harmonics for the dicrotic notch to survive
    -- pushing it down to 5 Hz visibly rounds the notch off, and the notch is where the
    reflected-wave information lives.

    Args:
        x: 1-D signal.
        fs: sampling rate in Hz.
        lo: low cutoff in Hz.
        hi: high cutoff in Hz. Clamped below Nyquist.
        order: Butterworth order per pass (filtfilt doubles the effective order).

    Returns:
        Filtered signal, same shape and dtype family as ``x``.
    """
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    if lo >= hi:
        raise ValueError(f"bandpass: lo={lo} must be below hi={hi} (fs={fs})")

    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    # padlen guard: filtfilt raises if the signal is shorter than its default padding, and
    # short windows are exactly what the robustness sweep produces.
    padlen = min(3 * (2 * order + 1), max(len(x) - 1, 0))
    return sps.sosfiltfilt(sos, x, padlen=padlen)


def remove_baseline(x: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    """High-pass only, for cases where the high-frequency content must be kept intact.

    Used by the artifact simulator to strip wander it has itself injected without also
    re-smoothing the signal, which would make the corruption easier to undo than it is in
    reality.
    """
    nyq = fs / 2.0
    sos = sps.butter(2, cutoff / nyq, btype="high", output="sos")
    padlen = min(15, max(len(x) - 1, 0))
    return sps.sosfiltfilt(sos, x, padlen=padlen)


def normalize_window(x: np.ndarray, method: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    """Per-window amplitude normalisation.

    This is a genuinely load-bearing choice, not housekeeping. PPG amplitude depends on
    skin tone, sensor pressure, temperature and probe placement, and none of those carry
    blood-pressure information -- a model that keys on raw amplitude learns the sensor, not
    the subject, and collapses the moment the watch is worn slightly differently.

    So amplitude is discarded and the model is forced onto *morphology and timing*. That is
    also why the physics terms matter: with amplitude gone, compliance can only be
    recovered through the shape and the decay rate of the pulse, which is what the
    Windkessel constraint describes.

    ``robust`` uses median/IQR instead of mean/std and is the better choice when the window
    may contain a motion spike, since a single large excursion barely moves the IQR.

    Args:
        x: 1-D signal.
        method: ``"zscore"``, ``"robust"``, or ``"minmax"``.
        eps: floor on the scale, guarding a flat window.

    Returns:
        Normalised signal.
    """
    x = np.asarray(x, dtype=np.float64)
    if method == "zscore":
        return (x - x.mean()) / (x.std() + eps)
    if method == "robust":
        med = np.median(x)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        return (x - med) / (iqr + eps)
    if method == "minmax":
        lo, hi = x.min(), x.max()
        return 2.0 * (x - lo) / (hi - lo + eps) - 1.0
    raise ValueError(f"unknown normalisation method: {method!r}")


def derivatives(x: np.ndarray, fs: float, window_s: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives (VPG and APG) via Savitzky-Golay.

    The APG in particular is a standard vascular-ageing marker: the ratio of its b-wave to
    its a-wave tracks arterial stiffness, which is the same physical property the model
    represents as compliance C. Handing the network the APG explicitly means it does not
    have to spend capacity learning a second-derivative operator.

    Args:
        x: 1-D signal.
        fs: sampling rate in Hz.
        window_s: smoothing window in seconds. 80 ms is roughly a tenth of a cardiac cycle
            -- long enough to suppress noise, short enough to leave the systolic upstroke.

    Returns:
        ``(vpg, apg)``, each the same length as ``x``, in units of x/s and x/s^2.
    """
    n = max(5, int(round(window_s * fs)))
    if n % 2 == 0:
        n += 1
    n = min(n, len(x) - 1 if len(x) % 2 == 0 else len(x))
    if n < 5:
        # Window too short for a cubic fit; fall back to plain differences rather than
        # failing, and accept the noise.
        vpg = np.gradient(x, 1.0 / fs)
        return vpg, np.gradient(vpg, 1.0 / fs)

    vpg = sps.savgol_filter(x, n, polyorder=3, deriv=1, delta=1.0 / fs)
    apg = sps.savgol_filter(x, n, polyorder=3, deriv=2, delta=1.0 / fs)
    return vpg, apg


def find_systolic_peaks(
    x: np.ndarray, fs: float, min_hr: float = 40.0, max_hr: float = 180.0
) -> np.ndarray:
    """Locate systolic peaks.

    Peak finding is constrained by physiology rather than by a fixed threshold: the minimum
    spacing comes from the maximum plausible heart rate, so the detector cannot report a
    300 bpm rhythm no matter what the noise looks like. The prominence floor is a fraction
    of the window's own interquartile range, which keeps it scale-free after normalisation.

    Args:
        x: 1-D signal, band-passed.
        fs: sampling rate in Hz.
        min_hr: lowest plausible heart rate in bpm (sets the maximum spacing checked).
        max_hr: highest plausible heart rate in bpm (sets minimum peak distance).

    Returns:
        Integer array of peak indices, possibly empty.
    """
    min_distance = max(1, int(fs * 60.0 / max_hr))
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    prominence = 0.25 * iqr if iqr > 0 else None

    peaks, _ = sps.find_peaks(x, distance=min_distance, prominence=prominence)

    # A window shorter than one beat at min_hr cannot support an interval estimate; return
    # what was found and let the caller's SQI decide.
    return peaks


def segment_beats(
    x: np.ndarray, fs: float, peaks: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """Split a signal into beat intervals, foot-to-foot.

    Beats are delimited at the pulse *foot* (the minimum before each systolic upstroke)
    rather than at the peak. The foot is the onset of ejection, so a foot-to-foot segment
    is one cardiac cycle starting where the Windkessel model starts its own beat -- which
    is what makes the measured and simulated waveforms directly comparable.

    Args:
        x: 1-D signal.
        fs: sampling rate.
        peaks: precomputed systolic peaks; detected if omitted.

    Returns:
        List of ``(start, end)`` index pairs, half-open.
    """
    if peaks is None:
        peaks = find_systolic_peaks(x, fs)
    if len(peaks) < 3:
        # Feet are only well defined *between* two detected peaks, so n peaks yield n-1 feet
        # and n-2 beats. Fewer than three peaks means no complete beat.
        return []

    # Deliberately skip the interval before the first peak. Without a preceding peak the
    # search window is arbitrary, and at the start of a filtered recording it lands on the
    # filter's edge transient -- which produced a "beat" whose first sample sat a quarter of
    # a pulse height above its own foot.
    feet: list[int] = []
    for i in range(1, len(peaks)):
        lo, hi = int(peaks[i - 1]), int(peaks[i])
        seg = x[lo:hi]
        if len(seg) == 0:
            continue
        feet.append(lo + int(np.argmin(seg)))

    return [(feet[i], feet[i + 1]) for i in range(len(feet) - 1) if feet[i + 1] > feet[i]]


def sliding_windows(
    x: np.ndarray, fs: float, window_s: float, stride_s: float
) -> np.ndarray:
    """Cut a long recording into fixed-length overlapping windows.

    Overlapping windows from one recording are *not* independent samples. They must never
    be split across train and test -- see ``pinnbp.data.datasets`` for the subject-level
    grouping that prevents it. This function only cuts; the grouping is enforced elsewhere,
    deliberately, so there is one place to audit it.

    Args:
        x: 1-D signal.
        fs: sampling rate.
        window_s: window length in seconds.
        stride_s: hop between window starts in seconds.

    Returns:
        Array of shape ``(n_windows, window_samples)``. Empty if the signal is too short.
    """
    n = int(round(window_s * fs))
    hop = max(1, int(round(stride_s * fs)))
    if len(x) < n:
        return np.empty((0, n), dtype=np.float64)
    starts = range(0, len(x) - n + 1, hop)
    return np.stack([x[s : s + n] for s in starts]).astype(np.float64)


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample to a common rate using polyphase filtering.

    Datasets arrive at different rates -- PPG-BP at 1000 Hz, MIMIC-derived sets at 125 Hz,
    consumer wearables often at 25-64 Hz -- and the model needs one. ``resample_poly``
    rather than ``resample`` because the FFT method assumes periodicity and produces edge
    artifacts on a finite window, which is exactly what this project must not silently add.
    """
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    from math import gcd

    fi, fo = int(round(fs_in)), int(round(fs_out))
    g = gcd(fi, fo)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), fo // g, fi // g)
