"""Tests for filtering, feature extraction, and signal quality.

The tau test is the one that matters beyond correctness: it feeds a physics loss, so a
silently wrong tau would not crash anything -- it would teach the network a wrong compliance
and quietly degrade the parameter-recovery result the project reports as evidence.


"""

from __future__ import annotations

import numpy as np
import pytest

from pinnbp.data.artifacts import apply_artifacts
from pinnbp.data.synthetic import sample_subject, simulate_subject
from pinnbp.dsp.features import diastolic_tau, heart_rate, window_features
from pinnbp.dsp.preprocess import (
    bandpass,
    derivatives,
    find_systolic_peaks,
    normalize_window,
    resample_signal,
    segment_beats,
    sliding_windows,
)
from pinnbp.dsp.sqi import signal_quality

FS = 125.0


def synthetic_ppg(seconds: float = 10.0, hypertensive: bool = False, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = sample_subject(rng, 0, hypertensive)
    rec = simulate_subject(p, seconds, FS, rng)
    return rec["ppg"], p


# --------------------------------------------------------------------------- filtering


def test_bandpass_removes_drift_and_keeps_the_cardiac_band():
    t = np.arange(0, 10, 1 / FS)
    cardiac = np.sin(2 * np.pi * 1.2 * t)
    drift = 5.0 * np.sin(2 * np.pi * 0.05 * t)

    out = bandpass(cardiac + drift, FS)

    # Drift is 5x the cardiac amplitude going in; it must be essentially gone coming out.
    assert np.std(out - cardiac) < 0.2 * np.std(cardiac)


def test_bandpass_rejects_an_inverted_band():
    with pytest.raises(ValueError):
        bandpass(np.zeros(500), FS, lo=9.0, hi=0.5)


def test_bandpass_handles_a_window_shorter_than_default_padding():
    """Short windows are exactly what the robustness sweep produces; filtfilt must not raise."""
    short = np.sin(np.linspace(0, 6, 40))
    out = bandpass(short, FS)
    assert out.shape == short.shape and np.all(np.isfinite(out))


def test_zero_phase_filtering_does_not_shift_the_signal_in_time():
    """filtfilt must preserve timing where a causal filter of the same design does not.

    Group delay is measured by cross-correlating each filtered signal against the source,
    which is unambiguous. Matching peak *indices* is not: the causal filter's startup
    transient can swallow the first peak, after which peak i of one signal is peak i+1 of
    the other and the apparent lag comes out negative.

    Timing matters here beyond tidiness -- pulse transit and the position of the dicrotic
    notch are how compliance reaches the model once amplitude has been destroyed by motion.
    """
    from scipy import signal as sps

    x, _ = synthetic_ppg(12.0)
    nyq = FS / 2
    sos = sps.butter(4, [0.5 / nyq, 8.0 / nyq], btype="band", output="sos")

    zero_phase = bandpass(x, FS)
    causal = sps.sosfilt(sos, x)

    def lag_samples(a, b, max_lag=60):
        """Lag of b relative to a, by peak cross-correlation."""
        trim = slice(int(2 * FS), -int(1 * FS))  # drop filter transients at both ends
        a = np.asarray(a[trim], dtype=float)
        b = np.asarray(b[trim], dtype=float)
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        lags = np.arange(-max_lag, max_lag + 1)
        scores = [np.dot(a[max_lag:-max_lag], np.roll(b, -k)[max_lag:-max_lag]) for k in lags]
        return int(lags[int(np.argmax(scores))])

    src = x - np.mean(x)
    assert abs(lag_samples(src, zero_phase)) <= 2, "filtfilt should not shift the signal"

    # Magnitude only: the sign depends on the roll convention, the physics does not.
    causal_lag = abs(lag_samples(src, causal))
    assert causal_lag > 3, (
        f"causal filter showed no group delay (lag {causal_lag}); "
        "the test is not isolating what it claims to"
    )


def test_normalisation_discards_amplitude():
    """A model must not be able to read blood pressure off raw sensor gain."""
    x, _ = synthetic_ppg(8.0)
    a = normalize_window(x, "robust")
    b = normalize_window(3.7 * x, "robust")
    assert np.allclose(a, b, atol=1e-9)


def test_robust_normalisation_resists_a_single_spike():
    x, _ = synthetic_ppg(8.0)
    spiked = x.copy()
    spiked[400] += 50 * np.std(x)

    z_shift = np.abs(normalize_window(spiked, "zscore") - normalize_window(x, "zscore"))
    r_shift = np.abs(normalize_window(spiked, "robust") - normalize_window(x, "robust"))
    # Ignore the spike sample itself; the question is what happened to everything else.
    mask = np.ones(len(x), dtype=bool)
    mask[400] = False
    assert np.median(r_shift[mask]) < np.median(z_shift[mask])


def test_unknown_normalisation_raises():
    with pytest.raises(ValueError):
        normalize_window(np.zeros(10), "quantile")


def test_savgol_derivatives_beat_finite_differences_on_noise():
    """The APG is unusable under plain differencing; that is why Savitzky-Golay is used."""
    t = np.arange(0, 8, 1 / FS)
    clean = np.sin(2 * np.pi * 1.2 * t)
    noisy = clean + np.random.default_rng(0).normal(0, 0.02, len(t))

    true_apg = -((2 * np.pi * 1.2) ** 2) * clean
    _, sg_apg = derivatives(noisy, FS)
    naive_apg = np.gradient(np.gradient(noisy, 1 / FS), 1 / FS)

    inner = slice(30, -30)
    assert np.std(sg_apg[inner] - true_apg[inner]) < np.std(naive_apg[inner] - true_apg[inner])


def test_resample_preserves_duration_and_frequency():
    t = np.arange(0, 4, 1 / 1000.0)
    x = np.sin(2 * np.pi * 1.5 * t)
    out = resample_signal(x, 1000.0, 125.0)
    assert abs(len(out) - 500) <= 2
    peaks = find_systolic_peaks(out, 125.0)
    assert abs(np.mean(np.diff(peaks)) / 125.0 - (1 / 1.5)) < 0.05


# --------------------------------------------------------------------------- beats


def test_heart_rate_recovers_the_simulated_rate():
    rng = np.random.default_rng(3)
    p = sample_subject(rng, 0, False)
    rec = simulate_subject(p, 20.0, FS, rng)
    hr, sd = heart_rate(bandpass(rec["ppg"], FS), FS)
    assert hr == pytest.approx(p.HR, rel=0.06)
    assert 0 <= sd < 200


def test_heart_rate_returns_nan_when_there_is_no_pulse():
    """NaN, not a plausible default: a fabricated 70 bpm would be undetectable downstream."""
    hr, _ = heart_rate(np.zeros(1000), FS)
    assert np.isnan(hr)


def test_beats_are_segmented_foot_to_foot():
    x, p = synthetic_ppg(12.0)
    x = bandpass(x, FS)
    beats = segment_beats(x, FS)
    assert len(beats) >= 8
    durations = np.array([(e - s) / FS for s, e in beats])
    assert durations.mean() == pytest.approx(60.0 / p.HR, rel=0.10)
    # Each segment must *start* at a foot, which is what foot-to-foot means. A beat both
    # begins and ends at a foot, so argmin can legitimately land at either end; the
    # meaningful assertion is that the first sample sits at the bottom of the pulse.
    for s, e in beats:
        seg = x[s:e]
        amp = seg.max() - seg.min()
        assert seg[0] - seg.min() < 0.15 * amp
        assert int(np.argmax(seg)) < 0.6 * len(seg)


def test_sliding_windows_shape_and_short_input():
    x = np.arange(1000, dtype=float)
    w = sliding_windows(x, FS, window_s=2.0, stride_s=1.0)
    assert w.shape[1] == 250
    assert w.shape[0] == (1000 - 250) // 125 + 1
    assert sliding_windows(np.arange(10, dtype=float), FS, 2.0, 1.0).shape[0] == 0


# --------------------------------------------------------------------------- tau


def test_diastolic_tau_recovers_a_known_decay():
    """Fitted against a synthetic exponential where the answer is known exactly."""
    tau_true = 1.4
    n = 200
    t = np.arange(n) / FS
    # A rising limb followed by an exponential decay, i.e. the shape of a real pulse.
    beat = np.concatenate([np.linspace(0, 1, 30), np.exp(-t[: n - 30] / tau_true)])
    tau, r2 = diastolic_tau(beat, FS)
    assert r2 > 0.98
    assert tau == pytest.approx(tau_true, rel=0.15)


def test_diastolic_tau_refuses_a_rising_signal():
    """A negative time constant would be matched in log space and poison the physics loss."""
    tau, r2 = diastolic_tau(np.linspace(0, 1, 100), FS)
    assert np.isnan(tau) and r2 == 0.0


def test_diastolic_tau_refuses_a_flat_signal():
    tau, r2 = diastolic_tau(np.ones(100), FS)
    assert np.isnan(tau)


def test_measured_tau_tracks_the_simulated_rc_product():
    """The PPG decay constant must *rank* subjects by R*C, which is all the loss uses.

    Deliberately asserts correlation and not agreement. The band-passed PPG value is roughly
    four times smaller than true R*C -- the 0.5 Hz high-pass removes most of a 2 s
    exponential -- so ``tau_consistency`` standardises within the batch and uses only the
    ordering. Asserting absolute agreement here would encode the very mistake that loss was
    rewritten to avoid.
    """
    measured, true = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        p = sample_subject(rng, seed, hypertensive=bool(seed % 3 == 0))
        rec = simulate_subject(p, 20.0, FS, rng)
        x = bandpass(rec["ppg"], FS)

        taus = []
        for s, e in segment_beats(x, FS):
            if e - s >= 12:
                tv, r2 = diastolic_tau(x[s:e], FS)
                if np.isfinite(tv) and r2 >= 0.5 and 0.2 < tv < 5.0:
                    taus.append(tv)
        if len(taus) >= 2:
            measured.append(float(np.median(taus)))
            true.append(p.R * p.C)

    assert len(measured) >= 8, f"only {len(measured)} of 30 subjects yielded a usable tau"
    r = float(np.corrcoef(measured, true)[0, 1])
    assert r > 0.35, f"measured tau barely tracks R*C (r={r:.2f})"


def test_measured_tau_is_biased_low_which_is_why_the_loss_is_scale_free():
    """Pin down the bias that motivates the standardised tau loss.

    If a future change to the filter or the transduction ever made the PPG value agree with
    R*C absolutely, this test fails -- and that would be the signal to revisit
    ``tau_consistency``, which currently discards scale information on purpose.
    """
    ratios = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        p = sample_subject(rng, seed, hypertensive=False)
        rec = simulate_subject(p, 20.0, FS, rng)
        x = bandpass(rec["ppg"], FS)
        taus = [
            tv
            for s, e in segment_beats(x, FS)
            if e - s >= 12
            for tv, r2 in [diastolic_tau(x[s:e], FS)]
            if np.isfinite(tv) and r2 >= 0.5 and 0.2 < tv < 5.0
        ]
        if len(taus) >= 2:
            ratios.append(float(np.median(taus)) / (p.R * p.C))

    assert len(ratios) >= 5
    assert np.median(ratios) < 0.6, (
        f"PPG tau is no longer strongly biased low (ratio {np.median(ratios):.2f}); "
        "revisit tau_consistency, which discards scale on the assumption that it is"
    )


def test_window_features_have_the_declared_length_and_are_mostly_finite():
    from pinnbp.dsp.features import FEATURE_NAMES

    x, _ = synthetic_ppg(10.0)
    f = window_features(bandpass(x, FS), FS)
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).mean() > 0.8


# --------------------------------------------------------------------------- SQI


def test_clean_signal_scores_higher_than_corrupted():
    x, _ = synthetic_ppg(10.0)
    clean = normalize_window(bandpass(x, FS), "robust")
    dirty_raw = apply_artifacts(x, FS, 0.9, np.random.default_rng(1))
    dirty = normalize_window(bandpass(dirty_raw, FS), "robust")

    assert signal_quality(clean, FS).overall > signal_quality(dirty, FS).overall


def test_pure_noise_and_flat_line_are_rejected():
    noise = np.random.default_rng(0).normal(0, 1, 1000)
    assert not signal_quality(noise, FS).acceptable
    assert signal_quality(np.zeros(1000), FS).overall == 0.0


def test_too_few_beats_caps_the_score():
    """With fewer than three beats there is no repetition to check, so confidence is capped."""
    x, _ = synthetic_ppg(10.0)
    x = normalize_window(bandpass(x, FS), "robust")
    short = x[: int(1.5 * FS)]
    q = signal_quality(short, FS)
    assert q.n_beats < 3
    assert q.overall <= 0.3


def test_sqi_components_are_individually_reported():
    """A rejection has to be explainable, not merely asserted."""
    x, _ = synthetic_ppg(10.0)
    q = signal_quality(normalize_window(bandpass(x, FS), "robust"), FS)
    for v in (q.skewness, q.perfusion, q.template, q.beat_regularity):
        assert 0.0 <= v <= 1.0
    assert q.n_beats >= 5
