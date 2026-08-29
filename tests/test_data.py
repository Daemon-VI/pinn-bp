"""Tests for the simulator, the artifact models, and the splitting.

The split tests are non-negotiable. Subject leakage is the single failure that would
invalidate every number the project reports while leaving the code apparently working and
the results looking better, so it is checked from several directions.


"""

from __future__ import annotations

import numpy as np
import pytest

from pinnbp.data.artifacts import ARTIFACT_KINDS, DEFAULT_ARTIFACT_KINDS, apply_artifacts
from pinnbp.data.datasets import subject_split
from pinnbp.data.synthetic import CohortConfig, generate_cohort, sample_subject, simulate_subject

# --------------------------------------------------------------------------- splits


def test_subject_split_partitions_are_disjoint_in_subjects():
    subjects = np.repeat(np.arange(50), 7)
    s = subject_split(subjects, seed=0)
    assert set(s.train_subjects).isdisjoint(s.val_subjects)
    assert set(s.train_subjects).isdisjoint(s.test_subjects)
    assert set(s.val_subjects).isdisjoint(s.test_subjects)


def test_subject_split_covers_every_window_exactly_once():
    """No window may be dropped or duplicated; either would silently distort the metrics."""
    subjects = np.repeat(np.arange(37), 5)
    s = subject_split(subjects, seed=3)
    allidx = np.concatenate([s.train, s.val, s.test])
    assert len(allidx) == len(subjects)
    assert len(np.unique(allidx)) == len(subjects)


def test_subject_split_is_deterministic_for_a_seed():
    """The sweep depends on this: the same seed must select the same test subjects."""
    subjects = np.repeat(np.arange(40), 6)
    a = subject_split(subjects, seed=11)
    b = subject_split(subjects, seed=11)
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(a.test_subjects, b.test_subjects)


def test_random_split_does_leak_and_is_therefore_only_a_demonstration():
    """The leakage mode must actually leak, or the demonstration would prove nothing."""
    subjects = np.repeat(np.arange(30), 20)
    s = subject_split(subjects, seed=0, mode="random")
    assert set(s.train_subjects) & set(s.test_subjects)


def test_unknown_split_mode_raises():
    with pytest.raises(ValueError):
        subject_split(np.arange(10), mode="stratified")


# --------------------------------------------------------------------------- simulator


def test_cohort_pressures_are_physiological():
    c = generate_cohort(CohortConfig(n_subjects=25, duration_s=20.0, seed=2))
    sbp, dbp = c["sbp"], c["dbp"]
    assert np.all(sbp > dbp)
    assert 85 < sbp.mean() < 165
    assert 50 < dbp.mean() < 105
    pp = sbp - dbp
    assert np.all(pp > 10) and np.all(pp < 120)


def test_cohort_reproduces_known_physiological_correlations():
    """Compliance must lower pulse pressure and resistance must raise mean pressure.

    If these were absent the cohort would be arbitrary numbers wearing physiological units,
    and the physics constraints would have nothing real to latch onto.
    """
    c = generate_cohort(CohortConfig(n_subjects=60, duration_s=20.0, seed=5))
    R = c["params"][:, 0]
    C = c["params"][:, 1]
    sbp, dbp = c["sbp"], c["dbp"]
    pp = sbp - dbp
    map_ = dbp + pp / 3.0

    assert np.corrcoef(C, pp)[0, 1] < -0.3
    assert np.corrcoef(R, map_)[0, 1] > 0.3


def test_cohort_is_reproducible_from_seed():
    a = generate_cohort(CohortConfig(n_subjects=6, duration_s=15.0, seed=9))
    b = generate_cohort(CohortConfig(n_subjects=6, duration_s=15.0, seed=9))
    assert np.allclose(a["ppg"], b["ppg"])
    assert np.allclose(a["sbp"], b["sbp"])


def test_artifact_severity_does_not_perturb_the_cohort():
    """The property that makes the robustness sweep a valid experiment.

    Physiology and artifacts draw from independent streams, so changing severity must leave
    labels, subjects and parameters bit-identical and change only the signals. Without this
    each severity level would be a different cohort.
    """
    kw = dict(n_subjects=8, duration_s=20.0, seed=4, artifact_prob=1.0)
    clean = generate_cohort(CohortConfig(artifact_severity=0.0, **kw))
    dirty = generate_cohort(CohortConfig(artifact_severity=0.8, **kw))

    assert np.array_equal(clean["sbp"], dirty["sbp"])
    assert np.array_equal(clean["dbp"], dirty["dbp"])
    assert np.array_equal(clean["subject"], dirty["subject"])
    assert np.allclose(clean["params"], dirty["params"])
    assert not np.allclose(clean["ppg"], dirty["ppg"])


def test_higher_severity_lowers_signal_quality_monotonically():
    means = []
    for sev in (0.0, 0.3, 0.6, 0.9):
        c = generate_cohort(
            CohortConfig(n_subjects=8, duration_s=20.0, seed=4,
                         artifact_severity=sev, artifact_prob=1.0)
        )
        means.append(float(c["sqi"].mean()))
    assert means == sorted(means, reverse=True), means
    assert means[0] - means[-1] > 0.15


def test_simulated_ptt_is_physiological_and_falls_with_stiffness():
    """Pulse transit time must land in a real range and shorten as arteries stiffen.

    This is the timing route by which compliance reaches the PPG. If PTT were unrealistic or
    had the wrong sign, the physics constraint would be fitting a relationship the data does
    not contain.
    """
    rng = np.random.default_rng(0)
    soft = sample_subject(rng, 0, hypertensive=False)
    soft.C = 2.4
    stiff = sample_subject(rng, 1, hypertensive=True)
    stiff.C = 0.9

    ptt_soft = simulate_subject(soft, 10.0, 125.0, np.random.default_rng(1))["ptt"]
    ptt_stiff = simulate_subject(stiff, 10.0, 125.0, np.random.default_rng(1))["ptt"]

    assert 0.05 < ptt_stiff < ptt_soft < 0.35
    assert ptt_stiff < ptt_soft


# --------------------------------------------------------------------------- artifacts


def test_zero_severity_is_a_no_op():
    """A clean control must be exactly clean, so a sweep can include severity 0."""
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 20, 1000))
    assert np.array_equal(apply_artifacts(x, 125.0, 0.0, rng), x)


def test_every_artifact_kind_changes_the_signal_and_stays_finite():
    x = np.sin(np.linspace(0, 20, 1000))
    for kind in ARTIFACT_KINDS:
        out = apply_artifacts(x, 125.0, 0.8, np.random.default_rng(1), kinds=(kind,), n_kinds=1)
        assert out.shape == x.shape, kind
        assert np.all(np.isfinite(out)), kind
        assert not np.allclose(out, x), kind


def test_contact_loss_is_excluded_from_the_default_pool():
    """It removes information rather than degrading it, so it must be opt-in.

    Mixing it into the general pool would contaminate the graceful-degradation curve with
    windows carrying no recoverable signal at all. Checked against the declared pool rather
    than by sniffing the output, because several mechanisms leave flat stretches and a
    signal-shape heuristic cannot tell them apart.
    """
    assert "contact_loss" in ARTIFACT_KINDS
    assert "contact_loss" not in DEFAULT_ARTIFACT_KINDS
    assert set(DEFAULT_ARTIFACT_KINDS) == set(ARTIFACT_KINDS) - {"contact_loss"}


def test_contact_loss_still_reachable_when_named_explicitly():
    x = np.sin(np.linspace(0, 20, 2000))
    out = apply_artifacts(x, 125.0, 0.9, np.random.default_rng(0),
                          kinds=("contact_loss",), n_kinds=1)
    assert not np.allclose(out, x)


def test_amplitude_drift_preserves_pulse_timing():
    """The mechanism the physics constraints are supposed to survive.

    A multiplicative gain change must leave peak *positions* untouched while altering their
    heights. If it moved the peaks, the experiment could not distinguish a model that reads
    timing from one that reads amplitude.
    """
    from scipy.signal import find_peaks

    fs = 125.0
    t = np.arange(0, 8, 1 / fs)
    x = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)

    out = apply_artifacts(x, fs, 0.9, np.random.default_rng(2),
                          kinds=("amplitude_drift",), n_kinds=1)

    p_before, _ = find_peaks(x, distance=int(fs * 0.4))
    p_after, _ = find_peaks(out, distance=int(fs * 0.4))
    assert len(p_before) == len(p_after)
    assert np.max(np.abs(p_before - p_after)) <= 1


def test_clipping_preserves_the_diastolic_limb():
    """Clipping must remove the peak while leaving the lower part of the pulse intact."""
    x = np.sin(np.linspace(0, 20, 1000))
    out = apply_artifacts(x, 125.0, 0.8, np.random.default_rng(3),
                          kinds=("clipping",), n_kinds=1)
    low = x < np.percentile(x, 40)
    assert np.allclose(out[low], x[low])
    assert out.max() < x.max()
