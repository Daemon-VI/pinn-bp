"""A physiologically grounded PPG + arterial-pressure simulator.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This generates simulated data. Numbers measured on it validate that the *pipeline* works --
that the physics solver, the losses, the training loop, the splits and the metrics are all
correct and that the model can recover parameters it is in principle able to recover. They
are **not** clinical validation and must never be quoted as accuracy on people. Real-data
adapters live in ``pinnbp.data.real``; ``docs/RESULTS.md`` keeps simulated and real results
in separate tables for exactly this reason.

The simulator exists because of a hardware constraint that is worth stating plainly: this
project is developed on a machine with about a gigabyte of free RAM and no GPU. The UCI
cuffless-BP set is several gigabytes of MATLAB v7.3 files. A pipeline that can only be
exercised by first loading that is a pipeline that cannot be tested, so the synthetic path
is the default and the real path is opt-in.

HOW THE SIGNAL IS BUILT
-----------------------
Working forwards from physiology rather than backwards from a plausible-looking waveform::

    subject parameters (R, C, SV, HR, Tsys, Zc, age)
        |                                                    2-element Windkessel,
        v                                                    exact ZOH discretisation
    aortic pressure P(t)  <--- driven by half-sine ejection flow Q(t)
        |
        |  peripheral transfer: transit delay set by Bramwell-Hill PWV(C),
        |  plus a reflected wave whose return time also scales with PWV
        v
    peripheral pressure
        |
        |  nonlinear volume-pressure relation (arterial compliance saturates)
        v
    arterial volume  ==  PPG
        |
        |  respiratory modulation, sensor noise, then optionally motion artifact
        v
    observed PPG window          labels: SBP, DBP read off the true P(t) in that window

The important property is that compliance C reaches the PPG through **two independent
routes** -- pulse amplitude via the volume-pressure curve, and pulse *timing* via PWV and
the reflected wave. That is what makes the physics constraints non-trivial: when motion
artifact destroys amplitude information, timing survives, and a model that has learned the
physical relationship between them can still recover C. A model that only learned amplitude
cannot. The robustness experiment is designed to detect exactly this difference.

The discretisation deserves a note. The 2-element Windkessel is a first-order linear system,
so under a zero-order hold its exact discrete solution is a one-pole IIR filter::

    P[n+1] = a P[n] + R(1-a) Q[n],    a = exp(-dt / (R*C))

That is not an approximation of the ODE, it is its closed-form solution for piecewise-
constant flow, which is why ``scipy.signal.lfilter`` is used instead of a stepped solver.
It is both faster and more accurate than RK4 here. The torch solver in
``pinnbp.physics.windkessel`` cannot use this trick because there R and C are per-sample
predictions that must stay in the autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

__all__ = [
    "SubjectParams",
    "CohortConfig",
    "analytic_bp",
    "sample_subject",
    "simulate_subject",
    "generate_cohort",
]


@dataclass
class SubjectParams:
    """One simulated person's cardiovascular and sensor parameters."""

    subject_id: int
    age: float
    R: float
    C: float
    SV: float
    HR: float
    Tsys: float
    Zc: float
    # Peripheral / optical transduction
    reflection_coeff: float
    p_half: float
    beta: float
    # Nuisance
    resp_rate: float
    resp_depth: float
    noise_sd: float
    hrv_sd: float

    def as_dict(self) -> dict[str, float]:
        return {
            "subject_id": self.subject_id,
            "age": self.age,
            "R": self.R,
            "C": self.C,
            "SV": self.SV,
            "HR": self.HR,
            "Tsys": self.Tsys,
            "Zc": self.Zc,
        }


@dataclass
class CohortConfig:
    """Generation settings for a whole simulated cohort."""

    n_subjects: int = 200
    duration_s: float = 60.0
    fs: float = 125.0
    window_s: float = 8.0
    stride_s: float = 4.0
    hypertensive_fraction: float = 0.35
    seed: int = 0
    artifact_severity: float = 0.0
    artifact_prob: float = 0.5
    extra: dict = field(default_factory=dict)


def analytic_bp(p: SubjectParams) -> tuple[float, float, float]:
    """Closed-form (SBP, DBP, MAP) for a subject, without integrating anything.

    DBP is the exact steady-state phase-zero pressure of the 2-element Windkessel (the same
    derivation as :func:`pinnbp.physics.windkessel.steady_state_p0`), MAP is CO*R plus the
    3-element term Zc*CO, and SBP follows from the MAP rule of thumb, SBP = 3*MAP - 2*DBP.

    Only SBP is approximate, and it is used solely as an inclusion criterion when drawing
    the cohort -- the labels themselves are always read off the fully simulated waveform.
    """
    tau = p.R * p.C
    T = 60.0 / p.HR
    q0 = np.pi * p.SV / (2.0 * p.Tsys)
    a, b = 1.0 / tau, np.pi / p.Tsys
    integral = q0 * b * (np.exp(p.Tsys / tau) + 1.0) / (a * a + b * b)
    decay = np.exp(-T / tau)
    dbp = float((integral / p.C) * decay / (1.0 - decay))

    co = p.SV * p.HR / 60.0
    map_ = float(co * p.R + p.Zc * co)
    sbp = float(3.0 * map_ - 2.0 * dbp)
    return sbp, dbp, map_


# Cohort inclusion criteria. A simulated cohort should look like a recruitable one, and an
# unconstrained draw does not: independent tails in compliance and resistance occasionally
# combine into a subject at 239/95 with a 125 mmHg pulse pressure, which is not a person a
# wearable study would enrol. Real studies apply exactly this kind of screening, so the
# simulator does too -- and it is applied to the *parameters* before simulation, never to
# the labels afterwards, which would be censoring the very cases the model finds hardest.
_BP_INCLUSION = {"sbp": (85.0, 200.0), "dbp": (45.0, 125.0), "pp": (25.0, 100.0)}


def _acceptable(p: SubjectParams) -> bool:
    sbp, dbp, _ = analytic_bp(p)
    lo, hi = _BP_INCLUSION["sbp"]
    if not (lo <= sbp <= hi):
        return False
    lo, hi = _BP_INCLUSION["dbp"]
    if not (lo <= dbp <= hi):
        return False
    lo, hi = _BP_INCLUSION["pp"]
    return lo <= (sbp - dbp) <= hi


def sample_subject(
    rng: np.random.Generator, subject_id: int, hypertensive: bool, max_tries: int = 50
) -> SubjectParams:
    """Draw a subject that satisfies the cohort inclusion criteria.

    Redraws rather than clipping. Clipping a parameter to a bound piles probability mass on
    that bound and creates a spike of subjects with identical compliance, which the model
    can then learn as a spurious mode.
    """
    for _ in range(max_tries):
        p = _sample_subject_once(rng, subject_id, hypertensive)
        if _acceptable(p):
            return p
    # Falling through means the criteria and the priors disagree; that is a bug worth
    # surfacing rather than quietly returning an out-of-range subject.
    raise RuntimeError(
        f"could not draw an acceptable subject in {max_tries} tries "
        f"(hypertensive={hypertensive}); check _BP_INCLUSION against the priors"
    )


def _sample_subject_once(
    rng: np.random.Generator, subject_id: int, hypertensive: bool
) -> SubjectParams:
    """Draw one subject's parameters, with the physiological correlations kept intact.

    Parameters are emphatically *not* drawn independently. Age drives arterial stiffening,
    which lowers compliance, which widens pulse pressure and raises systolic pressure --
    the well-established isolated-systolic-hypertension pattern of ageing. Hypertension
    additionally raises peripheral resistance, which raises the mean pressure.

    Drawing independently would produce a cohort in which SBP and DBP are nearly
    uncorrelated and every parameter is separately identifiable from any other. Real
    cohorts are not like that, and a model tuned on such data would look far better than it
    is -- most of the reported skill would come from the absence of the confounding that
    makes the real problem hard.

    Args:
        rng: seeded generator.
        subject_id: identifier, carried through to the grouped train/test split.
        hypertensive: whether to draw from the hypertensive part of the distribution.

    Returns:
        A fully specified :class:`SubjectParams`.
    """
    age = float(np.clip(rng.normal(48 if hypertensive else 38, 14), 18, 85))

    # Compliance falls roughly linearly with age; the spread is wide because vascular age
    # and chronological age are only loosely coupled.
    c_base = 2.30 - 0.016 * (age - 20.0)
    C = float(np.clip(rng.normal(c_base, 0.28), 0.60, 3.2))
    if hypertensive:
        C = float(np.clip(C * rng.uniform(0.78, 0.94), 0.60, 3.2))

    r_base = 1.08 + (0.30 if hypertensive else 0.0)
    R = float(np.clip(rng.normal(r_base, 0.16), 0.55, 2.0))

    SV = float(np.clip(rng.normal(72.0, 11.0), 38.0, 120.0))
    HR = float(np.clip(rng.normal(70.0, 10.0), 45.0, 110.0))

    # Weissler's regression: ejection time shortens as heart rate rises.
    Tsys = float(np.clip(0.37 - 0.0015 * HR + rng.normal(0, 0.012), 0.18, 0.42))

    Zc = float(np.clip(rng.normal(0.055, 0.014), 0.02, 0.11))

    # Stiffer arteries reflect more strongly and the reflection returns sooner, which is
    # what raises the augmentation index with age.
    reflection = float(np.clip(rng.normal(0.34, 0.07) * (2.0 / max(C, 0.5)) * 0.55, 0.08, 0.62))

    # Operating point and steepness of the volume-pressure curve. p_half tracks the
    # subject's own mean pressure so that everyone sits on a comparable part of the curve
    # rather than hypertensives all being pushed into saturation.
    map_est = (SV * HR / 60.0) * R
    p_half = float(map_est + rng.normal(0, 6.0))
    beta = float(np.clip(rng.normal(26.0, 5.0), 12.0, 45.0))

    return SubjectParams(
        subject_id=subject_id,
        age=age,
        R=R,
        C=C,
        SV=SV,
        HR=HR,
        Tsys=Tsys,
        Zc=Zc,
        reflection_coeff=reflection,
        p_half=p_half,
        beta=beta,
        resp_rate=float(rng.uniform(0.16, 0.33)),
        resp_depth=float(rng.uniform(0.04, 0.16)),
        noise_sd=float(rng.uniform(0.004, 0.030)),
        hrv_sd=float(rng.uniform(0.012, 0.055)),
    )


def _rr_series(p: SubjectParams, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Beat-to-beat RR intervals with respiratory sinus arrhythmia.

    RSA -- heart rate rising on inspiration and falling on expiration -- is modelled
    explicitly rather than as white jitter because it correlates the RR series with the
    respiratory modulation of the PPG baseline. Uncorrelated jitter would let a model use
    baseline wander to predict beat timing in a way it cannot on a real recording.
    """
    mean_rr = 60.0 / p.HR
    rr: list[float] = []
    t = 0.0
    while t < duration_s + 2 * mean_rr:
        rsa = p.hrv_sd * np.sin(2 * np.pi * p.resp_rate * t)
        interval = mean_rr * (1.0 + rsa) + rng.normal(0, p.hrv_sd * 0.4 * mean_rr)
        interval = float(np.clip(interval, 0.33, 1.6))
        rr.append(interval)
        t += interval
    return np.array(rr)


def _flow_signal(p: SubjectParams, rr: np.ndarray, fs: float, n: int) -> np.ndarray:
    """Build the aortic inflow Q(t) over the whole recording from the RR series."""
    q = np.zeros(n, dtype=np.float64)
    t0 = 0.0
    for interval in rr:
        start = int(round(t0 * fs))
        tsys = min(p.Tsys, 0.9 * interval)
        length = int(round(tsys * fs))
        if start >= n:
            break
        if length >= 2:
            k = np.arange(length)
            # Half-sine normalised so its integral is exactly one stroke volume.
            profile = np.sin(np.pi * k / length)
            profile *= p.SV / (np.trapezoid(profile) / fs)
            end = min(start + length, n)
            q[start:end] += profile[: end - start]
        t0 += interval
    return q


def _windkessel_lfilter(q: np.ndarray, R: float, C: float, fs: float, p_init: float) -> np.ndarray:
    """Exact zero-order-hold solution of the 2-element Windkessel as a one-pole IIR.

    See the module docstring: for piecewise-constant flow this is the closed-form solution,
    not a numerical approximation. ``zi`` carries the initial condition so the recording
    does not begin with a transient from zero pressure.
    """
    dt = 1.0 / fs
    a = float(np.exp(-dt / (R * C)))
    b = R * (1.0 - a)
    zi = np.array([a * p_init])
    out, _ = sps.lfilter([0.0, b], [1.0, -a], q, zi=zi)
    return out


def _peripheral_transfer(
    p_aortic: np.ndarray, p: SubjectParams, fs: float
) -> tuple[np.ndarray, float]:
    """Propagate aortic pressure to the measurement site.

    Two effects, both keyed to pulse wave velocity, which is itself a function of compliance
    via Bramwell-Hill. This is the timing route by which C becomes observable.

    Returns:
        ``(peripheral_pressure, ptt_seconds)``.
    """
    # Bramwell-Hill scaling, calibrated so a typical total compliance gives a typical
    # large-artery PWV. Kept identical to pinnbp.physics.windkessel.bramwell_hill_pwv so the
    # simulator and the model share one definition of the compliance-to-timing relation --
    # if these two ever disagreed, the physics constraint would be fitting a relationship
    # the data does not contain, and the whole timing route to compliance would be spurious.
    pwv = 8.0 * float(np.sqrt(1.6 / max(p.C, 1e-6)))
    path_len_m = 0.85  # heart to wrist, roughly
    ptt = path_len_m / pwv

    delay = int(round(ptt * fs))
    delayed = np.roll(p_aortic, delay)
    delayed[:delay] = p_aortic[0]

    # Reflected wave returning from the periphery: a second traverse of the path, so twice
    # the transit time, attenuated by the reflection coefficient.
    refl_delay = int(round(2.0 * ptt * fs))
    reflected = np.roll(p_aortic, refl_delay)
    reflected[:refl_delay] = p_aortic[0]

    combined = delayed + p.reflection_coeff * (reflected - reflected.mean())

    # Viscoelastic damping of the arterial wall smooths the sharpest features on the way
    # out to the periphery.
    sos = sps.butter(2, min(12.0, 0.45 * fs) / (fs / 2), btype="low", output="sos")
    return sps.sosfiltfilt(sos, combined), ptt


def _pressure_to_volume(p_periph: np.ndarray, p: SubjectParams) -> np.ndarray:
    """Nonlinear arterial volume-pressure relation.

    V(P) = 0.5 + arctan((P - p_half)/beta)/pi

    Arterial compliance is not constant: the vessel distends easily at low pressure and
    stiffens as it approaches its elastic limit, so the curve is sigmoidal. Modelling this
    matters for honesty about the method's limits -- it is the reason PPG-derived estimates
    are biased at high pressures, and the reason the tau measured from a PPG diastolic limb
    only approximates the true arterial R*C. A linear map would hide that whole failure
    mode and make the task easier than it is.
    """
    return 0.5 + np.arctan((p_periph - p.p_half) / p.beta) / np.pi


def simulate_subject(
    p: SubjectParams, duration_s: float, fs: float, rng: np.random.Generator
) -> dict:
    """Simulate one subject's recording.

    Returns:
        Dict with ``ppg``, ``abp``, ``t``, ``ptt``, and the subject parameters. ``abp`` is
        the true peripheral pressure and is the source of the SBP/DBP labels; it is kept in
        the output so that plots can show the model's target alongside its input, and so
        the label extraction is auditable rather than hidden.
    """
    n = int(round(duration_s * fs))
    rr = _rr_series(p, duration_s, rng)
    q = _flow_signal(p, rr, fs, n)

    p_init = (p.SV * p.HR / 60.0) * p.R  # start at the analytic MAP, so burn-in is short
    p_wk = _windkessel_lfilter(q, p.R, p.C, fs, p_init)
    p_aortic = p_wk + p.Zc * q  # 3-element form

    p_periph, ptt = _peripheral_transfer(p_aortic, p, fs)
    volume = _pressure_to_volume(p_periph, p)

    t = np.arange(n) / fs

    # Respiration modulates both the baseline (venous return) and the pulse amplitude
    # (intrathoracic pressure). Both are present in real wrist PPG.
    resp = np.sin(2 * np.pi * p.resp_rate * t)
    baseline = p.resp_depth * resp
    am = 1.0 + 0.35 * p.resp_depth * np.sin(2 * np.pi * p.resp_rate * t + 0.7)

    ppg = volume * am + baseline
    ppg = ppg + rng.normal(0, p.noise_sd, size=n)

    return {
        "ppg": ppg,
        "abp": p_periph,
        "t": t,
        "ptt": ptt,
        "params": p,
    }


def generate_cohort(cfg: CohortConfig) -> dict:
    """Generate a full cohort of windowed examples with labels.

    Labels are computed per window from the true pressure waveform inside *that window*,
    not once per subject. A 60-second recording contains real beat-to-beat variation in
    SBP, and averaging it away would both discard a genuine source of label variance and
    make the task artificially easy.

    Returns:
        Dict of arrays: ``ppg`` ``(N, L)``, ``sbp`` ``(N,)``, ``dbp`` ``(N,)``,
        ``subject`` ``(N,)``, ``params`` ``(N, 5)`` in order (R, C, SV, HR, Tsys),
        ``sqi`` ``(N,)``, plus ``feature_names`` and the config used.
    """
    from .artifacts import apply_artifacts
    from ..dsp.preprocess import bandpass, normalize_window, sliding_windows
    from ..dsp.sqi import signal_quality

    rng = np.random.default_rng(cfg.seed)
    # Artifacts draw from their own independent stream. This is not tidiness -- it is what
    # makes the robustness sweep valid. Sharing one generator would mean that changing the
    # severity changes how many numbers are drawn per window, which shifts every subsequent
    # subject's physiology, so each severity level would be a different cohort and the
    # resulting degradation curve would confound corruption with cohort change.
    art_rng = np.random.default_rng(cfg.seed + 99991)

    ppg_out: list[np.ndarray] = []
    sbp_out: list[float] = []
    dbp_out: list[float] = []
    subj_out: list[int] = []
    par_out: list[list[float]] = []
    sqi_out: list[float] = []

    n_hyper = int(round(cfg.n_subjects * cfg.hypertensive_fraction))
    flags = np.array([True] * n_hyper + [False] * (cfg.n_subjects - n_hyper))
    rng.shuffle(flags)

    for sid in range(cfg.n_subjects):
        p = sample_subject(rng, sid, bool(flags[sid]))
        rec = simulate_subject(p, cfg.duration_s, cfg.fs, rng)

        ppg_windows = sliding_windows(rec["ppg"], cfg.fs, cfg.window_s, cfg.stride_s)
        abp_windows = sliding_windows(rec["abp"], cfg.fs, cfg.window_s, cfg.stride_s)

        for w_ppg, w_abp in zip(ppg_windows, abp_windows, strict=True):
            sig = w_ppg
            # Draw the coin from the artifact stream unconditionally, so the number of
            # draws does not depend on the severity setting.
            corrupt = art_rng.random() < cfg.artifact_prob
            kinds = cfg.extra.get("artifact_kinds")
            if cfg.artifact_severity > 0 and corrupt:
                sig = apply_artifacts(
                    sig, cfg.fs, cfg.artifact_severity, art_rng,
                    kinds=tuple(kinds) if kinds else None,
                )

            sig = bandpass(sig, cfg.fs)
            sig = normalize_window(sig, method="robust")

            q = signal_quality(sig, cfg.fs)

            ppg_out.append(sig)
            sbp_out.append(float(w_abp.max()))
            dbp_out.append(float(w_abp.min()))
            subj_out.append(sid)
            par_out.append([p.R, p.C, p.SV, p.HR, p.Tsys])
            sqi_out.append(q.overall)

    return {
        "ppg": np.stack(ppg_out).astype(np.float32),
        "sbp": np.array(sbp_out, dtype=np.float32),
        "dbp": np.array(dbp_out, dtype=np.float32),
        "subject": np.array(subj_out, dtype=np.int64),
        "params": np.array(par_out, dtype=np.float32),
        "sqi": np.array(sqi_out, dtype=np.float32),
        "param_names": ["R", "C", "SV", "HR", "Tsys"],
        "fs": cfg.fs,
        "config": cfg,
    }
