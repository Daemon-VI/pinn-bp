"""Datasets, channel construction, and subject-disjoint splitting.

THE SPLIT IS THE MOST IMPORTANT THING IN THIS FILE
--------------------------------------------------
Published cuffless-BP results are frequently inflated by one specific mistake: splitting
overlapping windows at random. Windows from the same person, often overlapping in time,
then land in both train and test. The model memorises "this person reads 132/85" and the
reported MAE collapses to 2-3 mmHg, which looks like a breakthrough and generalises to
nobody. The gap between random-split and subject-split error on identical data and an
identical model is routinely a factor of three or more.

Every split here is therefore **grouped by subject**, and :func:`subject_split` asserts
disjointness rather than trusting the caller. ``--split random`` exists solely so the
inflation can be measured and reported in ``docs/RESULTS.md``; it is never the default, and
the number it produces is labelled as a leakage demonstration, not as a result.

CHANNELS
--------
Three channels per window: the PPG, its first derivative (VPG) and its second (APG). The
derivatives are computed once here rather than learned, because a first layer that has to
discover differentiation spends capacity on something a 5-tap kernel already does exactly,
and the APG's a/b wave ratio is a directly interpretable stiffness marker.

Each channel is normalised independently. The derivatives are one and two orders of
magnitude smaller than the signal, so sharing a scale would leave the APG numerically
invisible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..dsp.features import diastolic_tau, heart_rate, window_features
from ..dsp.preprocess import derivatives, normalize_window, segment_beats

__all__ = [
    "BPWindows",
    "SplitIndices",
    "subject_split",
    "build_windows",
    "collate",
]


@dataclass
class SplitIndices:
    """Index arrays for the three partitions, plus the subject ids in each."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_subjects: np.ndarray
    val_subjects: np.ndarray
    test_subjects: np.ndarray

    def summary(self) -> str:
        return (
            f"train {len(self.train):5d} windows / {len(self.train_subjects):3d} subjects | "
            f"val {len(self.val):5d} / {len(self.val_subjects):3d} | "
            f"test {len(self.test):5d} / {len(self.test_subjects):3d}"
        )


def subject_split(
    subjects: np.ndarray,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    mode: str = "subject",
) -> SplitIndices:
    """Partition windows into train/val/test.

    Args:
        subjects: ``(N,)`` subject id per window.
        ratios: train/val/test fractions. Applied to *subjects*, not windows, so the
            resulting window counts will not match the ratios exactly when subjects
            contribute different numbers of windows. That is correct and intended.
        seed: RNG seed.
        mode: ``"subject"`` for a disjoint split (always use this), or ``"random"`` to
            deliberately leak subjects across partitions in order to quantify the
            inflation. See the module docstring.

    Returns:
        :class:`SplitIndices`.

    Raises:
        AssertionError: if a subject-mode split turns out not to be disjoint. This is a
            hard assertion on purpose -- a silent leak invalidates every number the project
            produces, so it is worth crashing over.
    """
    rng = np.random.default_rng(seed)
    n = len(subjects)

    if mode == "random":
        idx = rng.permutation(n)
        n_tr = int(ratios[0] * n)
        n_va = int(ratios[1] * n)
        tr, va, te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]
        return SplitIndices(
            tr, va, te, np.unique(subjects[tr]), np.unique(subjects[va]), np.unique(subjects[te])
        )

    if mode != "subject":
        raise ValueError(f"unknown split mode {mode!r}")

    uniq = np.unique(subjects)
    rng.shuffle(uniq)
    n_tr = int(round(ratios[0] * len(uniq)))
    n_va = int(round(ratios[1] * len(uniq)))
    s_tr = uniq[:n_tr]
    s_va = uniq[n_tr : n_tr + n_va]
    s_te = uniq[n_tr + n_va :]

    tr = np.flatnonzero(np.isin(subjects, s_tr))
    va = np.flatnonzero(np.isin(subjects, s_va))
    te = np.flatnonzero(np.isin(subjects, s_te))

    assert set(s_tr).isdisjoint(s_va), "subject leak between train and val"
    assert set(s_tr).isdisjoint(s_te), "subject leak between train and test"
    assert set(s_va).isdisjoint(s_te), "subject leak between val and test"

    return SplitIndices(tr, va, te, s_tr, s_va, s_te)


def _window_tau(x: np.ndarray, fs: float, min_r2: float = 0.50) -> tuple[float, bool]:
    """Median diastolic tau across the beats in a window, with a validity flag.

    Only beats whose decay fit reaches ``min_r2`` are counted, and the window is marked
    invalid if fewer than two survive. Refusing to supply a number is the safe failure: a
    tau fitted to a motion artifact does not merely add noise, it would teach the network a
    wrong compliance.

    The 0.5 threshold is calibrated against the measured yield rather than chosen for
    strictness. A 2 s decay observed through a 0.4 s diastole and a 0.5 Hz high-pass is an
    intrinsically poorly conditioned fit, so demanding r2 >= 0.9 rejected essentially every
    window and left the tau physics term dead -- present in the loss, contributing nothing.
    At 0.5 roughly 40% of clean windows yield a usable value, which is enough for the
    batch-standardised constraint in ``pinnbp.physics.losses.tau_consistency``.
    """
    taus: list[float] = []
    for s, e in segment_beats(x, fs):
        seg = x[s:e]
        if len(seg) < 12:
            continue
        tau, r2 = diastolic_tau(seg, fs)
        if np.isfinite(tau) and r2 >= min_r2 and 0.2 < tau < 5.0:
            taus.append(tau)

    if len(taus) < 2:
        return 1.5, False  # placeholder value, masked out by the flag
    return float(np.median(taus)), True


def build_windows(
    ppg: np.ndarray,
    sbp: np.ndarray,
    dbp: np.ndarray,
    subject: np.ndarray,
    fs: float,
    compute_features: bool = True,
    sqi: np.ndarray | None = None,
) -> dict:
    """Turn raw windows into the arrays the models consume.

    Computes derivative channels, the measured tau and its validity mask, heart rate, and
    optionally the classical feature matrix used by the ridge baseline. Done once and
    cached, because the Savitzky-Golay derivatives and per-beat fits are the slowest part
    of the pipeline and are identical for every model and every epoch.

    Args:
        ppg: ``(N, L)`` filtered, normalised windows.
        sbp, dbp: ``(N,)`` labels in mmHg.
        subject: ``(N,)`` subject ids.
        fs: sampling rate.
        compute_features: also build the ``(N, F)`` classical feature matrix.
        sqi: optional precomputed quality scores; computed here if omitted.

    Returns:
        Dict of numpy arrays ready to be wrapped by :class:`BPWindows`.
    """
    from ..dsp.sqi import signal_quality

    N, L = ppg.shape
    chans = np.zeros((N, 3, L), dtype=np.float32)
    tau = np.zeros(N, dtype=np.float32)
    tau_ok = np.zeros(N, dtype=bool)
    hr = np.zeros(N, dtype=np.float32)
    feats = np.zeros((N, 20), dtype=np.float32) if compute_features else None
    quality = np.zeros(N, dtype=np.float32) if sqi is None else sqi.astype(np.float32)

    for i in range(N):
        x = ppg[i].astype(np.float64)
        vpg, apg = derivatives(x, fs)

        chans[i, 0] = normalize_window(x, "robust")
        chans[i, 1] = normalize_window(vpg, "robust")
        chans[i, 2] = normalize_window(apg, "robust")

        t, ok = _window_tau(x, fs)
        tau[i] = t
        tau_ok[i] = ok

        h, _ = heart_rate(x, fs)
        # A window with no detectable pulse still needs a number for the tensor. 75 bpm is
        # the cohort centre; the SQI is what tells downstream code not to trust it.
        hr[i] = h if np.isfinite(h) else 75.0

        if sqi is None:
            quality[i] = signal_quality(x, fs).overall

        if compute_features:
            f = window_features(x, fs)
            feats[i] = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    out = {
        "channels": chans,
        "sbp": sbp.astype(np.float32),
        "dbp": dbp.astype(np.float32),
        "subject": subject.astype(np.int64),
        "tau": tau,
        "tau_valid": tau_ok,
        "hr": hr,
        "sqi": quality,
        "fs": float(fs),
    }
    if compute_features:
        out["features"] = feats
    return out


class BPWindows(Dataset):
    """Torch dataset over prepared windows.

    Holds a view into shared arrays and an index list, so the three splits do not copy the
    data three times. On a machine with a gigabyte free that is not a micro-optimisation.
    """

    def __init__(self, prepared: dict, indices: np.ndarray):
        self.d = prepared
        self.idx = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        j = int(self.idx[i])
        return {
            "x": torch.from_numpy(self.d["channels"][j]),
            "sbp": torch.tensor(self.d["sbp"][j]),
            "dbp": torch.tensor(self.d["dbp"][j]),
            "tau": torch.tensor(self.d["tau"][j]),
            "tau_valid": torch.tensor(self.d["tau_valid"][j]),
            "hr": torch.tensor(self.d["hr"][j]),
            "sqi": torch.tensor(self.d["sqi"][j]),
            "subject": torch.tensor(self.d["subject"][j]),
        }

    @property
    def features(self) -> np.ndarray:
        """Classical feature matrix for this split, for the ridge baseline."""
        if "features" not in self.d:
            raise KeyError("features were not computed for this dataset")
        return self.d["features"][self.idx]

    @property
    def targets(self) -> np.ndarray:
        """``(n, 2)`` SBP/DBP labels for this split."""
        return np.stack([self.d["sbp"][self.idx], self.d["dbp"][self.idx]], axis=1)

    @property
    def subjects(self) -> np.ndarray:
        return self.d["subject"][self.idx]


def collate(batch: list[dict]) -> dict:
    """Default collation, kept explicit so the batch keys are documented in one place."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def save_prepared(prepared: dict, path: str | Path) -> None:
    """Cache prepared arrays to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in prepared.items() if isinstance(v, np.ndarray)},
                        fs=np.array(prepared["fs"]))


def load_prepared(path: str | Path) -> dict:
    """Load arrays cached by :func:`save_prepared`."""
    with np.load(Path(path), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["fs"] = float(out["fs"])
    return out
