"""Assemble a ready-to-train dataset from a config, with caching.

One entry point, :func:`build_dataset`, so that every model, every ablation and every sweep
gets its data through the same code path. The alternative -- each script building its own
loader -- is how two models end up quietly trained on differently preprocessed inputs and
compared as though they were not.

Caching is keyed by a hash of the fields that actually affect the arrays. Window
construction (Savitzky-Golay derivatives, per-beat exponential fits, the classical feature
matrix) takes roughly a minute for the default cohort and is identical across every model
and every epoch, so recomputing it per run would dominate short experiments. The key
deliberately excludes fields like ``split`` and ``ratios``: those change which indices go
where, not what the windows contain, and including them would evict a perfectly good cache
every time a split seed moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import Config
from .datasets import SplitIndices, build_windows, load_prepared, save_prepared, subject_split

__all__ = ["build_dataset", "cache_key"]


def cache_key(cfg: Config) -> str:
    """Stable hash of the config fields that change the prepared arrays."""
    d = cfg.data
    payload = {
        "dataset": d.dataset,
        "root": d.root,
        "n_subjects": d.n_subjects,
        "duration_s": d.duration_s,
        "fs": d.fs,
        "window_s": d.window_s,
        "stride_s": d.stride_s,
        "hypertensive_fraction": d.hypertensive_fraction,
        "train_artifact_severity": d.train_artifact_severity,
        "train_artifact_prob": d.train_artifact_prob,
        "seed": d.seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_raw(cfg: Config) -> dict:
    """Fetch raw windows from whichever source the config names."""
    d = cfg.data

    if d.dataset == "synthetic":
        from .synthetic import CohortConfig, generate_cohort

        cohort = generate_cohort(
            CohortConfig(
                n_subjects=d.n_subjects,
                duration_s=d.duration_s,
                fs=d.fs,
                window_s=d.window_s,
                stride_s=d.stride_s,
                hypertensive_fraction=d.hypertensive_fraction,
                seed=d.seed,
                artifact_severity=d.train_artifact_severity,
                artifact_prob=d.train_artifact_prob,
            )
        )
        return {
            "ppg": cohort["ppg"],
            "sbp": cohort["sbp"],
            "dbp": cohort["dbp"],
            "subject": cohort["subject"],
            "sqi": cohort["sqi"],
            "fs": d.fs,
            "true_params": cohort["params"],
            "param_names": cohort["param_names"],
        }

    if d.root is None:
        raise ValueError(
            f"dataset={d.dataset!r} needs data.root pointing at the downloaded files. "
            "See docs/DATA.md."
        )

    if d.dataset == "ppgbp":
        from .real import load_ppgbp

        return load_ppgbp(d.root, fs_out=d.fs)

    if d.dataset == "uci":
        from .real import load_uci_cuffless

        return load_uci_cuffless(
            d.root, fs_out=d.fs, window_s=d.window_s, stride_s=d.stride_s
        )

    raise ValueError(f"unknown dataset {d.dataset!r} (expected synthetic | ppgbp | uci)")


def build_dataset(
    cfg: Config, use_cache: bool = True, verbose: bool = True
) -> tuple[dict, SplitIndices]:
    """Build (or load) the prepared arrays and the split for a config.

    Args:
        cfg: experiment configuration.
        use_cache: read and write the ``.npz`` cache under ``cfg.data.cache_dir``.
        verbose: print what happened, including whether the cache was hit. Silence here
            would make it impossible to tell a fresh build from a stale cache being reused
            after a generator change.

    Returns:
        ``(prepared, splits)``.
    """
    key = cache_key(cfg)
    cache_path = Path(cfg.data.cache_dir) / f"{cfg.data.dataset}_{key}.npz"

    if use_cache and cache_path.exists():
        prepared = load_prepared(cache_path)
        if verbose:
            print(f"[data] cache hit  {cache_path}  ({len(prepared['sbp'])} windows)")
    else:
        if verbose:
            print(f"[data] building {cfg.data.dataset} (no cache at {cache_path})")
        raw = _load_raw(cfg)
        prepared = build_windows(
            ppg=raw["ppg"],
            sbp=raw["sbp"],
            dbp=raw["dbp"],
            subject=raw["subject"],
            fs=raw["fs"],
            compute_features=True,
            sqi=raw.get("sqi"),
        )
        if "true_params" in raw:
            # Only the simulator knows these. They let the evaluation ask whether the
            # network recovered the actual physical parameters or merely found some set of
            # values that happened to reproduce the right pressures -- a distinction that
            # cannot be checked at all on real data.
            prepared["true_params"] = raw["true_params"]
        if use_cache:
            save_prepared(prepared, cache_path)
            if verbose:
                print(f"[data] cached -> {cache_path}")

    splits = subject_split(
        prepared["subject"],
        ratios=tuple(cfg.data.ratios),
        seed=cfg.data.seed,
        mode=cfg.data.split,
    )

    if verbose:
        print(f"[data] {splits.summary()}")
        if cfg.data.split == "random":
            print(
                "[data] WARNING: random split leaks subjects across partitions. "
                "Its numbers quantify that leakage; they are not a result."
            )
        sbp, dbp = prepared["sbp"], prepared["dbp"]
        print(
            f"[data] SBP {sbp.mean():.1f} +/- {sbp.std():.1f} "
            f"[{sbp.min():.0f}-{sbp.max():.0f}] | "
            f"DBP {dbp.mean():.1f} +/- {dbp.std():.1f} "
            f"[{dbp.min():.0f}-{dbp.max():.0f}] mmHg"
        )
        good = float((prepared["sqi"] >= 0.5).mean() * 100)
        tau_ok = float(prepared["tau_valid"].mean() * 100)
        print(f"[data] SQI >= 0.5: {good:.1f}% of windows | usable tau fit: {tau_ok:.1f}%")

    return prepared, splits


def training_mean(prepared: dict, splits: SplitIndices) -> np.ndarray:
    """Mean SBP/DBP over the training split only.

    Computed on train alone because it is used as a baseline predictor evaluated on test;
    taking the mean over everything would leak the test distribution into the baseline and
    make it look better than a real deployment could.
    """
    idx = splits.train
    return np.array([prepared["sbp"][idx].mean(), prepared["dbp"][idx].mean()], dtype=np.float64)
