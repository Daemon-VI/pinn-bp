"""Adapters for the public cuffless-BP datasets.

Both adapters produce the same ``(ppg, sbp, dbp, subject, fs)`` shape that
:func:`pinnbp.data.datasets.build_windows` consumes, so a real run is the same command as a
synthetic one with a different ``--dataset``. Nothing downstream knows the difference.

Neither dataset is downloaded automatically. They are large, they require accepting terms,
and silently pulling several gigabytes onto a machine with 52 GB free and a metered
connection is not a decision this code should make on its own. ``docs/DATA.md`` has the
retrieval instructions; each loader raises with the exact expected layout when files are
missing.

An honest note about what these datasets can and cannot support, which belongs next to the
code that loads them rather than buried in a report:

**PPG-BP** (Liang et al., *Scientific Data* 2018) gives 219 subjects with cuff readings and
three 2.1-second PPG segments each. Its virtues are that it is small enough to work with
here, and that it is genuinely per-subject, so a subject-disjoint split is meaningful. Its
limitation is severe: 2.1 seconds is two or three beats, and one seated cuff measurement per
subject means the model is being asked to predict a *between-subject* difference, not a
within-subject change. Good numbers on it do not demonstrate that a device would track your
pressure as it varies.

**UCI Cuffless BP** (MIMIC-II derived) gives continuous simultaneous PPG and arterial line
pressure, so labels vary within a record and beat-to-beat tracking can actually be tested.
Two serious caveats. First, subject identity was not preserved when the set was assembled,
so records cannot be reliably grouped by person -- the "subject-disjoint" split degrades to
record-disjoint, and some leakage may remain. This is a known and widely ignored problem
with this dataset; the loader is explicit about it rather than quiet. Second, it is ICU
data: the pressure distribution is not that of healthy people wearing a smartwatch, which
is the deployment this project describes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..dsp.preprocess import bandpass, normalize_window, resample_signal, sliding_windows

__all__ = ["load_ppgbp", "load_uci_cuffless", "DATASET_SOURCES"]

DATASET_SOURCES = {
    "ppgbp": {
        "name": "PPG-BP Database (Liang et al. 2018)",
        "url": "https://doi.org/10.6084/m9.figshare.5459299",
        "size": "~30 MB",
        "layout": "<root>/Data File/0_subject/<id>_<n>.txt  +  <root>/Data File/PPG-BP dataset.xlsx",
    },
    "uci": {
        "name": "UCI Cuff-Less Blood Pressure Estimation (MIMIC-II derived)",
        "url": "https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation",
        "size": "~2.5 GB",
        "layout": "<root>/Part_1.mat ... Part_4.mat  (MATLAB v7.3 / HDF5)",
    },
}


def _missing(root: Path, key: str) -> FileNotFoundError:
    src = DATASET_SOURCES[key]
    return FileNotFoundError(
        f"{src['name']} not found under {root}.\n"
        f"  Download: {src['url']}  ({src['size']})\n"
        f"  Expected layout: {src['layout']}\n"
        f"  See docs/DATA.md. Use --dataset synthetic to run without it."
    )


def load_ppgbp(
    root: str | Path,
    fs_out: float = 125.0,
    min_sbp: float = 70.0,
    max_sbp: float = 220.0,
) -> dict:
    """Load the PPG-BP database.

    Each subject contributes up to three 2.1 s segments recorded at 1 kHz, which are
    resampled to ``fs_out`` and kept whole -- they are far too short to cut into overlapping
    windows, and doing so would manufacture correlated samples out of nothing.

    Segments are kept as separate examples rather than averaged, because the three
    recordings of one subject differ by more than measurement noise and averaging them would
    understate the real within-subject variability.

    Args:
        root: dataset root containing ``Data File/``.
        fs_out: target sampling rate.
        min_sbp, max_sbp: physiological sanity bounds; records outside are dropped and
            counted in the returned ``n_rejected``.

    Returns:
        Dict with ``ppg`` ``(N, L)``, ``sbp``, ``dbp``, ``subject``, ``fs``, ``age``,
        ``n_rejected``.

    Raises:
        FileNotFoundError: with retrieval instructions if the layout is not present.
    """
    root = Path(root)
    data_dir = root / "Data File" / "0_subject"
    if not data_dir.is_dir():
        data_dir = root / "0_subject"
    if not data_dir.is_dir():
        raise _missing(root, "ppgbp")

    xlsx = next(root.rglob("*PPG-BP*.xlsx"), None)
    if xlsx is None:
        raise _missing(root, "ppgbp")

    import pandas as pd

    meta = pd.read_excel(xlsx, header=1)
    meta.columns = [str(c).strip() for c in meta.columns]

    def _col(*candidates: str) -> str:
        for c in meta.columns:
            for cand in candidates:
                if cand.lower() in c.lower():
                    return c
        raise KeyError(f"none of {candidates} present in {list(meta.columns)}")

    c_id = _col("subject_ID", "subject")
    c_sbp = _col("Systolic")
    c_dbp = _col("Diastolic")
    c_age = _col("Age")

    ppg: list[np.ndarray] = []
    sbp: list[float] = []
    dbp: list[float] = []
    subj: list[int] = []
    ages: list[float] = []
    rejected = 0

    lengths: list[int] = []
    raw: list[tuple[np.ndarray, float, float, int, float]] = []

    for _, row in meta.iterrows():
        sid_raw = row[c_id]
        try:
            sid = int(str(sid_raw).strip().split("_")[0])
        except (ValueError, AttributeError):
            continue

        s, d = float(row[c_sbp]), float(row[c_dbp])
        if not (min_sbp <= s <= max_sbp) or not (30.0 <= d < s):
            rejected += 1
            continue

        for seg_file in sorted(data_dir.glob(f"{sid}_*.txt")):
            vals = np.loadtxt(seg_file, delimiter="\t" if "\t" in seg_file.read_text()[:200] else ",")
            vals = np.atleast_1d(vals).ravel().astype(np.float64)
            if len(vals) < 500:
                rejected += 1
                continue
            sig = resample_signal(vals, 1000.0, fs_out)
            raw.append((sig, s, d, sid, float(row[c_age])))
            lengths.append(len(sig))

    if not raw:
        raise _missing(root, "ppgbp")

    # Trim every segment to the shortest so the batch is rectangular. Trimming rather than
    # zero-padding: a padded tail is a flat line that the SQI and the beat detector would
    # both read as contact loss.
    L = int(np.median(lengths))
    for sig, s, d, sid, age in raw:
        if len(sig) < L:
            continue
        x = sig[:L]
        x = bandpass(x, fs_out)
        ppg.append(normalize_window(x, "robust"))
        sbp.append(s)
        dbp.append(d)
        subj.append(sid)
        ages.append(age)

    return {
        "ppg": np.stack(ppg).astype(np.float32),
        "sbp": np.array(sbp, dtype=np.float32),
        "dbp": np.array(dbp, dtype=np.float32),
        "subject": np.array(subj, dtype=np.int64),
        "age": np.array(ages, dtype=np.float32),
        "fs": fs_out,
        "n_rejected": rejected,
        "source": "ppgbp",
    }


def load_uci_cuffless(
    root: str | Path,
    fs_in: float = 125.0,
    fs_out: float = 125.0,
    window_s: float = 8.0,
    stride_s: float = 4.0,
    max_records: int | None = 200,
    max_windows_per_record: int = 30,
    min_sbp: float = 70.0,
    max_sbp: float = 220.0,
) -> dict:
    """Load the UCI cuffless BP set (MIMIC-II derived).

    Reads MATLAB v7.3 files, which are HDF5 underneath and need ``h5py``. Records are
    streamed and capped rather than loaded wholesale -- the full set will not fit in this
    machine's memory, and ``max_records``/``max_windows_per_record`` exist to make a partial
    run possible rather than an impossible one. Any cap is reported in the return value so
    that a result computed on a subset is never mistaken for one computed on the whole set.

    Labels come from the simultaneous arterial line: SBP and DBP are the max and min of the
    ABP channel *within each window*, which is what makes this dataset able to test
    within-subject tracking.

    Args:
        root: directory holding ``Part_*.mat``.
        fs_in: native rate of the dataset (125 Hz).
        fs_out: target rate.
        window_s, stride_s: windowing.
        max_records: stop after this many records; ``None`` for all.
        max_windows_per_record: cap per record, keeping the cohort balanced across records
            instead of dominated by the longest ones.
        min_sbp, max_sbp: sanity bounds on the derived labels.

    Returns:
        The same dict shape as :func:`load_ppgbp`, plus ``capped`` and ``n_records``.

    Raises:
        FileNotFoundError: if no ``Part_*.mat`` is present.
        ImportError: if ``h5py`` is not installed.
    """
    root = Path(root)
    parts = sorted(root.glob("Part_*.mat"))
    if not parts:
        raise _missing(root, "uci")

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "The UCI set is MATLAB v7.3 (HDF5). Install h5py to read it:\n"
            "    uv add h5py\n"
            "h5py is not a base dependency because the synthetic path does not need it."
        ) from exc

    ppg: list[np.ndarray] = []
    sbp: list[float] = []
    dbp: list[float] = []
    subj: list[int] = []
    rejected = 0
    record_id = 0

    for part in parts:
        with h5py.File(part, "r") as f:
            key = next(iter(f.keys()))
            refs = f[key]
            n_rec = refs.shape[0]

            for r in range(n_rec):
                if max_records is not None and record_id >= max_records:
                    break
                arr = np.array(f[refs[r][0]])
                # Stored as (3, N) or (N, 3) depending on the file; PPG is row 0, ABP row 1.
                if arr.shape[0] != 3 and arr.shape[-1] == 3:
                    arr = arr.T
                if arr.shape[0] < 2:
                    continue

                sig_ppg = arr[0].astype(np.float64)
                sig_abp = arr[1].astype(np.float64)

                if fs_in != fs_out:
                    sig_ppg = resample_signal(sig_ppg, fs_in, fs_out)
                    sig_abp = resample_signal(sig_abp, fs_in, fs_out)

                w_ppg = sliding_windows(sig_ppg, fs_out, window_s, stride_s)
                w_abp = sliding_windows(sig_abp, fs_out, window_s, stride_s)
                if len(w_ppg) == 0:
                    continue

                keep = 0
                for a, b in zip(w_ppg, w_abp, strict=True):
                    if keep >= max_windows_per_record:
                        break
                    s, d = float(b.max()), float(b.min())
                    if not (min_sbp <= s <= max_sbp) or not (30.0 <= d < s - 10.0):
                        rejected += 1
                        continue
                    if not np.all(np.isfinite(a)) or a.std() < 1e-6:
                        rejected += 1
                        continue
                    x = bandpass(a, fs_out)
                    ppg.append(normalize_window(x, "robust"))
                    sbp.append(s)
                    dbp.append(d)
                    # Record index stands in for subject identity. See the module docstring
                    # -- this is a genuine weakness of the dataset, not of this loader.
                    subj.append(record_id)
                    keep += 1

                record_id += 1

        if max_records is not None and record_id >= max_records:
            break

    if not ppg:
        raise RuntimeError(f"no usable windows in {root}; {rejected} rejected by sanity bounds")

    return {
        "ppg": np.stack(ppg).astype(np.float32),
        "sbp": np.array(sbp, dtype=np.float32),
        "dbp": np.array(dbp, dtype=np.float32),
        "subject": np.array(subj, dtype=np.int64),
        "fs": fs_out,
        "n_rejected": rejected,
        "n_records": record_id,
        "capped": max_records is not None and record_id >= max_records,
        "source": "uci",
        "subject_ids_are_records": True,
    }
