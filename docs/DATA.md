# Datasets

The pipeline runs on three sources through one interface. Nothing is downloaded
automatically: the real sets are large, require accepting terms, and pulling several
gigabytes onto a metered connection is not a decision this code should make on its own.

## 1. Synthetic (default, always available)

A physiological simulator, not a noise generator. Subject parameters (R, C, SV, HR, Tsys, Zc)
are drawn with their real correlations intact — age stiffens arteries, which lowers compliance,
which widens pulse pressure — then pushed through a Windkessel model, a peripheral transfer
with a transit delay and a reflected wave, and a nonlinear volume–pressure curve.

```bash
uv run pinnbp experiment --config configs/experiment.yaml
```

**Its numbers are pipeline validation, never clinical accuracy.** Two limitations to keep in
view when reading `docs/RESULTS.md`:

- SBP spread is wider than a real cohort (SD ≈ 29 vs ≈ 20 mmHg), which flatters R².
- SBP and DBP correlate at ≈ 0.86, higher than the ≈ 0.75 typical of real cohorts, so
  predicting the mean pressure well gets you further here than it should.

## 2. PPG-BP (Liang et al., *Scientific Data* 2018)

219 subjects, three 2.1 s PPG segments each at 1 kHz, with seated cuff readings.

- Download: <https://doi.org/10.6084/m9.figshare.5459299> (~30 MB)
- Expected layout: `<root>/Data File/0_subject/<id>_<n>.txt` and `<root>/Data File/PPG-BP dataset.xlsx`

```bash
uv run pinnbp train --config configs/experiment.yaml --model pinn \
  # then set data.dataset: ppgbp and data.root: path/to/PPG-BP in the config
```

Small enough to work with on this machine, and genuinely per-subject, so a subject-disjoint
split means something. But 2.1 s is two or three beats, and one cuff reading per subject means
the model is asked to predict a *between-subject* difference, not a within-subject change.
Good numbers here do not show that a device would track your pressure as it varies.

## 3. UCI Cuff-Less BP (MIMIC-II derived)

Continuous simultaneous PPG and arterial-line pressure, so labels vary within a record and
beat-to-beat tracking can actually be tested.

- Download: <https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation> (~2.5 GB)
- Expected layout: `<root>/Part_1.mat` … `Part_4.mat` (MATLAB v7.3 / HDF5)
- Needs `h5py`: `uv add h5py` — not a base dependency because the synthetic path does not need it.

Two caveats the loader is explicit about rather than quiet:

1. **Subject identity was not preserved** when the set was assembled, so records cannot be
   reliably grouped by person. The subject-disjoint split degrades to record-disjoint and some
   leakage may remain. This is a known, widely ignored problem with this dataset.
2. **It is ICU data.** The pressure distribution is not that of healthy people wearing a
   smartwatch, which is the deployment this project describes.

Records are streamed and capped (`max_records`, `max_windows_per_record`) because the full set
will not fit in this machine's memory. Any cap is reported in the return value so a result
computed on a subset is never mistaken for one computed on the whole.
