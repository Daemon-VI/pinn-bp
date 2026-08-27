# PINN-BP — Physics-Informed Neural Networks for Cuffless Blood Pressure Estimation from PPG

Estimating systolic and diastolic blood pressure from a wearable photoplethysmogram, using a
network that predicts a **pressure waveform** constrained by the Windkessel model of arterial
haemodynamics, rather than regressing two numbers directly.

---

## The idea in one picture

```
PPG window (PPG, VPG, APG)
      |
      v
  CNN encoder ──> latent z ──> parameter head ──> R, C, SV, HR, Tsys, Zc
      |                                            (bounded, real units)
      v
  P(z, t)  a continuous, differentiable pressure field
      |
      +--> SBP = max P(t),  DBP = min P(t)      over one cardiac cycle
      |
      +--> physics losses:  C dP/dt + P/R = Q(t)     (Windkessel collocation)
                            P(0) = P(T)              (cycle closure)
                            rank(R*C) ~ rank(tau)    (measured decay constant)
                            physiological hinges
```

The model never outputs SBP and DBP as free parameters. It outputs a pressure wave, and the
readings are taken off it the way a clinician reads an arterial line. That is what gives the
physics somewhere to act — and it is what makes the prediction inspectable: every estimate
comes with a waveform and a set of cardiovascular parameters behind it.

## Why physics should help

Compliance reaches the PPG by **two** routes: pulse *amplitude*, through the nonlinear
volume–pressure curve, and pulse *timing*, through pulse wave velocity and the reflected wave.
Motion artifact destroys amplitude first. A model that only learned amplitude collapses; a
model that has learned the physical relationship between the two can fall back on timing.

The robustness experiment is built to detect exactly that difference — see
[`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start

```bash
uv sync --extra dev            # torch CPU, scipy, sklearn — no CUDA needed
uv run pinnbp info             # environment + dataset availability
uv run pytest -q               # 73 tests

uv run pinnbp experiment --config configs/quick.yaml --tag smoke   # ~3 min, meaningless numbers
uv run pinnbp experiment --config configs/experiment.yaml          # the real run, ~30 min
```

`experiment` trains all three models on identical data, runs the robustness sweep, checks
physical parameter recovery, draws every figure, and regenerates `docs/RESULTS.md` from the
numbers it just measured. That file is generated, never hand-edited.

Other commands:

```bash
uv run pinnbp train --config configs/experiment.yaml --model pinn
uv run pinnbp gallery                    # one window under each artifact mechanism
uv run pinnbp figures --run reports/pinn
```

## What is being compared

| | |
|---|---|
| **PINN** | pressure field + physical parameters + Windkessel losses |
| **CNN** | *matched ablation* — identical encoder, width, depth, optimiser and schedule; regresses SBP/DBP directly, no physics |
| **Ridge** | *sanity floor* — 20 classical pulse-wave-analysis features, linear model, subject-grouped CV |
| **Mean predictor** | reported automatically; the floor every model must clear |

The CNN is not a strawman: it has slightly more head capacity than the PINN's parameter head,
so if it loses, it does not lose on parameter count.

## Honesty notes

These matter more than the headline numbers, so they are at the top level rather than buried.

- **The default results are on simulated data.** The cohort comes from a physiological
  simulator ([`pinnbp/data/synthetic.py`](src/pinnbp/data/synthetic.py)), not from people.
  Those numbers validate the *pipeline* — physics, losses, splits, metrics — and demonstrate
  what is in principle recoverable. They are **not** clinical accuracy. Real-dataset adapters
  (PPG-BP, UCI/MIMIC) are in [`pinnbp/data/real.py`](src/pinnbp/data/real.py); see
  [`docs/DATA.md`](docs/DATA.md).
- **Splits are subject-disjoint and asserted.** Random splitting of overlapping windows is the
  standard way this literature inflates results, often threefold. `configs/leakage_demo.yaml`
  measures that inflation deliberately, and it is reported as a leakage figure, not a result.
- **AAMI/BHS numbers are threshold checks on a test split**, not device validations under the
  ANSI/AAMI/ISO 81060-2 protocol. The `note` field in every AAMI result says so.
- **The measured decay constant is an uncalibrated proxy.** On a band-passed PPG it correlates
  with true R·C at r ≈ 0.58 but sits ~4× low, because the 0.5 Hz high-pass removes most of a
  2-second exponential. The `tau` loss therefore constrains *rank*, not value — comparing them
  absolutely drove compliance four times too low, which is the kind of bug that fails silently.

## Layout

```
src/pinnbp/
  physics/    windkessel.py   differentiable solver, closed-form steady state
              losses.py       collocation residual, periodicity, tau, priors
  dsp/        preprocess.py   filtering, derivatives, beat segmentation
              features.py     PWA features + diastolic decay constant
              sqi.py          four-index signal quality
  data/       synthetic.py    physiological simulator
              artifacts.py    six motion-artifact mechanisms
              real.py         PPG-BP and UCI adapters
              datasets.py     channels + subject-disjoint splitting
  models/     encoder.py      shared 1-D CNN
              pinn.py         parameter head + pressure field
              baselines.py    matched CNN, ridge, mean predictor
  train.py  evaluate.py  metrics.py  plots.py  cli.py
configs/    experiment · quick · ablation_no_physics · ablation_raw_basis · leakage_demo
docs/       RESULTS.md (generated) · ARCHITECTURE.md · DATA.md · PROJECT_STATE.md · ROADMAP.md
```

## Hardware

Developed and run entirely on a 6-thread CPU with no GPU and ~1 GB free RAM. Everything here
is sized for that: the encoder is ~190k parameters, torch is pinned to the CPU wheel, and the
simulator exists partly so the pipeline can be exercised without loading a multi-gigabyte
dataset first. The full experiment takes roughly half an hour.

## License

MIT — see [LICENSE](LICENSE).
