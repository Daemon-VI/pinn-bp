# Architecture

## Why a pressure field instead of two numbers

A network that regresses SBP and DBP directly gives physics nowhere to act. You can add a
penalty on the outputs, but two scalars carry no dynamics, so any "physics" term reduces to a
prior on plausible values — a regulariser wearing a lab coat.

So the model predicts a continuous pressure waveform `P(z, t)` and reads SBP and DBP off it as
the maximum and minimum over one cardiac cycle. Now the Windkessel ODE has a function to
constrain, its time derivative is available by autograd, and the readings inherit the
constraint instead of being free.

```
PPG, VPG, APG  ──►  PPGEncoder  ──►  z (128)
                                      ├──►  ParameterHead  ──►  R, C, SV, HR, Tsys, Zc
                                      └──►  PressureField(z, t)  ──►  P(t) [mmHg]
                                                                        │
                                                          SBP = max P,  DBP = min P
```

## Components

**`PPGEncoder`** (~190k params). Four residual stages, dilation doubling with depth so the
receptive field spans more than a cardiac cycle without throwing away temporal resolution.
Squeeze-excitation channel gates rather than attention — at this data scale attention overfits
and costs far more. Both average and max pooling: average captures sustained morphology, max
captures the sharpest event, which is the systolic upstroke (and, usefully, a motion spike).

**GroupNorm, not BatchNorm.** Not a style preference. The robustness sweep evaluates on
batches whose corruption level is homogeneous, so batch statistics would shift systematically
between conditions and the measured degradation would be partly an artifact of normalisation.

**`ParameterHead`.** Six values through a scaled sigmoid into generous physiological ranges —
a *hard* guarantee that R is positive and compliance non-zero, not a penalty the optimiser
might trade away. Ranges are deliberately wider than the hinge bands in `physiological_prior`,
so the soft prior still bites near the extremes. The bias is initialised by inverting the
sigmoid at healthy-adult values (R=1.1, C=1.6, SV=70, HR=72, Tsys=0.30), so an untrained
network predicts a plausible person. Initialising at the *range midpoint* would start every
subject at 112 bpm, which sets the wrong period and therefore the wrong time axis for the
field.

**`PressureField`.** An MLP over `(z, cardiac phase)`. Continuous in `t`, which is what makes
the collocation residual exact — `dP/dt` comes from autograd, not from finite differences on a
grid. Phase is expanded in **integer harmonics**, so `P(t)` is periodic with period `T` by
construction; steady-state arterial pressure is periodic, and building that in beats
penalising its violation. Eight harmonics band-limits to ~10 Hz, which covers the arterial
pressure spectrum while excluding noise a free-form decoder would happily fit.

Output is parameterised `P = 90 + 40·f(·)` with small final weights, so an untrained field is
a flat, physiologically plausible 90 mmHg and the physics terms are meaningful from step one.

## The loss

| term | what it does | default |
|---|---|---|
| `data` | Huber on SBP/DBP, δ=5 mmHg to match the AAMI tolerance | 1.0 |
| `ode` | Windkessel collocation residual, flow-normalised | 0.10 |
| `periodicity` | cycle closure — near-vacuous under `harmonic`, load-bearing under `raw` | 0.05 |
| `tau` | **rank** agreement between R·C and the measured decay constant | 0.05 |
| `map_rule` | field mean vs DBP + PP/3 | 0.02 |
| `prior` | one-sided hinges on parameters, SBP>DBP, pulse pressure | 0.10 |

Weights are set so no single physics term exceeds the data term at initialisation. Random
collocation points, resampled every step, stop the decoder satisfying the ODE at fixed
positions and misbehaving between them. The residual is divided by the subject's peak ejection
flow, otherwise large-stroke-volume subjects would silently get more physics weight.

**Physics ramp.** Physics weights scale by `min(1, epoch/10)`. An untrained field is nearly
flat, so its ODE residual is large and carries almost no useful direction; at full strength
from step zero the field collapses onto the smoothest waveform satisfying the ODE for some
degenerate parameter set, and the data term must climb back out.

**Early stopping on validation MAE, not validation loss.** For the PINN these are different
objectives — total loss includes physics terms a model can reduce while getting clinically
worse.

## Three findings that changed the design

Recorded because each was a silent failure that tests caught, and each would otherwise have
degraded results without ever raising an error.

1. **Burn-in.** With `p_init=80` and six beats, SBP still remembered the initial condition —
   at τ=1.76 s, ten beats is only ~4.9τ. Starting at the analytic mean MAP=CO·R was *worse*
   than it sounds: the cycle begins near diastolic pressure, not the mean, leaving ~3 mmHg
   after four beats. The 2-element Windkessel is linear, so its limit cycle has a closed form
   (`steady_state_p0`); using it removes burn-in entirely — one beat now matches twenty to
   0.006 mmHg.

2. **Bramwell–Hill.** Substituting *total* arterial compliance into the *segmental* formula
   gave 2–4 m/s, where real large-artery PWV is 5–12. The inverse-square-root scaling is kept
   exactly and the prefactor is calibrated to a population reference point — called a
   calibration, because dressing it up as a derivation would be a lie.

3. **The decay constant.** The original estimator subtracted `seg.min()` as the asymptote,
   which is not the asymptote but the last sample of a still-decaying curve; on a known τ=1.4 s
   decay it returned 0.16 s. Replaced with a derivative regression (`dy/dt` against `y`), where
   the baseline falls out of the intercept and never needs estimating — exact on the same test.
   Then a second, deeper problem: even estimated correctly, the value from a band-passed PPG is
   ~4× too low, because the 0.5 Hz high-pass removes most of a 2 s exponential. It correlates
   with true R·C at r≈0.58 but is not on the same scale, so `tau_consistency` was rewritten to
   constrain **rank** within a batch. Matching absolutely drove compliance 4× wrong, silently.

## Hardware envelope

6-thread CPU, no GPU, ~1 GB free RAM. Torch pinned to the CPU wheel; nothing pulls a CUDA
runtime. `torch.set_num_threads(5)` leaves a thread free — saturating all six makes the machine
unresponsive and, with this little RAM, slower overall. The full experiment is roughly half an
hour. `n_dense=128` samples per cycle is ~7 ms resolution; below ~64 the sampled maximum sits
measurably under the true peak and SBP biases low.
