# Roadmap

Ordered by what would most change what the project can honestly claim, not by effort.

## Next — moves the claim from "pipeline works" to "it works on people"

1. **Run on PPG-BP.** The adapter is written and the interface is identical; it needs the
   download and a config. 219 subjects, subject-disjoint, small enough for this machine. This
   is the single highest-value next step: every headline number today is simulated.
   *Caveat to state up front:* 2.1 s segments and one cuff reading per subject means it tests
   between-subject prediction, not within-subject tracking.

2. **Run on UCI/MIMIC for within-subject tracking.** The only way to test whether the model
   follows a pressure *change* rather than ranking people. Needs `h5py`, streaming, and honest
   reporting that record ≠ subject there, so some leakage may survive.

3. **Report the ablations.** `configs/ablation_no_physics.yaml` and
   `ablation_raw_basis.yaml` exist and run; their numbers are not yet in `RESULTS.md`. The
   no-physics ablation is the one that decides whether the central claim of this project holds:
   same architecture, physics weights zeroed. If it matches the PINN, the physics is decorative
   and the report should say so.

4. **Run the leakage demo and publish the gap.** `configs/leakage_demo.yaml`. Quantifying how
   much a random split inflates results on *this* data is a genuinely useful contribution to a
   report, because most papers in this area never show it.

## After that

5. **Calibration / personalisation.** A single cuff reading per user to anchor the offset is
   what actual cuffless devices do, and it converts a between-subject model into a usable one.
   Currently unmodelled.

6. **Multiple seeds and confidence intervals.** Every number today is one seed. Differences of
   1–2 mmHg between models are not currently distinguishable from noise, and the report should
   not pretend otherwise.

7. **Real-time constraint.** `filtfilt` is non-causal and the whole pipeline is offline. A
   wearable needs causal filtering with delay compensation; measuring what that costs would
   make the deployment story honest.

8. **Uncertainty output.** The SQI gates windows, but the model reports no confidence on the
   ones it accepts. A pressure estimate without an interval is the wrong output for a clinical
   device.

## Explicitly not planned

- **Chasing a lower MAE on synthetic data.** It measures the simulator, not the method.
- **Local GPU training.** No CUDA on this machine; if the model ever needs it, the run leaves
  the machine (see the `hardware-budget` skill).
- **Claiming AAMI/BHS compliance.** Those are validation protocols with specified populations
  and reference procedures. This project computes the same arithmetic on a test split, which is
  a different thing, and the code says so in the result object itself.
