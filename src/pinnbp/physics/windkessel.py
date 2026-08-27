"""Differentiable lumped-parameter arterial models.

The whole project rests on this file. A "physics-informed" network is only worth the name
if the physics is a real forward model that gradients can flow through, so the Windkessel
solver here is written in torch and integrated with fixed-step RK4 rather than handed off
to SciPy. Every quantity carries clinical units, because the loss terms compare model
output against millimetres of mercury measured on a person.

Units used throughout (do not mix these -- the physiological priors in
``pinnbp.physics.losses`` assume them):

    P     pressure                  mmHg
    Q     volumetric flow           mL/s
    R     peripheral resistance     mmHg*s/mL      typical 0.7 - 1.6
    C     arterial compliance       mL/mmHg        typical 0.8 - 2.5
    Zc    characteristic impedance  mmHg*s/mL      typical 0.03 - 0.10
    SV    stroke volume             mL             typical 45 - 110
    HR    heart rate                bpm            typical 45 - 130
    Tsys  systolic ejection time    s              typical 0.24 - 0.36

Sanity anchor, worth keeping in mind when reading the tests: SV=70 mL, HR=70 bpm,
R=1.1 mmHg*s/mL gives cardiac output 81.7 mL/s and therefore MAP = CO*R ~= 90 mmHg, while
C=1.6 mL/mmHg gives pulse pressure ~= SV/C ~= 44 mmHg. That lands on roughly 119/75, which
is what a healthy adult should read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "WindkesselParams",
    "ejection_flow",
    "simulate",
    "beat_pressures",
    "steady_state_p0",
    "map_from_sbp_dbp",
    "bramwell_hill_pwv",
]


@dataclass
class WindkesselParams:
    """A batch of subject-level cardiovascular parameters.

    Every field is a tensor of shape ``(B,)`` so a whole minibatch is simulated at once.
    The network predicts these; the solver turns them into a pressure waveform; the loss
    compares that waveform's SBP/DBP against the cuff labels. That round trip is the
    physics constraint.
    """

    R: Tensor
    C: Tensor
    SV: Tensor
    HR: Tensor
    Tsys: Tensor
    Zc: Tensor | None = None

    @property
    def tau(self) -> Tensor:
        """Diastolic decay time constant, tau = R*C, in seconds.

        This is the most directly observable physical quantity in the whole model: during
        diastole the flow source is off, so pressure decays as exp(-t/tau). The same decay
        shows up in the diastolic limb of a PPG pulse, which is why
        ``pinnbp.dsp.features.diastolic_tau`` can measure it straight off the signal and
        the loss can hold the network's implied R*C to it.
        """
        return self.R * self.C

    @property
    def period(self) -> Tensor:
        """Cardiac period T = 60/HR, in seconds."""
        return 60.0 / self.HR

    @property
    def cardiac_output(self) -> Tensor:
        """CO = SV * HR / 60, in mL/s."""
        return self.SV * self.HR / 60.0

    def to(self, device: torch.device | str) -> WindkesselParams:
        return WindkesselParams(
            R=self.R.to(device),
            C=self.C.to(device),
            SV=self.SV.to(device),
            HR=self.HR.to(device),
            Tsys=self.Tsys.to(device),
            Zc=None if self.Zc is None else self.Zc.to(device),
        )


def ejection_flow(t_in_beat: Tensor, SV: Tensor, Tsys: Tensor) -> Tensor:
    """Aortic inflow Q(t) for a half-sine ejection profile.

    Q(t) = Q0 * sin(pi * t / Tsys) while 0 <= t < Tsys, and zero through diastole. Q0 is
    fixed by conservation of volume rather than chosen freely: integrating the half-sine
    over the ejection window gives Q0 * 2*Tsys/pi, so Q0 = pi*SV/(2*Tsys) makes the area
    under the curve exactly one stroke volume. That is what keeps SV interpretable as
    millilitres instead of drifting into an arbitrary gain the network can abuse.

    The half-sine is the standard first-order approximation to an aortic flow pulse. It
    omits the backflow notch at valve closure, which is why the 3-element form (Zc) exists
    in :func:`simulate` -- Zc is what puts a recognisable incisura back into the waveform.

    Args:
        t_in_beat: time since the start of the current beat, any shape.
        SV: stroke volume in mL, broadcastable to ``t_in_beat``.
        Tsys: ejection duration in s, broadcastable to ``t_in_beat``.

    Returns:
        Flow in mL/s, same shape as ``t_in_beat``.
    """
    q0 = torch.pi * SV / (2.0 * Tsys)
    phase = torch.pi * t_in_beat / Tsys
    systolic = q0 * torch.sin(phase)
    # torch.where rather than in-place masking: this has to stay differentiable, and the
    # gradient through the diastolic branch must be exactly zero, not merely small.
    return torch.where(
        (t_in_beat >= 0) & (t_in_beat < Tsys), systolic, torch.zeros_like(systolic)
    )


def _dPdt(P: Tensor, Q: Tensor, R: Tensor, C: Tensor) -> Tensor:
    """Right-hand side of the 2-element Windkessel ODE: C dP/dt + P/R = Q."""
    return (Q - P / R) / C


def steady_state_p0(params: WindkesselParams) -> Tensor:
    """Closed-form pressure at the start of a beat, at steady state.

    The 2-element Windkessel is linear, so for a periodic flow input its limit cycle has an
    exact solution and there is no need to integrate towards it at all.

    Solving P' = (Q - P/R)/C by variation of parameters over one period and imposing
    P(T) = P(0) gives::

        P(0) = (1/C) * exp(-T/tau) * I / (1 - exp(-T/tau)),
        I    = integral over [0, Tsys] of Q(s) * exp(s/tau) ds

    and with the half-sine ejection profile Q(s) = Q0 sin(pi s / Tsys) that integral is
    itself closed-form, via the standard exp-times-sine antiderivative::

        I = Q0 * b * (exp(Tsys/tau) + 1) / (a^2 + b^2),   a = 1/tau,  b = pi/Tsys

    This matters because the obvious initial guesses are both wrong. Starting at a fixed
    80 mmHg makes the answer depend on an arbitrary constant. Starting at the analytic mean
    MAP = CO*R is worse than it sounds -- the mean is not the phase-zero value, the cycle
    *begins* near diastolic pressure, so a MAP start is roughly a pulse-pressure's worth too
    high and takes several tau to decay. At tau = 1.76 s that is a genuine multi-mmHg bias
    after four beats.

    Fully differentiable, so it costs nothing in the autograd graph.

    Returns:
        ``(B,)`` pressure in mmHg at t = 0 of a steady-state beat.
    """
    tau = params.tau
    T = params.period
    Tsys = params.Tsys
    q0 = torch.pi * params.SV / (2.0 * Tsys)

    a = 1.0 / tau
    b = torch.pi / Tsys

    integral = q0 * b * (torch.exp(Tsys / tau) + 1.0) / (a * a + b * b)
    decay = torch.exp(-T / tau)
    return (integral / params.C) * decay / (1.0 - decay)


def windkessel_residual(P: Tensor, dPdt: Tensor, Q: Tensor, R: Tensor, C: Tensor) -> Tensor:
    """Pointwise ODE residual r = C dP/dt + P/R - Q, in mL/s.

    Zero everywhere iff the waveform satisfies the Windkessel relation for those
    parameters. Exposed separately from the solver because the collocation loss evaluates
    it on waveforms the solver did not produce -- that is the whole point of a PINN.
    """
    return C.unsqueeze(-1) * dPdt + P / R.unsqueeze(-1) - Q


def simulate(
    params: WindkesselParams,
    steps_per_beat: int = 128,
    n_beats: int = 8,
    p_init: float | None = None,
    return_all_beats: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Integrate the Windkessel model and return the steady-state pressure waveform.

    Fixed-step RK4 with a per-sample step size dt = T/steps_per_beat. Fixed-step matters:
    an adaptive solver would make the number of autograd graph nodes depend on the
    parameter values, which makes backprop cost unpredictable and, on a machine with about
    a gigabyte free, occasionally fatal.

    ``n_beats`` beats are integrated but by default only the last is returned. The earlier
    beats are burn-in -- pressure starts at ``p_init`` and relaxes onto its limit cycle,
    with any initial offset decaying as exp(-t/tau) where tau = R*C is typically 1.2-2.5 s.

    That decay is slower than it first looks, and getting it wrong biases every SBP the
    model produces: at tau = 1.76 s and 70 bpm, ten beats is only ~4.9 tau and still leaves
    around a millimetre of mercury. Rather than paying for a long burn-in, the default
    ``p_init`` is the *exact* phase-zero pressure of the limit cycle from
    :func:`steady_state_p0`, which makes burn-in unnecessary -- the first returned beat is
    already the steady-state one. Passing an explicit float restores the old behaviour, and
    then ``n_beats >= 16`` is needed for the result to be independent of it.

    Args:
        params: batch of cardiovascular parameters, each field of shape ``(B,)``.
        steps_per_beat: RK4 steps per cardiac cycle. 128 holds SBP within ~0.25 mmHg of an
            adaptive SciPy reference solve (see ``tests/test_windkessel.py``) at a fraction
            of the cost.
        n_beats: total beats integrated, including burn-in.
        p_init: initial pressure in mmHg. ``None`` (default) uses the analytic MAP = CO*R
            per subject.
        return_all_beats: return every beat instead of only the final one. Useful for
            plots and for checking that the limit cycle really has settled.

    Returns:
        ``(pressure, flow, t)``. Pressure and flow have shape ``(B, steps_per_beat)``, or
        ``(B, n_beats*steps_per_beat)`` when ``return_all_beats`` is set; ``t`` is the
        matching time axis in seconds and has the same shape.
    """
    R, C, SV, Tsys = params.R, params.C, params.SV, params.Tsys
    T = params.period
    B = R.shape[0]
    device, dtype = R.device, R.dtype

    dt = T / steps_per_beat  # (B,)
    if p_init is None:
        # Exact phase-zero pressure of the limit cycle. See steady_state_p0 -- this removes
        # burn-in entirely rather than shortening it.
        P = steady_state_p0(params)
    else:
        P = torch.full((B,), float(p_init), device=device, dtype=dtype)

    total_steps = n_beats * steps_per_beat
    keep_from = 0 if return_all_beats else (n_beats - 1) * steps_per_beat

    p_hist: list[Tensor] = []
    q_hist: list[Tensor] = []
    t_hist: list[Tensor] = []

    for i in range(total_steps):
        # Phase within the current beat as a fraction, so a per-sample period T can be
        # applied without a Python-level branch on tensor values.
        frac = float(i % steps_per_beat) / steps_per_beat
        t_beat = frac * T  # (B,)

        q1 = ejection_flow(t_beat, SV, Tsys)
        q2 = ejection_flow(t_beat + 0.5 * dt, SV, Tsys)
        q4 = ejection_flow(t_beat + dt, SV, Tsys)

        k1 = _dPdt(P, q1, R, C)
        k2 = _dPdt(P + 0.5 * dt * k1, q2, R, C)
        k3 = _dPdt(P + 0.5 * dt * k2, q2, R, C)
        k4 = _dPdt(P + dt * k3, q4, R, C)
        P_next = P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if i >= keep_from:
            # Record the state at the start of the step alongside the flow driving it, so
            # pressure and flow samples share a time stamp.
            p_hist.append(P)
            q_hist.append(q1)
            t_hist.append(t_beat + (i // steps_per_beat) * T)

        P = P_next

    pressure = torch.stack(p_hist, dim=1)
    flow = torch.stack(q_hist, dim=1)
    t = torch.stack(t_hist, dim=1)

    if params.Zc is not None:
        # 3-element Windkessel: the characteristic impedance of the proximal aorta adds a
        # term in phase with flow, P = P_wk + Zc*Q. This sharpens the systolic upstroke and
        # restores the incisura; the 2-element form otherwise reads a few mmHg low at the
        # systolic peak.
        pressure = pressure + params.Zc.unsqueeze(1) * flow

    return pressure, flow, t


def beat_pressures(pressure: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce a pressure waveform to (SBP, DBP, MAP).

    MAP here is the true time average of the waveform, not the ``DBP + PP/3`` rule of
    thumb. Keeping the two distinct is deliberate: the gap between them is exactly what
    :func:`pinnbp.physics.losses.map_consistency` penalises, and collapsing them here would
    make that term vacuously zero.

    Args:
        pressure: shape ``(B, N)``.

    Returns:
        ``(sbp, dbp, map_)``, each of shape ``(B,)``, in mmHg.
    """
    sbp = pressure.max(dim=1).values
    dbp = pressure.min(dim=1).values
    map_ = pressure.mean(dim=1)
    return sbp, dbp, map_


def map_from_sbp_dbp(sbp: Tensor, dbp: Tensor) -> Tensor:
    """The clinical rule of thumb MAP ~= DBP + (SBP - DBP)/3.

    Exact only when the diastolic fraction of the cycle is about two thirds, which is why
    it drifts at high heart rates. Used as a soft constraint, never as ground truth.
    """
    return dbp + (sbp - dbp) / 3.0


def bramwell_hill_pwv(C: Tensor, pwv_ref: float = 8.0, c_ref: float = 1.6) -> Tensor:
    """Pulse wave velocity from compliance, using the Bramwell-Hill scaling.

        PWV = pwv_ref * sqrt(c_ref / C)

    This is the bridge between a quantity the network predicts (compliance) and a quantity
    the *timing* of the PPG reveals (how long the pulse took to arrive). Without it,
    compliance would only ever be constrained through pulse amplitude -- and amplitude is
    exactly what motion artifact destroys first, so a robustness claim would otherwise rest
    on the most fragile part of the signal.

    **On the prefactor, which is calibrated rather than derived.** Bramwell-Hill proper is
    PWV = sqrt(V / (rho * C_segment)), where C_segment is the compliance *of the vessel
    segment the wave is travelling through*. The ``C`` this project predicts is total
    arterial compliance -- a lumped, whole-tree quantity of order 1.6 mL/mmHg. Substituting
    one for the other is not dimensionally wrong but it is numerically wrong: doing so with
    any plausible segment volume yields 2-4 m/s, where real large-artery PWV is 5-12 m/s.

    So the physically meaningful content of the equation -- that wave speed scales as the
    inverse square root of compliance -- is kept exactly, and the constant of
    proportionality is set by a population reference point (a typical total compliance of
    1.6 mL/mmHg corresponding to a typical PWV of 8 m/s) instead of being derived from a
    segment geometry that does not correspond to the lumped model. Calling it a derivation
    would be dressing up a calibration.

    Args:
        C: total arterial compliance in mL/mmHg.
        pwv_ref: PWV in m/s at the reference compliance.
        c_ref: reference compliance in mL/mmHg.

    Returns:
        PWV in m/s.
    """
    return pwv_ref * torch.sqrt(c_ref / C.clamp(min=1e-6))
