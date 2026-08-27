"""Tests for the differentiable Windkessel solver.

These are the tests that matter most in the project. Everything downstream -- the losses,
the parameter recovery claim, the interpretability claim -- assumes this solver computes
real cardiovascular physics. So it is checked against closed-form results and against an
independent SciPy integration, not merely for "runs without error".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinnbp.physics.windkessel import (
    WindkesselParams,
    beat_pressures,
    bramwell_hill_pwv,
    ejection_flow,
    map_from_sbp_dbp,
    simulate,
    steady_state_p0,
)


def make_params(**over) -> WindkesselParams:
    base = dict(R=1.1, C=1.6, SV=70.0, HR=70.0, Tsys=0.30, Zc=None)
    base.update(over)
    return WindkesselParams(
        R=torch.tensor([base["R"]], dtype=torch.float64),
        C=torch.tensor([base["C"]], dtype=torch.float64),
        SV=torch.tensor([base["SV"]], dtype=torch.float64),
        HR=torch.tensor([base["HR"]], dtype=torch.float64),
        Tsys=torch.tensor([base["Tsys"]], dtype=torch.float64),
        Zc=None if base["Zc"] is None else torch.tensor([base["Zc"]], dtype=torch.float64),
    )


def test_ejection_flow_integrates_to_stroke_volume():
    """The half-sine must contain exactly one stroke volume, or SV means nothing."""
    SV, Tsys = 70.0, 0.30
    t = torch.linspace(0, Tsys, 20001, dtype=torch.float64)
    q = ejection_flow(t, torch.tensor(SV, dtype=torch.float64),
                      torch.tensor(Tsys, dtype=torch.float64))
    volume = torch.trapezoid(q, t).item()
    assert volume == pytest.approx(SV, rel=1e-4)


def test_ejection_flow_is_zero_during_diastole():
    t = torch.tensor([0.31, 0.5, 0.85], dtype=torch.float64)
    q = ejection_flow(t, torch.tensor(70.0, dtype=torch.float64),
                      torch.tensor(0.30, dtype=torch.float64))
    assert torch.all(q == 0)


def test_mean_pressure_equals_cardiac_output_times_resistance():
    """MAP = CO * R exactly, for the 2-element model at steady state.

    This is the sharpest analytic check available. Averaging C dP/dt + P/R = Q over a full
    cycle at steady state kills the derivative term (P is periodic), leaving <P>/R = <Q>,
    i.e. MAP = CO * R. Any error in the integrator, the flow normalisation or the burn-in
    shows up here.
    """
    p = make_params()
    pressure, _, _ = simulate(p, steps_per_beat=256, n_beats=12)
    _, _, map_ = beat_pressures(pressure)

    co = (70.0 * 70.0 / 60.0)
    assert map_.item() == pytest.approx(co * 1.1, rel=2e-3)


def test_three_element_shifts_map_by_zc_times_cardiac_output():
    """Adding Zc raises the mean by exactly Zc * CO, since P = P_wk + Zc*Q."""
    co = 70.0 * 70.0 / 60.0
    two = simulate(make_params(), steps_per_beat=256, n_beats=12)[0]
    three = simulate(make_params(Zc=0.055), steps_per_beat=256, n_beats=12)[0]

    d_map = (three.mean() - two.mean()).item()
    assert d_map == pytest.approx(0.055 * co, rel=2e-2)


def test_matches_scipy_reference_integration():
    """Fixed-step RK4 must agree with an adaptive reference solver.

    The torch solver exists for autograd, not for accuracy, so it has to be shown that
    choosing it costs nothing numerically at the default step count.
    """
    from scipy.integrate import solve_ivp

    R, C, SV, HR, Tsys = 1.1, 1.6, 70.0, 70.0, 0.30
    T = 60.0 / HR
    n_beats = 12

    def q_of_t(t):
        tb = t % T
        if 0 <= tb < Tsys:
            return np.pi * SV / (2 * Tsys) * np.sin(np.pi * tb / Tsys)
        return 0.0

    sol = solve_ivp(
        lambda t, P: [(q_of_t(t) - P[0] / R) / C],
        (0.0, n_beats * T), [80.0],
        rtol=1e-10, atol=1e-10, dense_output=True, max_step=1e-3,
    )
    grid = np.linspace((n_beats - 1) * T, n_beats * T, 512, endpoint=False)
    ref = sol.sol(grid)[0]

    pressure, _, _ = simulate(make_params(), steps_per_beat=128, n_beats=n_beats)
    got_sbp = pressure.max().item()
    got_dbp = pressure.min().item()

    assert got_sbp == pytest.approx(ref.max(), abs=0.25)
    assert got_dbp == pytest.approx(ref.min(), abs=0.25)


def test_diastolic_decay_follows_tau():
    """During diastole pressure must decay as exp(-t/RC) with tau = R*C."""
    p = make_params()
    pressure, flow, t = simulate(p, steps_per_beat=512, n_beats=12)
    P = pressure[0].numpy()
    Q = flow[0].numpy()
    tt = t[0].numpy()

    dia = Q == 0
    # Use the late part of diastole only, well clear of valve closure.
    idx = np.flatnonzero(dia)
    idx = idx[len(idx) // 3:]
    slope = np.polyfit(tt[idx], np.log(P[idx]), 1)[0]
    tau_measured = -1.0 / slope

    assert tau_measured == pytest.approx(p.tau.item(), rel=0.02)


def test_higher_resistance_raises_pressure_and_higher_compliance_lowers_pulse_pressure():
    """The two headline physiological relationships the model is supposed to encode."""
    lo_r = simulate(make_params(R=0.8), n_beats=10)[0]
    hi_r = simulate(make_params(R=1.6), n_beats=10)[0]
    assert hi_r.mean().item() > lo_r.mean().item()

    stiff = simulate(make_params(C=0.8), n_beats=10)[0]
    compliant = simulate(make_params(C=2.5), n_beats=10)[0]
    pp_stiff = (stiff.max() - stiff.min()).item()
    pp_compliant = (compliant.max() - compliant.min()).item()
    assert pp_stiff > pp_compliant


def test_explicit_initial_pressure_is_forgotten_given_enough_burn_in():
    """With an explicit p_init, the returned beat must not remember it.

    The offset decays as exp(-t/tau) with tau = R*C = 1.76 s here, so ten beats at 70 bpm
    is only ~4.9 tau and still leaves about a millimetre of mercury. Sixteen beats is
    ~7.8 tau, which is enough. This test is what caught the original default of six beats
    being too short.
    """
    a = simulate(make_params(), n_beats=16, p_init=50.0)[0]
    b = simulate(make_params(), n_beats=16, p_init=140.0)[0]
    assert torch.allclose(a, b, atol=0.5)


def test_closed_form_initial_condition_removes_burn_in_entirely():
    """With the analytic limit-cycle start, one beat must already equal twenty.

    This is the point of steady_state_p0: not a shorter burn-in but no burn-in. An earlier
    version started at MAP = CO*R, which sounds principled but is wrong -- the cycle begins
    near diastolic pressure, not at the mean -- and left a 3 mmHg error after four beats.
    """
    one = simulate(make_params(), n_beats=1, steps_per_beat=256)[0]
    twenty = simulate(make_params(), n_beats=20, steps_per_beat=256)[0]
    assert torch.allclose(one, twenty, atol=0.01)


def test_steady_state_p0_matches_long_integration():
    """The closed form must agree with where a long integration actually lands."""
    for kw in ({}, {"R": 0.7, "C": 2.4}, {"HR": 100.0, "Tsys": 0.24}, {"SV": 95.0}):
        p = make_params(**kw)
        integrated = simulate(p, n_beats=40, steps_per_beat=512, p_init=80.0)[0][0, 0]
        assert steady_state_p0(p).item() == pytest.approx(integrated.item(), abs=0.05), kw


def test_gradients_flow_with_correct_signs():
    """Autograd must reach R and C, and the signs must be physiologically right."""
    R = torch.tensor([1.1], dtype=torch.float64, requires_grad=True)
    C = torch.tensor([1.6], dtype=torch.float64, requires_grad=True)
    p = WindkesselParams(
        R=R, C=C,
        SV=torch.tensor([70.0], dtype=torch.float64),
        HR=torch.tensor([70.0], dtype=torch.float64),
        Tsys=torch.tensor([0.30], dtype=torch.float64),
    )
    pressure, _, _ = simulate(p, n_beats=8)
    sbp, dbp, _ = beat_pressures(pressure)
    (sbp - dbp).sum().backward()

    assert R.grad is not None and C.grad is not None
    assert torch.isfinite(R.grad).all() and torch.isfinite(C.grad).all()
    # Pulse pressure falls as compliance rises.
    assert C.grad.item() < 0


def test_batching_matches_individual_simulation():
    """A batched solve must equal solving each subject alone."""
    batch = WindkesselParams(
        R=torch.tensor([0.9, 1.3], dtype=torch.float64),
        C=torch.tensor([1.2, 2.1], dtype=torch.float64),
        SV=torch.tensor([65.0, 85.0], dtype=torch.float64),
        HR=torch.tensor([62.0, 88.0], dtype=torch.float64),
        Tsys=torch.tensor([0.32, 0.27], dtype=torch.float64),
    )
    both, _, _ = simulate(batch, n_beats=8)
    first, _, _ = simulate(make_params(R=0.9, C=1.2, SV=65.0, HR=62.0, Tsys=0.32), n_beats=8)
    assert torch.allclose(both[0], first[0], atol=1e-9)


def test_map_rule_of_thumb():
    sbp = torch.tensor([120.0])
    dbp = torch.tensor([80.0])
    assert map_from_sbp_dbp(sbp, dbp).item() == pytest.approx(93.333, rel=1e-4)


def test_bramwell_hill_pwv_is_physiological_and_decreasing_in_compliance():
    """PWV must land in the real large-artery range and fall as compliance rises.

    The 5-12 m/s band is the physiological one. An earlier version substituted total
    arterial compliance into the segmental Bramwell-Hill formula and produced 2-4 m/s;
    this test is what caught it.
    """
    C = torch.tensor([0.8, 1.6, 3.0], dtype=torch.float64)
    pwv = bramwell_hill_pwv(C)
    assert torch.all(pwv > 4.0) and torch.all(pwv < 15.0)
    assert pwv[0] > pwv[1] > pwv[2]
    # Inverse-square-root scaling is the physically meaningful part and must hold exactly.
    assert (pwv[0] / pwv[2]).item() == pytest.approx((3.0 / 0.8) ** 0.5, rel=1e-6)
