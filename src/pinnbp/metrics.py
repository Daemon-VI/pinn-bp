"""Evaluation metrics, including the clinical standards.

MAE alone is not enough to say anything about a blood pressure device, so this module
computes what the field actually requires:

**AAMI / ANSI-AAMI-ISO 81060-2.** A device passes if the mean error is within +/-5 mmHg and
the standard deviation of the error is at most 8 mmHg, for SBP and DBP independently. Mean
and SD, not MAE -- a model can post an excellent MAE and still fail, because MAE hides
bias, and a device that reads 8 mmHg low on everyone is dangerous in a way its MAE does not
reveal.

**BHS grading.** Grades A/B/C by the cumulative proportion of absolute errors within 5, 10
and 15 mmHg. Grade A needs 60/85/95%. This catches a different failure from AAMI: a model
with good average behaviour and a heavy error tail fails BHS while passing AAMI.

An honest caveat that must accompany every use of these functions in this project: both
standards are defined for a validation protocol with a specified subject population,
reference measurement procedure and pressure distribution. Computing the same arithmetic on
a held-out split of a public dataset -- let alone on simulated data -- is *not* a device
validation and cannot be reported as "AAMI compliant". The correct phrasing, used throughout
``docs/RESULTS.md``, is "meets the AAMI error thresholds on this test split". The
:func:`aami_check` result carries that caveat in its own ``note`` field so it travels with
the number.

There is one more reason to be careful. On a cohort with narrow pressure spread, predicting
the mean can pass AAMI outright while having zero predictive value, which is why
:func:`regression_report` always reports the mean-predictor baseline alongside and why
:func:`correlation_report` reports how much of the variance was actually explained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "ErrorStats",
    "AAMIResult",
    "BHSResult",
    "error_stats",
    "aami_check",
    "bhs_grade",
    "bland_altman",
    "regression_report",
    "correlation_report",
]


@dataclass
class ErrorStats:
    """Basic error statistics for one pressure channel, all in mmHg."""

    n: int
    mae: float
    rmse: float
    me: float
    sd: float
    median_ae: float
    p95_ae: float
    max_ae: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class AAMIResult:
    """AAMI / ISO 81060-2 error thresholds applied to a test split."""

    me: float
    sd: float
    n: int
    passes: bool
    note: str = (
        "Threshold check on a held-out split, not a device validation under the "
        "ANSI/AAMI/ISO 81060-2 protocol. Do not report as 'AAMI compliant'."
    )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BHSResult:
    """British Hypertension Society cumulative-error grading."""

    within_5: float
    within_10: float
    within_15: float
    grade: str

    def as_dict(self) -> dict:
        return asdict(self)


def error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> ErrorStats:
    """Compute error statistics for one channel.

    ``sd`` is the standard deviation of the *signed* error, which is the quantity AAMI
    bounds. It is not the standard deviation of the absolute error, and confusing the two
    makes a failing model look like a passing one.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    ae = np.abs(err)
    return ErrorStats(
        n=int(len(err)),
        mae=float(ae.mean()),
        rmse=float(np.sqrt((err**2).mean())),
        me=float(err.mean()),
        sd=float(err.std(ddof=1)) if len(err) > 1 else 0.0,
        median_ae=float(np.median(ae)),
        p95_ae=float(np.percentile(ae, 95)),
        max_ae=float(ae.max()),
    )


def aami_check(y_true: np.ndarray, y_pred: np.ndarray) -> AAMIResult:
    """Apply the AAMI mean-error and standard-deviation thresholds.

    Also enforces the standard's minimum of 85 subject measurements: below that the check
    is reported as not passing regardless of the numbers, because the thresholds are not
    meaningful on a handful of samples.
    """
    st = error_stats(y_true, y_pred)
    passes = abs(st.me) <= 5.0 and st.sd <= 8.0 and st.n >= 85
    return AAMIResult(me=st.me, sd=st.sd, n=st.n, passes=bool(passes))


def bhs_grade(y_true: np.ndarray, y_pred: np.ndarray) -> BHSResult:
    """Grade by cumulative absolute error at the 5/10/15 mmHg thresholds.

    Grade A requires 60/85/95%, B requires 50/75/90%, C requires 40/65/85%; anything less
    is D. All three thresholds must be met for a grade, so the returned grade is the best
    one whose every requirement is satisfied.
    """
    err = np.abs(np.asarray(y_pred, dtype=np.float64).ravel()
                 - np.asarray(y_true, dtype=np.float64).ravel())
    w5 = float((err <= 5).mean() * 100)
    w10 = float((err <= 10).mean() * 100)
    w15 = float((err <= 15).mean() * 100)

    if w5 >= 60 and w10 >= 85 and w15 >= 95:
        grade = "A"
    elif w5 >= 50 and w10 >= 75 and w15 >= 90:
        grade = "B"
    elif w5 >= 40 and w10 >= 65 and w15 >= 85:
        grade = "C"
    else:
        grade = "D"

    return BHSResult(within_5=w5, within_10=w10, within_15=w15, grade=grade)


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Bland-Altman agreement statistics.

    Returns the bias and the 95% limits of agreement. Bland-Altman is the right tool for
    comparing two measurement methods -- correlation is not, because two methods can
    correlate near-perfectly while disagreeing by a constant 20 mmHg.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    diff = y_pred - y_true
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "mean_of_means": float(((y_true + y_pred) / 2).mean()),
    }


def correlation_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Pearson r and the coefficient of determination.

    R^2 is computed against the variance of the truth, so a model that always predicts the
    mean scores exactly 0.0 and a model that is worse than the mean scores negative. That
    negative range is the useful part: it is the unambiguous signal that a model has not
    learned anything about the individual, which a correlation coefficient alone can hide.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2 or y_true.std() < 1e-12:
        return {"pearson_r": float("nan"), "r2": float("nan")}
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {"pearson_r": r, "r2": 1.0 - ss_res / ss_tot}


def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, train_mean: np.ndarray | None = None
) -> dict:
    """Full report for a ``(n, 2)`` SBP/DBP prediction.

    Args:
        y_true: ``(n, 2)`` ground truth, columns (SBP, DBP).
        y_pred: ``(n, 2)`` predictions.
        train_mean: ``(2,)`` training-set mean. When given, the same metrics are computed
            for a constant mean predictor and included under ``"mean_baseline"``, so every
            reported number arrives with its floor attached rather than needing a separate
            lookup to interpret.

    Returns:
        Nested dict keyed by ``"sbp"``/``"dbp"``, each with error stats, AAMI, BHS,
        Bland-Altman and correlation.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 2)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 2)

    out: dict = {}
    for i, name in enumerate(("sbp", "dbp")):
        out[name] = {
            **error_stats(y_true[:, i], y_pred[:, i]).as_dict(),
            "aami": aami_check(y_true[:, i], y_pred[:, i]).as_dict(),
            "bhs": bhs_grade(y_true[:, i], y_pred[:, i]).as_dict(),
            "bland_altman": bland_altman(y_true[:, i], y_pred[:, i]),
            **correlation_report(y_true[:, i], y_pred[:, i]),
        }

    if train_mean is not None:
        const = np.tile(np.asarray(train_mean, dtype=np.float64).reshape(1, 2), (len(y_true), 1))
        out["mean_baseline"] = {
            name: error_stats(y_true[:, i], const[:, i]).as_dict()
            for i, name in enumerate(("sbp", "dbp"))
        }
        # The headline comparison: how much of the mean predictor's error was removed.
        out["skill_vs_mean"] = {
            name: 1.0 - out[name]["mae"] / max(out["mean_baseline"][name]["mae"], 1e-9)
            for name in ("sbp", "dbp")
        }

    return out
