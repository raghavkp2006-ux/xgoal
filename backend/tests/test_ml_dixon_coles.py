"""Phase 2.3 tests — the Dixon-Coles goals model.

Covers the four checks the build plan asks for (convergence, ``sum(alpha) = 0``,
score-matrix normalisation, and a strong side out-rating a relegated one from the
same season) plus two guards for bugs that have actually bitten this code:

* a finite-difference check of the analytic gradient — an inconsistent gradient
  still "works" until L-BFGS-B's line search fails, which is how the missing
  ``d log(tau)/d lambda`` terms went unnoticed; and
* a rejection test for artifacts fitted under the old defence sign convention,
  which would otherwise invert every club's defensive strength silently.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

import app.ml.dixon_coles as dc
from app.ml.dixon_coles import (
    DEFAULT_HALF_LIFE_DAYS,
    DixonColesConvergenceError,
    DixonColesModel,
    half_life_to_xi,
    tau,
    xi_to_half_life,
)

XI_540 = half_life_to_xi(DEFAULT_HALF_LIFE_DAYS)
MAX_GOALS = 10


# --------------------------------------------------------------------------- #
# convergence
# --------------------------------------------------------------------------- #


def test_fit_converges_and_records_diagnostics(synthetic_dataset):
    """A normal fit reports success, and says so in its diagnostics."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)

    assert model.diagnostics is not None
    assert model.diagnostics.converged is True
    assert model.diagnostics.n_iterations > 0
    assert model.diagnostics.status == 0
    # L-BFGS-B may converge on relative function reduction rather than on the
    # gradient, so this is a sanity bound, not a KKT check.
    assert model.diagnostics.max_abs_gradient < 1e-3
    assert np.isfinite(model.diagnostics.negative_log_likelihood)


def test_a_non_converged_fit_raises_instead_of_returning_parameters(
    synthetic_dataset, monkeypatch
):
    """The success flag is asserted: garbage must not look like a fitted model."""
    _, matches = synthetic_dataset
    real_minimize = dc.minimize

    def failing_minimize(*args, **kwargs):
        result = real_minimize(*args, **kwargs)
        result.success = False
        result.status = 2
        result.message = "ABNORMAL: injected for the test"
        return result

    monkeypatch.setattr(dc, "minimize", failing_minimize)
    with pytest.raises(DixonColesConvergenceError, match="did not converge"):
        DixonColesModel().fit(matches)


def test_analytic_gradient_matches_finite_differences(synthetic_dataset, monkeypatch):
    """The shipped gradient closure must agree with a numeric one.

    The tau correction depends on lambda and mu as well as rho; dropping those
    terms makes the analytic gradient inconsistent with the objective, and
    L-BFGS-B then aborts its line search.
    """
    _, matches = synthetic_dataset
    captured: dict[str, object] = {}
    real_minimize = dc.minimize

    def capturing_minimize(fun, x0, **kwargs):
        captured["fun"] = fun
        captured["x0"] = np.asarray(x0, dtype=np.float64)
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(dc, "minimize", capturing_minimize)
    DixonColesModel().fit(matches)

    fun = captured["fun"]
    assert callable(fun)
    pars = np.asarray(captured["x0"], dtype=np.float64).copy()
    pars = pars + np.random.default_rng(3).normal(0.0, 0.1, size=pars.shape)

    analytic = np.asarray(fun(pars)[1], dtype=np.float64)
    eps = 1e-6
    numeric = np.zeros_like(pars)
    for j in range(pars.size):
        step = np.zeros_like(pars)
        step[j] = eps
        numeric[j] = (fun(pars + step)[0] - fun(pars - step)[0]) / (2 * eps)

    assert np.max(np.abs(analytic - numeric)) < 1e-4


# --------------------------------------------------------------------------- #
# identifiability — sum(alpha) = 0
# --------------------------------------------------------------------------- #


def test_attack_ratings_satisfy_the_sum_to_zero_constraint(synthetic_dataset):
    """``sum(alpha) = 0`` is what makes the parameters unique."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    assert sum(model.attack.values()) == pytest.approx(0.0, abs=1e-9)


def test_shifting_alpha_and_beta_together_leaves_expected_goals_unchanged(
    synthetic_dataset,
):
    """The invariance the constraint removes: lambda/mu ignore the shift."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    before = model.expected_goals("Team 00", "Team 05")

    shift = 0.37
    for team in model.attack:
        model.attack[team] += shift
        model.defence[team] += shift

    assert model.expected_goals("Team 00", "Team 05") == pytest.approx(before)


# --------------------------------------------------------------------------- #
# score matrix
# --------------------------------------------------------------------------- #


def test_score_matrix_is_normalised_over_the_full_0_0_to_10_10_grid(synthetic_dataset):
    """The published matrix is 11x11 and sums to 1."""
    _, matches = synthetic_dataset
    model = DixonColesModel(max_goals=MAX_GOALS).fit(matches)
    matrix = model.matrix_for("Team 00", "Team 05")

    assert matrix.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
    assert float(matrix.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(matrix.min()) >= 0.0


def test_summarise_returns_wdl_and_expected_goals_that_agree_with_the_matrix(
    synthetic_dataset,
):
    """The standalone summation is the single source of W/D/L and expected goals."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    matrix = model.matrix_for("Team 00", "Team 05")
    summary = model.summarise("Team 00", "Team 05")

    assert summary.p_home + summary.p_draw + summary.p_away == pytest.approx(1.0)
    assert summary.p_home == pytest.approx(float(np.tril(matrix, -1).sum()))
    assert summary.p_draw == pytest.approx(float(np.trace(matrix)))
    assert summary.p_away == pytest.approx(float(np.triu(matrix, 1).sum()))
    goals = np.arange(matrix.shape[0], dtype=np.float64)
    assert summary.exp_home_goals == pytest.approx(
        float((matrix.sum(axis=1) * goals).sum())
    )
    assert summary.exp_away_goals == pytest.approx(
        float((matrix.sum(axis=0) * goals).sum())
    )


def test_tau_correction_only_touches_the_four_low_cells():
    """Every other scoreline keeps the plain Poisson product."""
    lam, mu, rho = 1.4, 1.1, -0.05
    assert tau(0, 0, lam, mu, rho) == pytest.approx(1.0 - lam * mu * rho)
    assert tau(1, 1, lam, mu, rho) == pytest.approx(1.0 - rho)
    assert tau(1, 0, lam, mu, rho) == pytest.approx(1.0 + mu * rho)
    assert tau(0, 1, lam, mu, rho) == pytest.approx(1.0 + lam * rho)
    for x, y in ((2, 0), (0, 2), (2, 2), (3, 1), (5, 5)):
        assert tau(x, y, lam, mu, rho) == 1.0



# --------------------------------------------------------------------------- #
# time decay
# --------------------------------------------------------------------------- #


def test_xi_and_half_life_are_reciprocal():
    """xi is primary; the half-life is a convenience with an exact conversion."""
    assert half_life_to_xi(540.0) == pytest.approx(0.0012836, abs=1e-7)
    assert xi_to_half_life(0.003) == pytest.approx(231.05, abs=0.01)
    assert half_life_to_xi(xi_to_half_life(0.004)) == pytest.approx(0.004)
    assert DixonColesModel(xi=0.003).half_life_days == pytest.approx(231.05, abs=0.01)
    assert DixonColesModel(half_life_days=540.0).xi == pytest.approx(XI_540)


def test_passing_both_xi_and_half_life_is_rejected():
    """Two ways to say the same thing must not be silently ambiguous."""
    with pytest.raises(ValueError, match="either xi or half_life_days"):
        DixonColesModel(xi=0.003, half_life_days=540.0)


def test_decay_weight_follows_the_exponential_formula():
    """``w = exp(-xi * days_old)``, and exactly 1 at the reference match."""
    model = DixonColesModel(xi=0.003)
    reference = datetime(2025, 5, 1, tzinfo=timezone.utc)
    older = datetime(2025, 4, 1, tzinfo=timezone.utc)
    assert model.decay_weight(reference, reference) == 1.0
    assert model.decay_weight(older, reference) == pytest.approx(np.exp(-0.003 * 30.0))


def test_zero_xi_disables_decay_entirely():
    """xi = 0 is the "no decay" switch, kept for the undecayed comparison."""
    model = DixonColesModel(xi=0.0)
    assert model.half_life_days is None
    assert (
        model.decay_weight(
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        == 1.0
    )


# --------------------------------------------------------------------------- #
# artifact convention guard
# --------------------------------------------------------------------------- #


def test_artifact_round_trip_preserves_the_convention(synthetic_dataset):
    """A new artifact carries its convention marker and reloads exactly."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    artifact = model.to_artifact()
    assert artifact["defence_convention"] == "beta_defence_strength"

    restored = DixonColesModel.from_artifact(artifact)
    assert restored.xi == pytest.approx(model.xi)
    assert restored.defence == pytest.approx(model.defence)


def test_artifact_without_the_convention_marker_is_rejected(synthetic_dataset):
    """An old-convention artifact must fail loudly, never invert beta silently."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)

    stale = model.to_artifact()
    stale.pop("defence_convention")
    with pytest.raises(ValueError, match="incompatible defence convention"):
        DixonColesModel.from_artifact(stale)

    wrong = model.to_artifact()
    wrong["defence_convention"] = "defence_quality"
    with pytest.raises(ValueError, match="incompatible defence convention"):
        DixonColesModel.from_artifact(wrong)



# --------------------------------------------------------------------------- #
# real-data sanity: a strong side must out-rate a relegated one (build plan 2.3)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sp1_season():
    """One completed La Liga season as (labels, matches), or skip without a DB."""
    from sqlalchemy import text

    from app.database import SessionLocal
    from app.ml.dataset import load_labeled_matches, load_team_names
    from app.models import Competition

    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        session.close()
        pytest.skip(f"database unavailable: {exc}")
    try:
        competition = (
            session.query(Competition).filter(Competition.code == "SP1").one_or_none()
        )
        if competition is None:
            pytest.skip("SP1 not ingested")
        labels, matches = load_labeled_matches(
            session, competition.id, load_team_names(session)
        )
        season = "2024/25"
        rows = [m for label, m in zip(labels, matches, strict=True) if label == season]
        if len(rows) < 300:
            pytest.skip(f"{season} has only {len(rows)} finished matches")
        return rows
    finally:
        session.close()


@pytest.mark.slow
def test_the_champions_out_rate_the_relegated_side_in_the_same_season(sp1_season):
    """Fit a real La Liga season; the best attack must belong to the best team.

    The comparison is derived from the season's own results rather than from
    hard-coded club names, so it cannot break on a team-alias change.
    """
    from collections import defaultdict

    points: dict[object, int] = defaultdict(int)
    for match in sp1_season:
        if match.outcome == 0:
            points[match.home] += 3
        elif match.outcome == 2:
            points[match.away] += 3
        else:
            points[match.home] += 1
            points[match.away] += 1
    assert len(points) == 20

    champion = max(points, key=lambda team: (points[team], str(team)))
    relegated = min(points, key=lambda team: (points[team], str(team)))

    model = DixonColesModel().fit(sp1_season)
    assert model.diagnostics is not None and model.diagnostics.converged

    print(
        f"\n  alpha[{champion}]={model.attack[champion]:+.4f} ({points[champion]} pts)  "
        f"alpha[{relegated}]={model.attack[relegated]:+.4f} ({points[relegated]} pts)"
    )
    print(f"  rho={model.rho:+.5f}  home_adv={model.home_adv:+.4f}")
    assert model.attack[champion] > model.attack[relegated]
    # Under lambda = exp(alpha - beta + gamma) a *higher* beta means a stronger
    # defence, so the champion must out-rate the relegated side here too.
    assert model.defence[champion] > model.defence[relegated]

