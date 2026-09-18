"""Dixon-Coles Poisson scoreline model (Dixon & Coles, 1997) — the primary model.

Model
-----
For a fixture with home team ``h`` and away team ``a``:

    lambda_home = exp(alpha[h] - beta[a] + gamma)
    lambda_away = exp(alpha[a] - beta[h])

    P(x, y) = tau(x, y) * Poisson(x; lambda_home) * Poisson(y; lambda_away)

``alpha`` is attacking strength (higher = scores more), ``beta`` is defensive
*strength* (higher = concedes fewer) and ``gamma`` is the global home advantage.

Note the **minus** on ``beta``. A minus sign means a larger ``beta`` lowers the
opponent's scoring rate, so ``beta`` measures how hard a side is to score
against. Dixon & Coles' own paper writes ``lambda = alpha_i * beta_j * gamma``
with ``beta`` as a *leakiness* term under a **plus** sign; that is the same model
with ``beta -> -beta``. The two are not interchangeable in storage: a model
fitted under one convention and loaded under the other has every club's
defensive rating inverted, silently. The artifact carries a convention marker to
make that failure loud (see :meth:`DixonColesModel.from_artifact`).

Low-score correction
--------------------
Independent Poisson under-predicts 0-0, 1-1, 1-0 and 0-1. The original
correction multiplies exactly those four cells:

    tau(0, 0) = 1 - lambda*mu*rho      tau(1, 0) = 1 + mu*rho
    tau(0, 1) = 1 + lambda*rho         tau(1, 1) = 1 - rho      (1 elsewhere)

Fitting
-------
Weighted maximum likelihood by L-BFGS-B with an analytic gradient. Each match
carries an exponential time-decay weight

    w = exp(-xi * days_old)

measured against the most recent training match, which always gets weight 1.
``xi`` (per day) is the primary decay parameter. ``half_life_days`` is accepted
as a convenience and converted with ``xi = ln(2) / half_life_days``; the two are
exact reciprocals, so ``half_life_days=540`` is ``xi=0.00128`` and the plan's
starting point of ``xi=0.003`` is a half-life of ~231 days.

The fit is deterministic: the optimiser starts from an all-zero vector and
L-BFGS-B is a deterministic algorithm, so two runs on identical data produce
bit-identical parameters. No random seed is consumed. ``seed`` is recorded in
the artifact for provenance only.

Identifiability — the ``sum(alpha) = 0`` constraint
--------------------------------------------------
Every lambda is invariant under

    alpha -> alpha - c,   beta -> beta - c,   gamma -> gamma

for any constant ``c``, because the shift cancels inside each exponent. The
likelihood alone therefore pins down no unique solution — the optimiser is free
to slide all parameters along that ridge. The standard fix, applied here, is the
constraint

    sum(alpha) = 0

so that ``alpha = 0`` means a league-average attack. It is imposed *after* the
optimiser returns, by subtracting the mean from every ``alpha`` and the same
constant from every ``beta``: every lambda/mu is unchanged, but the parameters
become unique and comparable between fits. ``alpha = beta = 0`` together is then
a league-average side — also the fallback used for a club the model has never
seen.

Convergence
-----------
``scipy.optimize.minimize``'s ``success`` flag is asserted. A fit that fails to
converge raises :class:`DixonColesConvergenceError` rather than returning
plausible-looking numbers, because a non-converged attack/defence vector is
indistinguishable from a converged one at the prediction layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from app.ml.data import (
    FloatArray,
    MatchInput,
    MatrixSummary,
    ScorelinePrediction,
    TeamKey,
    summarise_score_matrix,
)

_LOG2 = math.log(2.0)
_TAU_CLIP = 1e-10
_DEFENCE_CONVENTION = "beta_defence_strength"
"""Artifact marker for the ``lambda = exp(alpha - beta + gamma)`` convention.

Under a **minus** sign a higher ``beta`` *reduces* the opponent's scoring rate,
so ``beta`` is defensive **strength** (higher = concedes fewer), not leakiness.
Dixon & Coles' own paper writes ``lambda = alpha_i * beta_j * gamma`` with
``beta`` as a leakiness term under a **plus** sign; the two are the same model
with ``beta -> -beta``. This marker exists because mixing the two conventions
inverts every club's defensive rating without raising an error.
"""

DEFAULT_HALF_LIFE_DAYS = 540.0
"""Current production default, i.e. ``xi = 0.00128`` per day."""


class DixonColesConvergenceError(RuntimeError):
    """Raised when the maximum-likelihood fit fails to converge."""


def half_life_to_xi(half_life_days: float) -> float:
    """Convert a decay half-life in days to the per-day decay rate ``xi``."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    return _LOG2 / float(half_life_days)


def xi_to_half_life(xi: float) -> float:
    """Convert a per-day decay rate ``xi`` to a half-life in days."""
    if xi <= 0:
        raise ValueError("xi must be positive")
    return _LOG2 / float(xi)


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score dependency factor (only the four smallest scorelines)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _log_factorials(max_goals: int) -> FloatArray:
    """``[ln(0!), ln(1!), ..., ln(max_goals!)]``."""
    steps = np.arange(1, max_goals + 1, dtype=np.float64)
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(np.log(steps))))


@dataclass(frozen=True)
class FitDiagnostics:
    """What the optimiser actually did, kept for reporting and provenance."""

    converged: bool
    n_iterations: int
    status: int
    message: str
    negative_log_likelihood: float
    gradient_norm: float
    max_abs_gradient: float


class DixonColesModel:
    """Fit and predict a full scoreline distribution for one league.

    Parameters
    ----------
    xi
        Exponential time-decay rate per day, used as ``w = exp(-xi * days_old)``.
        ``xi = 0`` disables decay. Preferred over ``half_life_days``.
    half_life_days
        Convenience alternative to ``xi``: ``xi = ln(2) / half_life_days``.
        Passing both is an error. Defaults to 540 days when neither is given,
        which preserves the production default (``xi = 0.00128``).
    """

    def __init__(
        self,
        xi: float | None = None,
        *,
        half_life_days: float | None = None,
        max_goals: int = 10,
        rho_bound: float = 0.10,
        l2: float = 1e-6,
        seed: int = 42,
    ) -> None:
        if max_goals < 1:
            raise ValueError("max_goals must be at least 1")
        if xi is not None and half_life_days is not None:
            raise ValueError("pass either xi or half_life_days, not both")
        if xi is None:
            if half_life_days is None:
                half_life_days = DEFAULT_HALF_LIFE_DAYS
            xi = 0.0 if half_life_days <= 0 else half_life_to_xi(half_life_days)
        if xi < 0:
            raise ValueError("xi must be >= 0")

        self.xi = float(xi)
        self.half_life_days = xi_to_half_life(xi) if xi > 0 else None
        self.max_goals = max_goals
        self.rho_bound = rho_bound
        self.l2 = l2
        self.seed = seed

        self.attack: dict[TeamKey, float] = {}
        """alpha: attacking strength, mean-centred so ``sum(alpha) == 0``."""

        self.defence: dict[TeamKey, float] = {}
        """beta: defensive *strength* — a higher value concedes fewer goals."""

        self.home_adv = 0.0
        self.rho = 0.0
        self.log_likelihood = 0.0
        self.n_train_matches = 0
        self.train_start: datetime | None = None
        self.train_end: datetime | None = None
        self.fitted = False
        self.diagnostics: FitDiagnostics | None = None

    # -- fitting ----------------------------------------------------------

    def decay_weight(self, kickoff: datetime, reference: datetime) -> float:
        """``w = exp(-xi * days_old)``; 1.0 for the reference match itself."""
        if self.xi <= 0:
            return 1.0
        age = (reference - kickoff).total_seconds() / 86400.0
        return math.exp(-self.xi * age)

    def fit(self, matches: Sequence[MatchInput]) -> DixonColesModel:
        """Fit alpha, beta, gamma and rho by weighted maximum likelihood.

        Raises :class:`DixonColesConvergenceError` if L-BFGS-B does not report
        success.
        """
        n_matches = len(matches)
        if n_matches == 0:
            raise ValueError("cannot fit without training matches")

        kickoffs = [m.kickoff for m in matches]
        teams = sorted({m.home for m in matches} | {m.away for m in matches}, key=str)
        index: dict[TeamKey, int] = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        reference = max(kickoffs)
        ages = np.array(
            [(reference - k).total_seconds() / 86400.0 for k in kickoffs],
            dtype=np.float64,
        )
        weights = np.exp(-self.xi * ages) if self.xi > 0 else np.ones(n_matches)

        home_idx = np.array([index[m.home] for m in matches], dtype=np.int64)
        away_idx = np.array([index[m.away] for m in matches], dtype=np.int64)
        home_goals = np.array([float(m.home_goals) for m in matches], dtype=np.float64)
        away_goals = np.array([float(m.away_goals) for m in matches], dtype=np.float64)

        is_00 = (home_goals == 0.0) & (away_goals == 0.0)
        is_01 = (home_goals == 0.0) & (away_goals == 1.0)
        is_10 = (home_goals == 1.0) & (away_goals == 0.0)
        is_11 = (home_goals == 1.0) & (away_goals == 1.0)
        log_clip = math.log(_TAU_CLIP)
        l2 = float(self.l2)


        def objective(pars: FloatArray) -> tuple[float, FloatArray]:
            """Negative weighted log-likelihood plus its analytic gradient.

            Under this convention ``lambda_home = exp(alpha[h] - beta[a] + gamma)``,
            so ``d(lambda)/d(beta) = -lambda`` and the beta gradient carries the
            opposite sign to the alpha gradient. That is the whole difference
            from a "defensive quality" parameterisation.
            """
            attack = pars[:n_teams]
            defence = pars[n_teams : 2 * n_teams]
            home_adv = pars[2 * n_teams]
            rho = pars[2 * n_teams + 1]

            lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] - defence[home_idx])

            log_tau = np.zeros(n_matches, dtype=np.float64)
            d_rho = np.zeros(n_matches, dtype=np.float64)
            d_lam = np.zeros(n_matches, dtype=np.float64)
            d_mu = np.zeros(n_matches, dtype=np.float64)

            # tau depends on lambda and mu as well as rho, so the score
            # gradients need d(log tau)/d(lambda) and d(log tau)/d(mu). Omitting
            # them leaves the analytic gradient inconsistent with the objective,
            # which L-BFGS-B punishes with a failed line search.

            # (0, 0): tau = 1 - lam*mu*rho  ->  d log(tau)/d rho = -lam*mu/tau
            t00 = 1.0 - lam * mu * rho
            keep = is_00 & (t00 > _TAU_CLIP)
            log_tau[keep] = np.log(t00[keep])
            d_rho[keep] = -(lam * mu)[keep] / t00[keep]
            d_lam[keep] = -(mu * rho)[keep] / t00[keep]
            d_mu[keep] = -(lam * rho)[keep] / t00[keep]
            log_tau[is_00 & ~keep] = log_clip

            # (0, 1): tau = 1 + lam*rho
            t01 = 1.0 + lam * rho
            keep = is_01 & (t01 > _TAU_CLIP)
            log_tau[keep] = np.log(t01[keep])
            d_rho[keep] = lam[keep] / t01[keep]
            d_lam[keep] = rho / t01[keep]
            log_tau[is_01 & ~keep] = log_clip

            # (1, 0): tau = 1 + mu*rho
            t10 = 1.0 + mu * rho
            keep = is_10 & (t10 > _TAU_CLIP)
            log_tau[keep] = np.log(t10[keep])
            d_rho[keep] = mu[keep] / t10[keep]
            d_mu[keep] = rho / t10[keep]
            log_tau[is_10 & ~keep] = log_clip

            # (1, 1): tau = 1 - rho  (no lambda/mu dependence)
            t11 = 1.0 - rho
            keep = is_11 & (t11 > _TAU_CLIP)
            log_tau[keep] = math.log(t11)
            d_rho[keep] = -1.0 / t11
            log_tau[is_11 & ~keep] = log_clip

            terms = log_tau + home_goals * np.log(lam) - lam
            terms = terms + away_goals * np.log(mu) - mu
            log_lik = float(np.sum(weights * terms))
            log_lik -= l2 * float(np.sum(attack**2) + np.sum(defence**2))

            # Poisson residual plus the tau contribution, per match.
            resid_home = (home_goals - lam) + lam * d_lam
            resid_away = (away_goals - mu) + mu * d_mu

            grad_attack = np.bincount(
                home_idx, weights=weights * resid_home, minlength=n_teams
            )
            grad_attack = grad_attack + np.bincount(
                away_idx, weights=weights * resid_away, minlength=n_teams
            )
            grad_attack = grad_attack - 2.0 * l2 * attack

            # beta enters each lambda with a minus, so its residual is negated.
            grad_defence = np.bincount(
                away_idx, weights=-weights * resid_home, minlength=n_teams
            )
            grad_defence = grad_defence + np.bincount(
                home_idx, weights=-weights * resid_away, minlength=n_teams
            )
            grad_defence = grad_defence - 2.0 * l2 * defence

            grad_home = float(np.sum(weights * resid_home))
            grad_rho = float(np.sum(weights * d_rho))
            grad = np.concatenate(
                (
                    grad_attack,
                    grad_defence,
                    np.array([grad_home, grad_rho], dtype=np.float64),
                )
            )
            return -log_lik, -np.asarray(grad, dtype=np.float64)


        # Deterministic start: all-zero parameters, so repeated fits agree bit
        # for bit. L-BFGS-B itself is deterministic; no RNG is involved.
        start = np.zeros(2 * n_teams + 2, dtype=np.float64)
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * (2 * n_teams)
        bounds.append((None, None))
        bounds.append((-abs(self.rho_bound), abs(self.rho_bound)))
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9},
        )
        params = np.asarray(result.x, dtype=np.float64)
        gradient = np.asarray(objective(params)[1], dtype=np.float64)

        diagnostics = FitDiagnostics(
            converged=bool(result.success),
            n_iterations=int(getattr(result, "nit", 0)),
            status=int(getattr(result, "status", -1)),
            message=str(getattr(result, "message", "")),
            negative_log_likelihood=float(result.fun),
            gradient_norm=float(np.linalg.norm(gradient)),
            max_abs_gradient=float(np.max(np.abs(gradient))) if gradient.size else 0.0,
        )
        self.diagnostics = diagnostics

        # Fail loudly: a non-converged alpha/beta vector is indistinguishable
        # from a converged one at the prediction layer.
        if not diagnostics.converged:
            raise DixonColesConvergenceError(
                "Dixon-Coles fit did not converge: "
                f"status={diagnostics.status} message={diagnostics.message!r} "
                f"iterations={diagnostics.n_iterations} "
                f"max|grad|={diagnostics.max_abs_gradient:.3e}"
            )
        if not bool(np.all(np.isfinite(params))):
            raise DixonColesConvergenceError(
                "Dixon-Coles optimisation produced non-finite parameters"
            )

        raw_attack = params[:n_teams]
        raw_defence = params[n_teams : 2 * n_teams]
        # Identifiability: sum(alpha) = 0. Shifting alpha and beta by the same
        # constant leaves every lambda/mu untouched, so this is free.
        centre = float(np.mean(raw_attack))
        attack = raw_attack - centre
        defence = raw_defence - centre

        self.attack = {team: float(attack[index[team]]) for team in teams}
        self.defence = {team: float(defence[index[team]]) for team in teams}
        self.home_adv = float(params[2 * n_teams])
        self.rho = float(params[2 * n_teams + 1])
        self.n_train_matches = n_matches
        self.train_start = min(kickoffs)
        self.train_end = reference
        self.fitted = True
        self.log_likelihood = self._weighted_log_likelihood(matches)
        return self


    def _weighted_log_likelihood(self, matches: Sequence[MatchInput]) -> float:
        """Log-likelihood of the fitted model, Poisson constants included."""
        if not matches:
            return 0.0
        reference = max(m.kickoff for m in matches)
        total = 0.0
        for match in matches:
            weight = self.decay_weight(match.kickoff, reference)
            lam, mu = self.expected_goals(match.home, match.away)
            x = match.home_goals
            y = match.away_goals
            factor = max(tau(x, y, lam, mu, self.rho), _TAU_CLIP)
            total += weight * (
                math.log(factor)
                + x * math.log(lam)
                - lam
                - math.lgamma(x + 1)
                + y * math.log(mu)
                - mu
                - math.lgamma(y + 1)
            )
        return total

    # -- prediction -------------------------------------------------------

    def expected_goals(self, home: TeamKey, away: TeamKey) -> tuple[float, float]:
        """``(lambda_home, lambda_away)``; unseen clubs fall back to average.

        ``lambda_home = exp(alpha[h] - beta[a] + gamma)`` and
        ``lambda_away = exp(alpha[a] - beta[h])``.
        """
        attack_home = self.attack.get(home, 0.0)
        attack_away = self.attack.get(away, 0.0)
        strength_home = self.defence.get(home, 0.0)
        strength_away = self.defence.get(away, 0.0)
        lam = math.exp(attack_home - strength_away + self.home_adv)
        mu = math.exp(attack_away - strength_home)
        return lam, mu

    def score_matrix(self, lam: float, mu: float) -> FloatArray:
        """Scoreline distribution on a (max_goals+1)^2 grid (rows = home goals).

        With the default ``max_goals=10`` this is the full 0-0 .. 10-10 grid.
        The tau correction is applied to the four low cells and the matrix is
        renormalised, so it always sums to 1.
        """
        goals = np.arange(self.max_goals + 1, dtype=np.float64)
        log_fact = _log_factorials(self.max_goals)
        p_home = np.exp(goals * math.log(lam) - lam - log_fact)
        p_away = np.exp(goals * math.log(mu) - mu - log_fact)
        matrix = np.outer(p_home, p_away)
        if self.rho != 0.0:
            matrix[0, 0] *= max(1.0 - lam * mu * self.rho, 0.0)
            matrix[0, 1] *= max(1.0 + lam * self.rho, 0.0)
            matrix[1, 0] *= max(1.0 + mu * self.rho, 0.0)
            matrix[1, 1] *= max(1.0 - self.rho, 0.0)
        matrix = np.maximum(matrix, 0.0)
        total = float(matrix.sum())
        if total <= 0.0:
            raise ValueError("score matrix collapsed to zero mass")
        return np.asarray(matrix / total, dtype=np.float64)

    def matrix_for(self, home: TeamKey, away: TeamKey) -> FloatArray:
        """The score matrix for a fixture, from its expected goals."""
        lam, mu = self.expected_goals(home, away)
        return self.score_matrix(lam, mu)

    def summarise(self, home: TeamKey, away: TeamKey) -> MatrixSummary:
        """W/D/L probabilities and expected goals for a fixture."""
        return summarise_score_matrix(self.matrix_for(home, away))

    def predict(self, home: TeamKey, away: TeamKey) -> ScorelinePrediction:
        """Outcome probabilities plus expected goals for a fixture."""
        if not self.fitted:
            raise RuntimeError("fit() must be called before predict()")
        return ScorelinePrediction.from_matrix(self.matrix_for(home, away))

    def predict_many(
        self, fixtures: Sequence[tuple[TeamKey, TeamKey]]
    ) -> list[ScorelinePrediction]:
        """Predict a batch of fixtures with one fitted model."""
        return [self.predict(home, away) for home, away in fixtures]


    # -- persistence -------------------------------------------------------

    def hyperparameters(self) -> dict[str, Any]:
        """Constructor settings, mirrored into ``model_versions.hyperparameters``."""
        return {
            "xi": self.xi,
            "half_life_days": self.half_life_days,
            "max_goals": self.max_goals,
            "rho_bound": self.rho_bound,
            "l2": self.l2,
            "seed": self.seed,
            "defence_convention": _DEFENCE_CONVENTION,
            "time_decay": bool(self.xi > 0),
        }

    def diagnostics_payload(self) -> dict[str, Any]:
        """Optimiser diagnostics for the artifact and for review reports."""
        if self.diagnostics is None:
            return {}
        return {
            "converged": self.diagnostics.converged,
            "n_iterations": self.diagnostics.n_iterations,
            "status": self.diagnostics.status,
            "message": self.diagnostics.message,
            "negative_log_likelihood": self.diagnostics.negative_log_likelihood,
            "gradient_norm": self.diagnostics.gradient_norm,
            "max_abs_gradient": self.diagnostics.max_abs_gradient,
        }

    def to_artifact(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (team keys canonicalised to strings).

        Carries ``defence_convention`` so a stale artifact cannot be loaded by
        this code and silently invert every club's defensive strength.
        """
        return {
            "model": "dixon_coles",
            "defence_convention": _DEFENCE_CONVENTION,
            "hyperparameters": self.hyperparameters(),
            "home_adv": self.home_adv,
            "rho": self.rho,
            "attack": {str(k): v for k, v in self.attack.items()},
            "defence": {str(k): v for k, v in self.defence.items()},
            "n_train_matches": self.n_train_matches,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "log_likelihood": self.log_likelihood,
            "diagnostics": self.diagnostics_payload(),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> DixonColesModel:
        """Rebuild a fitted model from :meth:`to_artifact` output.

        Raises :class:`ValueError` when the artifact predates the
        ``beta = leakiness`` convention, because loading it would invert every
        ``beta`` and produce confident nonsense.
        """
        convention = artifact.get("defence_convention")
        if convention != _DEFENCE_CONVENTION:
            raise ValueError(
                "artifact was fitted under an incompatible defence convention "
                f"(found {convention!r}, expected {_DEFENCE_CONVENTION!r}); "
                "refit the model instead of loading it"
            )
        hyper = artifact.get("hyperparameters") or {}
        model = cls(
            xi=float(hyper["xi"]) if hyper.get("xi") is not None else None,
            half_life_days=(
                float(hyper["half_life_days"])
                if hyper.get("xi") is None and hyper.get("half_life_days")
                else None
            ),
            max_goals=int(hyper.get("max_goals", 10)),
            rho_bound=float(hyper.get("rho_bound", 0.10)),
            l2=float(hyper.get("l2", 1e-6)),
            seed=int(hyper.get("seed", 42)),
        )
        model.attack = {str(k): float(v) for k, v in artifact["attack"].items()}
        model.defence = {str(k): float(v) for k, v in artifact["defence"].items()}
        model.home_adv = float(artifact["home_adv"])
        model.rho = float(artifact["rho"])
        model.n_train_matches = int(artifact.get("n_train_matches", 0))
        model.log_likelihood = float(artifact.get("log_likelihood", 0.0))
        start = artifact.get("train_start")
        end = artifact.get("train_end")
        model.train_start = datetime.fromisoformat(start) if start else None
        model.train_end = datetime.fromisoformat(end) if end else None
        model.fitted = True
        return model

