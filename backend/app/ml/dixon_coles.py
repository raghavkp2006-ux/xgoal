"""Dixon-Coles Poisson scoreline model (Dixon & Coles, 1997) — the primary model.

Model
-----
For a match with home team ``h`` and away team ``a``:

    lambda = exp(attack[h] + defence[a] + home_adv)
    mu     = exp(attack[a] + defence[h])

    P(x, y) = tau(x, y) * Poisson(x; lambda) * Poisson(y; mu)

with the low-score dependence correction of the original paper:

    tau(0, 0) = 1 - lambda*mu*rho      tau(1, 0) = 1 + mu*rho
    tau(0, 1) = 1 + lambda*rho         tau(1, 1) = 1 - rho      (1 elsewhere)

Fitting
-------
Weighted maximum likelihood with an analytic gradient (L-BFGS-B). Each match
carries an exponential time-decay weight
``w = exp(-ln(2) * days_old / half_life_days)`` measured against the most recent
training match (which always gets weight 1). ``rho`` is bounded to
``-rho_bound .. +rho_bound`` and every tau denominator is guarded during
optimisation so the objective stays finite. A tiny L2 penalty (``l2``) stops
rarely-seen teams from drifting to silly ratings.

After fitting, the attack vector is mean-centred and the defence vector shifted
by the opposite amount, which leaves every lambda/mu unchanged but makes
``attack = defence = 0`` a league-average team — also the fallback used for
teams the model has never seen.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from app.ml.data import FloatArray, MatchInput, ScorelinePrediction, TeamKey

_LOG2 = math.log(2.0)
_TAU_CLIP = 1e-10


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


class DixonColesModel:
    """Fit and predict a full scoreline distribution for one league."""

    def __init__(
        self,
        half_life_days: float | None = 540.0,
        max_goals: int = 10,
        rho_bound: float = 0.10,
        l2: float = 1e-6,
    ) -> None:
        if max_goals < 1:
            raise ValueError("max_goals must be at least 1")
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.rho_bound = rho_bound
        self.l2 = l2

        self.attack: dict[TeamKey, float] = {}
        self.defence: dict[TeamKey, float] = {}
        self.home_adv = 0.0
        self.rho = 0.0
        self.log_likelihood = 0.0
        self.n_train_matches = 0
        self.train_start: datetime | None = None
        self.train_end: datetime | None = None
        self.fitted = False

    # -- fitting ----------------------------------------------------------

    def fit(self, matches: Sequence[MatchInput]) -> DixonColesModel:
        """Fit attack/defence ratings, home advantage and rho by weighted MLE."""
        n_matches = len(matches)
        if n_matches == 0:
            raise ValueError("cannot fit without training matches")

        kickoffs = [m.kickoff for m in matches]
        teams = sorted({m.home for m in matches} | {m.away for m in matches}, key=str)
        index: dict[TeamKey, int] = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        reference = max(kickoffs)
        if not self.half_life_days or self.half_life_days <= 0:
            weights = np.ones(n_matches, dtype=np.float64)
        else:
            ages = np.array(
                [(reference - k).total_seconds() / 86400.0 for k in kickoffs],
                dtype=np.float64,
            )
            weights = np.exp(-_LOG2 * ages / float(self.half_life_days))

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
            """Negative weighted log-likelihood plus its analytic gradient."""
            attack = pars[:n_teams]
            defence = pars[n_teams : 2 * n_teams]
            home_adv = pars[2 * n_teams]
            rho = pars[2 * n_teams + 1]

            lam = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defence[home_idx])

            log_tau = np.zeros(n_matches, dtype=np.float64)
            d_rho = np.zeros(n_matches, dtype=np.float64)

            # (0, 0): tau = 1 - lam*mu*rho  ->  d log(tau)/d rho = -lam*mu/tau
            t00 = 1.0 - lam * mu * rho
            keep = is_00 & (t00 > _TAU_CLIP)
            log_tau[keep] = np.log(t00[keep])
            d_rho[keep] = -(lam * mu)[keep] / t00[keep]
            log_tau[is_00 & ~keep] = log_clip

            # (0, 1): tau = 1 + lam*rho
            t01 = 1.0 + lam * rho
            keep = is_01 & (t01 > _TAU_CLIP)
            log_tau[keep] = np.log(t01[keep])
            d_rho[keep] = lam[keep] / t01[keep]
            log_tau[is_01 & ~keep] = log_clip

            # (1, 0): tau = 1 + mu*rho
            t10 = 1.0 + mu * rho
            keep = is_10 & (t10 > _TAU_CLIP)
            log_tau[keep] = np.log(t10[keep])
            d_rho[keep] = mu[keep] / t10[keep]
            log_tau[is_10 & ~keep] = log_clip

            # (1, 1): tau = 1 - rho
            t11 = 1.0 - rho
            keep = is_11 & (t11 > _TAU_CLIP)
            log_tau[keep] = math.log(t11)
            d_rho[keep] = -1.0 / t11
            log_tau[is_11 & ~keep] = log_clip

            terms = log_tau + home_goals * np.log(lam) - lam
            terms = terms + away_goals * np.log(mu) - mu
            log_lik = float(np.sum(weights * terms))
            log_lik -= l2 * float(np.sum(attack**2) + np.sum(defence**2))

            grad_attack = np.bincount(
                home_idx, weights=weights * (home_goals - lam), minlength=n_teams
            )
            grad_attack = grad_attack + np.bincount(
                away_idx, weights=weights * (away_goals - mu), minlength=n_teams
            )
            grad_attack = grad_attack - 2.0 * l2 * attack

            grad_defence = np.bincount(
                away_idx, weights=weights * (home_goals - lam), minlength=n_teams
            )
            grad_defence = grad_defence + np.bincount(
                home_idx, weights=weights * (away_goals - mu), minlength=n_teams
            )
            grad_defence = grad_defence - 2.0 * l2 * defence

            grad_home = float(np.sum(weights * (home_goals - lam)))
            grad_rho = float(np.sum(weights * d_rho))
            grad = np.concatenate(
                (
                    grad_attack,
                    grad_defence,
                    np.array([grad_home, grad_rho], dtype=np.float64),
                )
            )
            return -log_lik, -np.asarray(grad, dtype=np.float64)

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
        if not bool(np.all(np.isfinite(params))):
            raise RuntimeError("Dixon-Coles optimisation produced non-finite parameters")

        raw_attack = params[:n_teams]
        raw_defence = params[n_teams : 2 * n_teams]
        centre = float(np.mean(raw_attack))
        attack = raw_attack - centre
        defence = raw_defence + centre
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
            weight = self._decay_weight(match.kickoff, reference)
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

    def _decay_weight(self, kickoff: datetime, reference: datetime) -> float:
        """Exponential time-decay weight relative to ``reference``."""
        if not self.half_life_days or self.half_life_days <= 0:
            return 1.0
        age = (reference - kickoff).total_seconds() / 86400.0
        return math.exp(-_LOG2 * age / float(self.half_life_days))

    # -- prediction -------------------------------------------------------

    def expected_goals(self, home: TeamKey, away: TeamKey) -> tuple[float, float]:
        """(lambda, mu) for a fixture; unseen teams fall back to league average."""
        attack_home = self.attack.get(home, 0.0)
        attack_away = self.attack.get(away, 0.0)
        defence_home = self.defence.get(home, 0.0)
        defence_away = self.defence.get(away, 0.0)
        lam = math.exp(attack_home + defence_away + self.home_adv)
        mu = math.exp(attack_away + defence_home)
        return lam, mu

    def score_matrix(self, lam: float, mu: float) -> FloatArray:
        """Scoreline distribution on a (max_goals+1)^2 grid (rows = home goals)."""
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

    def predict(self, home: TeamKey, away: TeamKey) -> ScorelinePrediction:
        """Outcome probabilities plus expected goals for a fixture."""
        if not self.fitted:
            raise RuntimeError("fit() must be called before predict()")
        lam, mu = self.expected_goals(home, away)
        return ScorelinePrediction.from_matrix(self.score_matrix(lam, mu))

    def predict_many(
        self, fixtures: Sequence[tuple[TeamKey, TeamKey]]
    ) -> list[ScorelinePrediction]:
        """Predict a batch of fixtures with one fitted model."""
        return [self.predict(home, away) for home, away in fixtures]

    # -- persistence helpers ----------------------------------------------

    def hyperparameters(self) -> dict[str, Any]:
        """Constructor settings, mirrored into ``model_versions.hyperparameters``."""
        return {
            "half_life_days": self.half_life_days,
            "max_goals": self.max_goals,
            "rho_bound": self.rho_bound,
            "l2": self.l2,
            "time_decay": bool(self.half_life_days and self.half_life_days > 0),
        }

    def to_artifact(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (team keys canonicalised to strings)."""
        return {
            "model": "dixon_coles",
            "hyperparameters": self.hyperparameters(),
            "home_adv": self.home_adv,
            "rho": self.rho,
            "attack": {str(k): v for k, v in self.attack.items()},
            "defence": {str(k): v for k, v in self.defence.items()},
            "n_train_matches": self.n_train_matches,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "log_likelihood": self.log_likelihood,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> DixonColesModel:
        """Rebuild a fitted model from :meth:`to_artifact` output."""
        hyper = artifact.get("hyperparameters") or {}
        model = cls(
            half_life_days=hyper.get("half_life_days"),
            max_goals=int(hyper.get("max_goals", 10)),
            rho_bound=float(hyper.get("rho_bound", 0.10)),
            l2=float(hyper.get("l2", 1e-6)),
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
