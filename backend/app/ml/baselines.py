"""Baseline forecasters used for the walk-forward comparison.

* :class:`UniformBaseline` — flat 1/3 for every outcome
* :class:`BaseRateBaseline` — historical H/D/A frequencies of the training set
* :class:`EloBaseline` — Elo ratings with home advantage, draws from the observed rate
* :func:`devig_probs` — de-vigged closing odds already stored on the match row
"""

from __future__ import annotations

from typing import Sequence

from app.ml.data import MatchInput, Probs, TeamKey

DEFAULT_RATING = 1500.0
_DRAW_FALLBACK = 0.25


def _chronological(matches: Sequence[MatchInput]) -> list[MatchInput]:
    """Stable chronological ordering used by every sequential baseline."""
    return sorted(matches, key=lambda m: m.kickoff)


class UniformBaseline:
    """Every outcome equally likely — the sanity floor for any real model."""

    def fit(self, matches: Sequence[MatchInput]) -> UniformBaseline:
        """No-op fit, present so all models share one interface."""
        return self

    def predict(self, home: TeamKey, away: TeamKey) -> Probs:
        """Flat 1/3 across home, draw and away."""
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class BaseRateBaseline:
    """Constant probabilities equal to the training-set H/D/A frequencies."""

    def __init__(self) -> None:
        self.probs: Probs = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self.n_train_matches = 0

    def fit(self, matches: Sequence[MatchInput]) -> BaseRateBaseline:
        """Count outcomes and turn them into probabilities."""
        counts = [0, 0, 0]
        for match in matches:
            counts[match.outcome] += 1
        total = sum(counts)
        if total:
            self.probs = (counts[0] / total, counts[1] / total, counts[2] / total)
            self.n_train_matches = total
        return self

    def predict(self, home: TeamKey, away: TeamKey) -> Probs:
        """The same base rate for every fixture."""
        return self.probs


class EloBaseline:
    """Elo ratings with home advantage; draws calibrated to the observed rate.

    Elo has no native draw outcome, so the empirical draw frequency of the
    training window is held fixed and the remaining mass is split between the
    two teams by the usual logistic rating difference.
    """

    def __init__(self, k: float = 20.0, home_adv: float = 60.0) -> None:
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[TeamKey, float] = {}
        self.draw_rate = _DRAW_FALLBACK
        self.n_train_matches = 0

    def rating(self, team: TeamKey) -> float:
        """Current rating of a team (default for sides never seen)."""
        return self.ratings.get(team, DEFAULT_RATING)

    def fit(self, matches: Sequence[MatchInput]) -> EloBaseline:
        """Reset and replay the given matches chronologically."""
        self.ratings = {}
        ordered = _chronological(matches)
        draws = sum(1 for m in ordered if m.outcome == 1)
        self.draw_rate = (draws / len(ordered)) if ordered else _DRAW_FALLBACK
        self.n_train_matches = len(ordered)
        for match in ordered:
            self.update(match)
        return self

    def update(self, match: MatchInput) -> None:
        """Fold one completed match into the ratings (margin-of-victory weighted)."""
        home_rating = self.rating(match.home)
        away_rating = self.rating(match.away)
        spread = home_rating + self.home_adv - away_rating
        expected_home = 1.0 / (1.0 + 10.0 ** (-spread / 400.0))
        actual = {0: 1.0, 1: 0.5, 2: 0.0}[match.outcome]
        margin = abs(match.goal_diff)
        multiplier = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11.0 + margin) / 8.0)
        delta = self.k * multiplier * (actual - expected_home)
        self.ratings[match.home] = home_rating + delta
        self.ratings[match.away] = away_rating - delta

    def predictability(self, home: TeamKey, away: TeamKey) -> float:
        """Home win probability ignoring the draw, from the current ratings."""
        spread = self.rating(home) + self.home_adv - self.rating(away)
        return float(1.0 / (1.0 + 10.0 ** (-spread / 400.0)))

    def predict(self, home: TeamKey, away: TeamKey) -> Probs:
        """Split ``1 - draw_rate`` between home and away by the Elo expectation."""
        decisive = self.predictability(home, away)
        return (
            (1.0 - self.draw_rate) * decisive,
            self.draw_rate,
            (1.0 - self.draw_rate) * (1.0 - decisive),
        )

    def predict_backtest(self, matches: Sequence[MatchInput]) -> list[Probs]:
        """Point-in-time predictions for a run of matches.

        Each fixture is predicted from the ratings *before* it was played and
        the ratings are then updated, exactly as a live system would.
        """
        predictions: list[Probs] = []
        for match in _chronological(matches):
            predictions.append(self.predict(match.home, match.away))
            self.update(match)
        return predictions


def devig_probs(match: MatchInput) -> Probs | None:
    """De-vigged closing odds stored on the match row, or ``None`` when absent.

    football-data.co.uk stores market-implied probabilities with the overround
    already removed; the renormalisation here only guards legacy/edge rows.
    """
    if (
        match.closing_p_home is None
        or match.closing_p_draw is None
        or match.closing_p_away is None
    ):
        return None
    probs = (
        float(match.closing_p_home),
        float(match.closing_p_draw),
        float(match.closing_p_away),
    )
    total = sum(probs)
    if total <= 0.0:
        return None
    return (probs[0] / total, probs[1] / total, probs[2] / total)
