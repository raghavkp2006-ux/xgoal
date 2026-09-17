"""Feature builders for the Phase 2 models.

Two builders live here:

``FeatureBuilder`` (Phase 2.2, stream form)
    The 14-feature rolling vector consumed by the XGBoost comparison model. It is
    fed a chronological match stream and keeps per-team state, so it is leak-free
    by construction as long as the caller feeds it in order.

``PointInTimeFeatureBuilder`` + :func:`build_features` (Phase 2.2, database form)
    The plan's contract: ``build_features(match_id, as_of) -> dict``. It may read
    only rows with ``kickoff_utc < as_of`` — no exceptions — and produces the
    venue-split form vector plus the match-level context features (rest-day
    differential, matchday, Elo differential, promotion flag, division tier).

The hard constraint is enforced in one place and tested at the database level:
``tests/test_ml_point_in_time.py`` builds each of 200 sampled fixtures twice, once
against the full ``matches`` table and once against a view filtered to
``kickoff_utc < match.kickoff_utc``, and demands byte-identical output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from app.ml.baselines import EloBaseline
from app.ml.data import FloatArray, IntArray, MatchIdentity, MatchInput, PointInTimeMatch, TeamKey

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import-light
    from sqlalchemy.orm import Session


FEATURE_NAMES: tuple[str, ...] = (
    "home_ppg",
    "home_goals_for",
    "home_goals_against",
    "home_sot_share",
    "home_win_rate",
    "home_rest_days",
    "home_matches_played",
    "away_ppg",
    "away_goals_for",
    "away_goals_against",
    "away_sot_share",
    "away_win_rate",
    "away_rest_days",
    "away_matches_played",
)

# League priors used for teams with no recorded history (roughly La Liga).
PRIOR_PPG = 1.35
PRIOR_GOALS_FOR = 1.35
PRIOR_GOALS_AGAINST = 1.35
PRIOR_WIN_RATE = 0.35
PRIOR_SOT_SHARE = 0.5
PRIOR_REST_DAYS = 7.0

# -- Phase 2.2 venue-split feature set -------------------------------------- #

POINT_IN_TIME_FEATURE_NAMES: tuple[str, ...] = (
    # Home side's form at home, its own venue.
    "home_ppg_l10",
    "home_goals_for_l10",
    "home_goals_against_l10",
    "home_sot_for_l10",
    "home_sot_against_l10",
    # Away side's form away. Kept separate from the home block on purpose:
    # home form at home is more predictive than overall form, so the two
    # venue histories are never collapsed together.
    "away_ppg_l10",
    "away_goals_for_l10",
    "away_goals_against_l10",
    "away_sot_for_l10",
    "away_sot_against_l10",
    # Match-level context.
    "rest_days_diff",
    "matchday",
    "elo_diff",
    "promotion_flag",
    "division_tier",
)
"""The 15-number Phase 2.2 vector, in fixed order.

Ten venue-specific rolling-form features — points per game, goals for and against,
and shots on target for and against over the last ten matches, the home side's
record at home and the away side's record away — plus five match-level features.
Every rolling stat shares the same ten-match window.

Shots on target are the shot-quality proxy (D28); raw shot counts are deliberately
dropped, being near-collinear with SoT and adding little at this sample size.

Also deliberately absent: current league position (redundant with rolling points,
and leakage-prone if computed carelessly), head-to-head record (D29 — weak at ~20
matches per pairing) and bookmaker odds (D27 — benchmark only, never a feature).
"""

FORM_WINDOW = 10
"""Rolling window for every form feature."""

REST_DAYS_CAP = 30.0
"""Rest days are capped here so a season opener's ~100-day gap cannot dominate."""

# Priors for a club with no recorded history in the window. The SoT priors are the
# ingested La Liga means (4.65 / 3.63 shots on target per game).
PIT_PRIOR_PPG = 1.35
PIT_PRIOR_GOALS_FOR_HOME = 1.50
PIT_PRIOR_GOALS_AGAINST_HOME = 1.14
PIT_PRIOR_GOALS_FOR_AWAY = 1.14
PIT_PRIOR_GOALS_AGAINST_AWAY = 1.50
PIT_PRIOR_SOT_FOR_HOME = 4.65
PIT_PRIOR_SOT_AGAINST_HOME = 3.63
PIT_PRIOR_SOT_FOR_AWAY = 3.63
PIT_PRIOR_SOT_AGAINST_AWAY = 4.65


@dataclass(frozen=True)
class FeatureDataset:
    """Design matrix plus the metadata needed to align predictions."""

    matrix: FloatArray
    outcomes: IntArray
    matches: list[MatchInput]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __len__(self) -> int:
        """Number of fixtures (rows) in the dataset."""
        return int(self.matrix.shape[0])


class FeatureBuilder:
    """Rolling, leak-free feature extraction over a chronological match stream.

    The builder keeps per-team history and is meant to be fed the match stream
    in order — training window first, then the fixtures being predicted — so the
    state always reflects only what was known at kick-off time.
    """

    def __init__(self, window: int = 6, rest_cap_days: float = 14.0) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self.rest_cap_days = rest_cap_days
        self.history: dict[
            TeamKey, deque[tuple[datetime, int, int, int, int | None, int | None]]
        ] = {}

    def reset(self) -> None:
        """Forget every team's history."""
        self.history = {}

    def export_state(self) -> dict[str, list[list[Any]]]:
        """JSON-serialisable snapshot of every team's rolling window.

        Team keys are written as strings, which is the key the ML package uses
        everywhere else (``dataset.py`` maps rows to canonical team names), so a
        restored builder lines up with the fixtures it is asked to predict.
        """
        return {
            str(team): [
                [entry[0].isoformat(), entry[1], entry[2], entry[3], entry[4], entry[5]]
                for entry in window
            ]
            for team, window in self.history.items()
        }

    def import_state(self, payload: Mapping[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`export_state`."""
        restored: dict[
            TeamKey, deque[tuple[datetime, int, int, int, int | None, int | None]]
        ] = {}
        for team, entries in payload.items():
            window: deque[tuple[datetime, int, int, int, int | None, int | None]] = deque(
                maxlen=self.window
            )
            for entry in entries:
                window.append(
                    (
                        datetime.fromisoformat(str(entry[0])),
                        int(entry[1]),
                        int(entry[2]),
                        int(entry[3]),
                        None if entry[4] is None else int(entry[4]),
                        None if entry[5] is None else int(entry[5]),
                    )
                )
            restored[str(team)] = window
        self.history = restored

    def build(self, matches: Sequence[MatchInput]) -> FeatureDataset:
        """Build features for ``matches`` in kick-off order, updating state as it goes."""
        ordered = sorted(matches, key=lambda m: m.kickoff)
        rows: list[list[float]] = []
        outcomes: list[int] = []
        for match in ordered:
            rows.append(self._row(match))
            outcomes.append(match.outcome)
            self._update(match)
        matrix = (
            np.asarray(rows, dtype=np.float64)
            if rows
            else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
        )
        return FeatureDataset(
            matrix=matrix,
            outcomes=np.asarray(outcomes, dtype=np.int64),
            matches=ordered,
        )

    # -- internals --------------------------------------------------------

    def _row(self, match: MatchInput) -> list[float]:
        """Feature row for one fixture: home block followed by the away block."""
        return self._team_features(match.home, match.kickoff) + self._team_features(
            match.away, match.kickoff
        )

    def _team_features(self, team: TeamKey, kickoff: datetime) -> list[float]:
        """The 7 features describing one team going into ``kickoff``."""
        history = self.history.get(team)
        if not history:
            return [
                PRIOR_PPG,
                PRIOR_GOALS_FOR,
                PRIOR_GOALS_AGAINST,
                PRIOR_SOT_SHARE,
                PRIOR_WIN_RATE,
                PRIOR_REST_DAYS,
                0.0,
            ]
        n = len(history)
        points = sum(entry[1] for entry in history)
        goals_for = sum(entry[2] for entry in history)
        goals_against = sum(entry[3] for entry in history)
        sot_for = sum(entry[4] for entry in history if entry[4] is not None)
        sot_against = sum(entry[5] for entry in history if entry[5] is not None)
        wins = sum(1 for entry in history if entry[1] == 3)
        total_sot = sot_for + sot_against
        sot_share = (sot_for / total_sot) if total_sot > 0 else PRIOR_SOT_SHARE
        rest = (kickoff - history[-1][0]).total_seconds() / 86400.0
        rest = min(max(rest, 0.0), self.rest_cap_days)
        return [
            points / n,
            goals_for / n,
            goals_against / n,
            sot_share,
            wins / n,
            rest,
            float(n),
        ]

    def _update(self, match: MatchInput) -> None:
        """Fold a finished match into both teams' rolling windows."""
        home_points = 3 if match.outcome == 0 else (1 if match.outcome == 1 else 0)
        away_points = 3 if match.outcome == 2 else (1 if match.outcome == 1 else 0)
        home_entry = (
            match.kickoff,
            home_points,
            match.home_goals,
            match.away_goals,
            match.home_shots_on_tgt,
            match.away_shots_on_tgt,
        )
        away_entry = (
            match.kickoff,
            away_points,
            match.away_goals,
            match.home_goals,
            match.away_shots_on_tgt,
            match.home_shots_on_tgt,
        )
        for team, entry in ((match.home, home_entry), (match.away, away_entry)):
            window = self.history.setdefault(team, deque(maxlen=self.window))
            window.append(entry)


def build_feature_dataset(matches: Sequence[MatchInput], window: int = 6) -> FeatureDataset:
    """Convenience wrapper: a one-shot leak-free dataset for ``matches``.

    Renamed from ``build_features`` when Phase 2.2 claimed that name for the
    database-facing contract ``build_features(match_id, as_of)``. Nothing in the
    repository called the old name; :class:`XGBoostModel` drives ``FeatureBuilder``
    directly.
    """
    return FeatureBuilder(window=window).build(matches)


# --------------------------------------------------------------------------- #
# Phase 2.2 — the database-facing point-in-time builder
# --------------------------------------------------------------------------- #


class PointInTimeFeatureBuilder:
    """Builds the Phase 2.2 vector for one fixture as of one moment.

    The single hard rule of Phase 2.2 lives in :meth:`visible`: **only rows with
    ``kickoff_utc < as_of`` may be read.** Every feature is derived from that one
    filtered list, so there is no second code path that could quietly see the
    future.

    Elo is not recomputed here. The differential is read off
    :class:`app.ml.baselines.EloBaseline` — the Phase 2.1 module — which is fitted
    on the very same visible history. Because the Elo stream spans every ingested
    division (La Liga, Segunda, and the rest of the training corpus), a newly
    promoted side arrives carrying the rating it earned in the lower division,
    which is the prior-division rating the plan asks for.
    """

    def __init__(self, *, elo: EloBaseline | None = None) -> None:
        self.elo = elo

    def visible(
        self, history: Sequence[PointInTimeMatch], as_of: datetime
    ) -> list[PointInTimeMatch]:
        """The only rows this builder is allowed to read."""
        return [row for row in history if row.kickoff_utc < as_of]

    def build(
        self,
        identity: MatchIdentity,
        as_of: datetime,
        history: Sequence[PointInTimeMatch],
    ) -> dict[str, float]:
        """The full feature dict for ``identity`` as of ``as_of``."""
        rows = self.visible(history, as_of)
        home_at_home = [row for row in rows if row.home_team_id == identity.home_team_id]
        away_away = [row for row in rows if row.away_team_id == identity.away_team_id]

        values: dict[str, float] = {}
        values.update(
            self._venue_block("home", home_at_home, identity.home_team_id, at_home=True)
        )
        values.update(
            self._venue_block("away", away_away, identity.away_team_id, at_home=False)
        )
        values["rest_days_diff"] = self._rest_days_diff(identity, rows, as_of)
        values["matchday"] = self._matchday(identity, rows)
        values["elo_diff"] = self._elo_diff(identity, rows)
        values["promotion_flag"] = self._promotion_flag(identity, rows)
        values["division_tier"] = float(identity.competition_tier)

        # Emit in the published order so the dict is stable and self-documenting.
        return {name: values[name] for name in POINT_IN_TIME_FEATURE_NAMES}

    # -- venue-specific rolling form --------------------------------------- #

    def _venue_block(
        self,
        prefix: str,
        rows: Sequence[PointInTimeMatch],
        team_id: int,
        *,
        at_home: bool,
    ) -> dict[str, float]:
        """Rolling form for one team restricted to one venue.

        ``rows`` already contains only that team's matches at that venue, so the
        home side's block never mixes in its away form and vice versa.
        """
        window = rows[-FORM_WINDOW:]
        return {
            f"{prefix}_ppg_l10": _rate(
                [float(row.points(team_id)) for row in window], PIT_PRIOR_PPG
            ),
            f"{prefix}_goals_for_l10": _rate(
                [float(row.goals_for(team_id)) for row in window],
                PIT_PRIOR_GOALS_FOR_HOME if at_home else PIT_PRIOR_GOALS_FOR_AWAY,
            ),
            f"{prefix}_goals_against_l10": _rate(
                [float(row.goals_against(team_id)) for row in window],
                PIT_PRIOR_GOALS_AGAINST_HOME if at_home else PIT_PRIOR_GOALS_AGAINST_AWAY,
            ),
            f"{prefix}_sot_for_l10": _rate_optional(
                [row.shots_on_target_for(team_id) for row in window],
                PIT_PRIOR_SOT_FOR_HOME if at_home else PIT_PRIOR_SOT_FOR_AWAY,
            ),
            f"{prefix}_sot_against_l10": _rate_optional(
                [row.shots_on_target_against(team_id) for row in window],
                PIT_PRIOR_SOT_AGAINST_HOME if at_home else PIT_PRIOR_SOT_AGAINST_AWAY,
            ),
        }

    # -- match-level context ----------------------------------------------- #

    def _rest_days(
        self, rows: Sequence[PointInTimeMatch], team_id: int, as_of: datetime
    ) -> float:
        """Days between ``as_of`` and the team's most recent match, capped."""
        last: datetime | None = None
        for row in rows:
            if row.involves(team_id) and (last is None or row.kickoff_utc > last):
                last = row.kickoff_utc
        if last is None:
            return REST_DAYS_CAP
        gap = (as_of - last).total_seconds() / 86400.0
        return float(min(max(gap, 0.0), REST_DAYS_CAP))

    def _rest_days_diff(
        self,
        identity: MatchIdentity,
        rows: Sequence[PointInTimeMatch],
        as_of: datetime,
    ) -> float:
        """Home rest minus away rest, in days.

        A single relative-freshness signal rather than two near-mirror columns,
        matching the ``elo_diff`` pattern: positive means the home side has had
        longer since its last match. Each side is capped at ``REST_DAYS_CAP``
        first, so the differential is bounded to ±``REST_DAYS_CAP``.
        """
        return self._rest_days(rows, identity.home_team_id, as_of) - self._rest_days(
            rows, identity.away_team_id, as_of
        )

    def _matchday(self, identity: MatchIdentity, rows: Sequence[PointInTimeMatch]) -> float:
        """Round number, or a derived equivalent when the column is unpopulated.

        The ingested ``matches.matchday`` column is currently NULL for every row,
        so the fallback is the home side's completed fixtures in this competition
        and season, plus one. It is also the more honest number: it can only ever
        count matches that have already happened.
        """
        if identity.matchday is not None:
            return float(identity.matchday)
        played = sum(
            1
            for row in rows
            if row.competition_id == identity.competition_id
            and row.season_id == identity.season_id
            and row.involves(identity.home_team_id)
        )
        return float(played + 1)

    def _elo_diff(self, identity: MatchIdentity, rows: Sequence[PointInTimeMatch]) -> float:
        """Home minus away Elo going into the match, from the Phase 2.1 module."""
        elo = self.elo
        if elo is None:
            elo = EloBaseline().fit([row.to_match_input() for row in rows])
        return float(
            elo.rating(identity.home_team_id) - elo.rating(identity.away_team_id)
        )

    def _promotion_flag(self, identity: MatchIdentity, rows: Sequence[PointInTimeMatch]) -> float:
        """1.0 when either side was playing a lower division the season before.

        A club is newly promoted into ``identity.competition_id`` when it has no
        fixtures in that competition in the previous season and does have fixtures
        in a lower tier. The check falls back to 0.0 when the previous season is
        not in the corpus, rather than guessing.
        """
        previous_start_year = identity.season_start_year - 1
        previous = [row for row in rows if row.season_start_year == previous_start_year]
        if not previous:
            return 0.0
        for team_id in (identity.home_team_id, identity.away_team_id):
            in_division = any(
                row.competition_id == identity.competition_id and row.involves(team_id)
                for row in previous
            )
            if in_division:
                continue
            from_lower = any(
                row.competition_tier > identity.competition_tier and row.involves(team_id)
                for row in previous
            )
            if from_lower:
                return 1.0
        return 0.0



def _rate(values: Sequence[float], prior: float) -> float:
    """Mean of ``values``, or ``prior`` when the window is empty."""
    if not values:
        return float(prior)
    return float(sum(values) / len(values))


def _rate_optional(values: Sequence[int | None], prior: float) -> float:
    """Mean over the rows that carry the stat, or ``prior`` when none do.

    Shot columns are populated for La Liga only, so a side in another division
    gets the documented league prior rather than a spurious zero.
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return float(prior)
    return float(sum(present) / len(present))


def build_features(
    match_id: int,
    as_of: datetime,
    *,
    identity: MatchIdentity | None = None,
    history: Sequence[PointInTimeMatch] | None = None,
    db: Session | None = None,
    elo: EloBaseline | None = None,
) -> dict[str, float]:
    """The Phase 2.2 contract: point-in-time features for one fixture.

    **Hard constraint:** this function reads only rows with ``kickoff_utc < as_of``.
    No exceptions. The cut is applied once, in
    :meth:`PointInTimeFeatureBuilder.visible`, and every feature is derived from
    that filtered list — rolling venue form, rest days, matchday, the Elo
    differential and the promotion flag alike.

    Parameters
    ----------
    match_id, as_of
        The fixture to price and the moment to price it from. ``as_of`` is
        normally the fixture's own kick-off, giving a true pre-match snapshot.
    identity, history
        Optional pre-loaded inputs. ``identity`` is the fixture's who/when/where
        (never its score); ``history`` is the match stream, which is filtered by
        ``as_of`` here rather than by the caller, so the leakage guard can hand in
        the *entire* table and confirm the future goes unused. Supplying both makes
        the function database-free, which is how the CI-runnable leakage test and
        the unit tests drive it.
    db
        SQLAlchemy session. Opened and closed here when omitted, and required only
        for whichever of ``identity``/``history`` was not supplied.
    elo
        A Phase 2.1 :class:`EloBaseline` already fitted at the cut. Omit it and the
        module is fitted on the visible history instead — same numbers, more work.

    Returns
    -------
    dict[str, float]
        All :data:`POINT_IN_TIME_FEATURE_NAMES` keys, in that order.
    """
    from app.ml.dataset import load_match_identity, load_match_stream

    owned: Session | None = None
    session = db
    if session is None and (identity is None or history is None):
        from app.database import SessionLocal

        owned = SessionLocal()
        session = owned

    try:
        if identity is None or history is None:
            if session is None:  # pragma: no cover - unreachable by construction
                raise RuntimeError("a database session is required to load match data")
            if identity is None:
                identity = load_match_identity(session, match_id)
            if history is None:
                history = load_match_stream(session)
        return PointInTimeFeatureBuilder(elo=elo).build(identity, as_of, history)
    finally:
        if owned is not None:
            owned.close()

