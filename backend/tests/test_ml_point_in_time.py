"""Phase 2.2 leakage test — the most valuable thing in the suite.

Protocol (build plan Part 3, Step 2.2)
--------------------------------------
For a random sample of matches, ``build_features(match_id, as_of)`` is run twice:

* **full table** — handed the whole ``matches`` table;
* **kickoff-filtered view** — handed only the rows a forecaster may legally see,
  i.e. ``kickoff_utc < match.kickoff_utc``.

The two dicts must serialise byte-for-byte identically. Any feature that reads a
row at or after its own kick-off makes them differ, so the test fails loudly
instead of silently inflating the model's backtest.

Fixed seed
----------
The 200-match sample is drawn with ``random.Random(20260917)`` — documented and
frozen so the exact same 200 fixtures are re-checked on every run.

Four layers, because one test can be passed by accident:

1. ``test_full_table_and_kickoff_filtered_view_agree_on_200_matches`` — the
   headline comparison, against the live database.
2. ``test_a_real_postgres_view_filtered_to_kickoff_matches_the_full_table`` — the
   same comparison where the restriction is enforced by Postgres itself (a
   temporary view named ``matches``), so a builder that quietly queries the
   database instead of the supplied history is caught too.
3. ``test_tampering_with_future_results_cannot_change_a_forecast`` — rewrites
   every result at or after the cut and demands an identical feature row.
4. ``test_the_leakage_harness_detects_a_leaky_feature`` — a deliberately leaky
   feature, proving the harness is sensitive rather than vacuously green.

A database-free synthetic variant of (1) runs in CI, where no database exists, so
the guard is never silently skipped.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import SessionLocal, engine
from app.ml.baselines import EloBaseline
from app.ml.data import MatchIdentity, PointInTimeMatch
from app.ml.dataset import load_match_stream
from app.ml.features import (
    PIT_PRIOR_GOALS_AGAINST_AWAY,
    PIT_PRIOR_GOALS_FOR_AWAY,
    PIT_PRIOR_PPG,
    POINT_IN_TIME_FEATURE_NAMES,
    build_features,
)

SAMPLE_SEED = 20260917
"""Fixed seed for the documented 200-match sample."""

SAMPLE_SIZE = 200
"""Matches drawn at random for the headline comparison."""

VIEW_SAMPLE_SIZE = 25
"""Matches re-checked against a real Postgres view (each costs a DB round trip)."""

TAMPERED_GOALS = 9
TAMPERED_SHOTS_ON_TGT = 40

EXPECTED_FEATURE_COUNT = 15


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def canonical(features: dict[str, float]) -> str:
    """Byte-stable serialisation of a feature dict (sorted keys, full precision)."""
    return json.dumps(features, sort_keys=True)


def before(stream: list[PointInTimeMatch], as_of: datetime) -> list[PointInTimeMatch]:
    """The rows a forecaster at ``as_of`` is allowed to read — the filtered view."""
    return [match for match in stream if match.kickoff_utc < as_of]


def leaky_total_goals(history: list[PointInTimeMatch], as_of: datetime) -> float:
    """A deliberately leaky feature: it ignores ``as_of`` and reads everything.

    Used only to prove the comparison protocol above is capable of failing.
    """
    return float(sum(match.home_goals + match.away_goals for match in history))


def synthetic_stream(count: int = 700, seed: int = 4242) -> list[PointInTimeMatch]:
    """A database-free chronological stream with venue- and shot-shaped history."""
    rng = random.Random(seed)
    teams = list(range(1, 21))
    kickoff = datetime(2019, 8, 10, 18, 0, tzinfo=timezone.utc)
    stream: list[PointInTimeMatch] = []
    for index in range(count):
        home, away = rng.sample(teams, 2)
        stream.append(
            PointInTimeMatch(
                match_id=index + 1,
                competition_id=1,
                competition_tier=1,
                season_id=1,
                season_start_year=2019,
                kickoff_utc=kickoff,
                home_team_id=home,
                away_team_id=away,
                home_goals=rng.randint(0, 4),
                away_goals=rng.randint(0, 4),
                home_shots_on_tgt=rng.randint(1, 9),
                away_shots_on_tgt=rng.randint(1, 9),
            )
        )
        kickoff += timedelta(hours=6)
    return stream


def synthetic_identity(match: PointInTimeMatch) -> MatchIdentity:
    """The fixture being forecast, expressed independently of the history stream."""
    return MatchIdentity(
        match_id=match.match_id,
        competition_id=match.competition_id,
        competition_tier=match.competition_tier,
        season_id=match.season_id,
        season_start_year=match.season_start_year,
        kickoff_utc=match.kickoff_utc,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
    )



# --------------------------------------------------------------------------- #
# fixtures (database-backed tests skip cleanly when no database is reachable)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def db():
    """A live database session, or a skip when the environment has no database."""
    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        session.close()
        pytest.skip(f"database unavailable, leakage test skipped: {exc}")
    yield session
    session.close()


@pytest.fixture(scope="module")
def stream(db) -> list[PointInTimeMatch]:
    """Every completed fixture in the corpus, chronological."""
    return load_match_stream(db)


@pytest.fixture(scope="module")
def sample(stream: list[PointInTimeMatch]) -> list[PointInTimeMatch]:
    """The documented 200-match random sample."""
    if len(stream) < SAMPLE_SIZE:
        pytest.skip(f"corpus has only {len(stream)} completed matches")
    return random.Random(SAMPLE_SEED).sample(stream, SAMPLE_SIZE)


# --------------------------------------------------------------------------- #
# 1. the headline test — 200 matches, full table vs kickoff-filtered view
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_full_table_and_kickoff_filtered_view_agree_on_200_matches(db, stream, sample):
    """200/200: a feature row may not move when the future is taken away."""
    assert len(sample) == SAMPLE_SIZE
    checked = 0
    for match in sample:
        as_of = match.kickoff_utc
        unseen = before(stream, as_of)
        assert len(unseen) < len(stream), (
            f"match {match.match_id} kicks off after the last fixture; "
            "the comparison would be vacuous"
        )
        full = build_features(match.match_id, as_of, db=db, history=stream)
        filtered = build_features(match.match_id, as_of, db=db, history=unseen)
        if canonical(full) != canonical(filtered):
            leaked = _first_difference(full, filtered)
            pytest.fail(
                f"match {match.match_id} ({match.home_team_id} v {match.away_team_id} "
                f"@ {as_of:%Y-%m-%d}) leaks: future rows change {leaked}"
            )
        checked += 1
    assert checked == SAMPLE_SIZE
    print(f"\nleakage (phase 2.2, live table): {checked}/{SAMPLE_SIZE} rows unchanged")


def _first_difference(
    full: dict[str, float], filtered: dict[str, float]
) -> list[tuple[str, float, float]]:
    """Feature names whose value moved when the future was removed."""
    return [
        (name, full[name], filtered[name])
        for name in sorted(full)
        if full[name] != filtered.get(name)
    ]


# --------------------------------------------------------------------------- #
# 2. the same comparison, restricted by Postgres itself
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_a_real_postgres_view_filtered_to_kickoff_matches_the_full_table(db, sample):
    """The database view is the enforcement: future rows are physically invisible.

    A builder that reads the table directly instead of using the supplied history
    is caught here, because the view removes the future at the SQL level.
    """
    view_sample = random.Random(SAMPLE_SEED + 1).sample(sample, VIEW_SAMPLE_SIZE)
    checked = 0
    with engine.connect() as connection:
        for match in view_sample:
            connection.execute(
                text(
                    "CREATE OR REPLACE TEMP VIEW matches AS "
                    "SELECT * FROM public.matches WHERE kickoff_utc < :cut"
                ),
                {"cut": match.kickoff_utc},
            )
            session = Session(bind=connection)
            try:
                restricted = build_features(match.match_id, match.kickoff_utc, db=session)
            finally:
                session.close()
            full = build_features(match.match_id, match.kickoff_utc, db=db)
            if canonical(full) != canonical(restricted):
                leaked = _first_difference(full, restricted)
                pytest.fail(
                    f"match {match.match_id} reads past its own kick-off: {leaked}"
                )
            checked += 1
        connection.execute(text("DROP VIEW IF EXISTS matches"))
    assert checked == VIEW_SAMPLE_SIZE
    print(
        f"\nleakage (phase 2.2, Postgres view): {checked}/{VIEW_SAMPLE_SIZE} rows unchanged"
    )


# --------------------------------------------------------------------------- #
# 3. tampering — rewrite the future, demand an identical row
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_tampering_with_future_results_cannot_change_a_forecast(db, stream, sample):
    """200/200: absurd future scorelines leave every forecast row bit-identical."""
    checked = 0
    for match in sample:
        as_of = match.kickoff_utc
        tampered = [
            replace(
                row,
                home_goals=TAMPERED_GOALS,
                away_goals=0,
                home_shots_on_tgt=TAMPERED_SHOTS_ON_TGT,
                away_shots_on_tgt=0,
            )
            if row.kickoff_utc >= as_of
            else row
            for row in stream
        ]
        honest = build_features(match.match_id, as_of, db=db, history=stream)
        rewritten = build_features(match.match_id, as_of, db=db, history=tampered)
        if canonical(honest) != canonical(rewritten):
            leaked = _first_difference(honest, rewritten)
            pytest.fail(f"match {match.match_id} used a later result: {leaked}")
        checked += 1
    assert checked == SAMPLE_SIZE
    print(f"\nleakage (phase 2.2, tampering): {checked}/{SAMPLE_SIZE} rows unchanged")


# --------------------------------------------------------------------------- #
# 4. the harness must be able to fail
# --------------------------------------------------------------------------- #


def test_the_leakage_harness_detects_a_leaky_feature(db, stream, sample):
    """A leaky feature moves; the real builder does not. Proves the test has teeth."""
    match = sample[0]
    as_of = match.kickoff_utc
    unseen = before(stream, as_of)

    leaky_full = leaky_total_goals(stream, as_of)
    leaky_filtered = leaky_total_goals(unseen, as_of)
    assert leaky_full != leaky_filtered, (
        "the full table and the filtered view carry identical information; "
        "the comparison could never fail"
    )

    clean_full = build_features(match.match_id, as_of, db=db, history=stream)
    clean_filtered = build_features(match.match_id, as_of, db=db, history=unseen)
    assert canonical(clean_full) == canonical(clean_filtered)
    print(
        f"\nleakage (harness sensitivity): leaky feature moved by "
        f"{leaky_full - leaky_filtered:.0f} goals, real builder moved 0"
    )


# --------------------------------------------------------------------------- #
# database-free guards — these run in CI, where no database exists
# --------------------------------------------------------------------------- #


def test_synthetic_full_table_and_kickoff_filtered_view_agree():
    """200/200 on a synthetic stream: the CI-runnable version of the headline test."""
    stream = synthetic_stream()
    assert len(stream) == 700
    sample = random.Random(SAMPLE_SEED).sample(stream, SAMPLE_SIZE)

    checked = 0
    for match in sample:
        as_of = match.kickoff_utc
        identity = synthetic_identity(match)
        full = build_features(identity.match_id, as_of, identity=identity, history=stream)
        filtered = build_features(
            identity.match_id, as_of, identity=identity, history=before(stream, as_of)
        )
        assert canonical(full) == canonical(filtered), f"match {identity.match_id} leaked"
        checked += 1
    assert checked == SAMPLE_SIZE
    print(f"\nleakage (phase 2.2, synthetic): {checked}/{SAMPLE_SIZE} rows unchanged")


def test_feature_vector_is_the_documented_size_and_shape():
    """The published feature list is exactly what the builder emits."""
    stream = synthetic_stream()
    match = stream[400]
    row = build_features(
        match.match_id,
        match.kickoff_utc,
        identity=synthetic_identity(match),
        history=stream,
    )
    assert len(row) == EXPECTED_FEATURE_COUNT
    assert tuple(sorted(row)) == tuple(sorted(POINT_IN_TIME_FEATURE_NAMES))
    assert all(isinstance(value, float) for value in row.values())
    assert all(value == value for value in row.values()), "NaN in the feature row"




def test_venue_form_is_computed_separately_for_home_and_away():
    """Home form at home is not the away side's away form, and neither is overall form.

    The home team plays match 1 at home (a 4-0 win) and match 2 away (a 5-0 defeat),
    so its home-venue block must still describe the 4-0 day.
    """
    stream = [
        PointInTimeMatch(
            match_id=1,
            competition_id=1,
            competition_tier=1,
            season_id=1,
            season_start_year=2019,
            kickoff_utc=datetime(2019, 8, 10, 18, 0, tzinfo=timezone.utc),
            home_team_id=10,
            away_team_id=20,
            home_goals=4,
            away_goals=0,
            home_shots_on_tgt=9,
            away_shots_on_tgt=0,
        ),
        PointInTimeMatch(
            match_id=2,
            competition_id=1,
            competition_tier=1,
            season_id=1,
            season_start_year=2019,
            kickoff_utc=datetime(2019, 8, 17, 18, 0, tzinfo=timezone.utc),
            home_team_id=30,
            away_team_id=10,
            home_goals=5,
            away_goals=0,
            home_shots_on_tgt=11,
            away_shots_on_tgt=0,
        ),
    ]
    identity = MatchIdentity(
        match_id=3,
        competition_id=1,
        competition_tier=1,
        season_id=1,
        season_start_year=2019,
        kickoff_utc=datetime(2019, 8, 24, 18, 0, tzinfo=timezone.utc),
        home_team_id=10,
        away_team_id=40,
    )
    row = build_features(3, identity.kickoff_utc, identity=identity, history=stream)

    assert set(row) == set(POINT_IN_TIME_FEATURE_NAMES)
    assert row["home_ppg_l10"] == 3.0, "home team's home-venue form should be a 4-0 win"
    assert row["home_goals_for_l10"] == 4.0
    assert row["home_goals_against_l10"] == 0.0
    assert row["home_sot_for_l10"] == 9.0
    assert row["home_sot_against_l10"] == 0.0

    # The away block is the away side's *away* record: none, so the league priors —
    # it must not have picked up the home side's 4-0.
    assert row["away_ppg_l10"] == PIT_PRIOR_PPG
    assert row["away_goals_for_l10"] == PIT_PRIOR_GOALS_FOR_AWAY
    assert row["away_goals_against_l10"] == PIT_PRIOR_GOALS_AGAINST_AWAY


def _capped_rest_days(stream, team_id, as_of):
    """Independent rest-days calculation, mirroring the documented cap."""
    from app.ml.features import REST_DAYS_CAP

    last = max(
        (row for row in stream if row.involves(team_id)),
        key=lambda row: row.kickoff_utc,
    )
    gap = (as_of - last.kickoff_utc).total_seconds() / 86400.0
    return min(max(gap, 0.0), REST_DAYS_CAP)


def test_rest_days_diff_is_home_minus_away():
    """``rest_days_diff`` is the home side's rest minus the away side's, in days."""
    stream = synthetic_stream()
    match = stream[300]
    identity = synthetic_identity(match)
    row = build_features(
        identity.match_id, identity.kickoff_utc, identity=identity, history=stream
    )
    unseen = before(stream, identity.kickoff_utc)
    assert len(unseen) == 300

    home = _capped_rest_days(unseen, identity.home_team_id, identity.kickoff_utc)
    away = _capped_rest_days(unseen, identity.away_team_id, identity.kickoff_utc)
    assert row["rest_days_diff"] == pytest.approx(home - away)
    assert row["rest_days_diff"] != 0.0, "synthetic stream should not be exactly level"
    assert abs(row["rest_days_diff"]) <= 30.0, "differential must stay inside the cap"


def test_rest_days_diff_is_bounded_and_zero_for_a_season_opener():
    """No history means both sides are capped, so the differential is exactly zero."""
    from app.ml.features import REST_DAYS_CAP

    stream = synthetic_stream()
    opener = MatchIdentity(
        match_id=999_001,
        competition_id=1,
        competition_tier=1,
        season_id=1,
        season_start_year=2019,
        kickoff_utc=datetime(2019, 8, 9, 18, 0, tzinfo=timezone.utc),  # before the stream
        home_team_id=10,
        away_team_id=20,
    )
    row = build_features(
        opener.match_id, opener.kickoff_utc, identity=opener, history=stream
    )
    assert row["rest_days_diff"] == 0.0
    assert REST_DAYS_CAP == 30.0


def test_elo_differential_is_pulled_from_the_phase_2_1_module():
    """``elo_diff`` is read off :class:`EloBaseline`; the builder never recomputes Elo."""
    stream = synthetic_stream()
    match = stream[500]
    identity = synthetic_identity(match)
    row = build_features(
        identity.match_id, identity.kickoff_utc, identity=identity, history=stream
    )

    module = EloBaseline().fit(
        [row.to_match_input() for row in before(stream, identity.kickoff_utc)]
    )
    expected = module.rating(identity.home_team_id) - module.rating(identity.away_team_id)
    assert row["elo_diff"] == pytest.approx(expected, abs=1e-12)
    assert row["elo_diff"] != 0.0


def test_promotion_flag_marks_a_side_that_was_not_in_the_division_last_season():
    """A club with no matches in this division last season is a promoted side."""
    stream = [
        PointInTimeMatch(
            match_id=1,
            competition_id=2,
            competition_tier=2,
            season_id=1,
            season_start_year=2019,
            kickoff_utc=datetime(2019, 9, 1, 18, 0, tzinfo=timezone.utc),
            home_team_id=10,
            away_team_id=20,
            home_goals=2,
            away_goals=1,
        ),
        PointInTimeMatch(
            match_id=2,
            competition_id=1,
            competition_tier=1,
            season_id=2,
            season_start_year=2020,
            kickoff_utc=datetime(2020, 9, 12, 18, 0, tzinfo=timezone.utc),
            home_team_id=30,
            away_team_id=40,
            home_goals=1,
            away_goals=1,
        ),
    ]
    identity = MatchIdentity(
        match_id=3,
        competition_id=1,
        competition_tier=1,
        season_id=2,
        season_start_year=2020,
        kickoff_utc=datetime(2020, 9, 19, 18, 0, tzinfo=timezone.utc),
        home_team_id=10,
        away_team_id=30,
    )
    row = build_features(3, identity.kickoff_utc, identity=identity, history=stream)
    assert row["promotion_flag"] == 1.0, "team 10 came from tier 2"
    assert row["division_tier"] == 1.0


def test_division_indicator_carries_the_multi_league_corpus_tier():
    """The tier of the fixture's competition is exposed for the multi-league corpus."""
    stream = synthetic_stream()
    match = stream[100]
    identity = replace(synthetic_identity(match), competition_tier=2)
    row = build_features(
        identity.match_id, identity.kickoff_utc, identity=identity, history=stream
    )
    assert row["division_tier"] == 2.0
