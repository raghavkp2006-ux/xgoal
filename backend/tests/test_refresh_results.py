from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.models import Match, MatchStatus
from jobs.refresh_results import (
    ScoreChangeError,
    UnknownTeamError,
    apply_result_rows,
    parse_played_rows,
    season_matches_md5,
)


def _match() -> Match:
    stamp = datetime(2026, 8, 1, tzinfo=timezone.utc)
    return Match(
        id=101,
        competition_id=2,
        season_id=1734,
        matchday=1,
        kickoff_utc=datetime(2026, 8, 15, 19, 30, tzinfo=timezone.utc),
        home_team_id=10,
        away_team_id=20,
        status=MatchStatus.NS,
        source="laliga_official",
        ingested_at=stamp,
        updated_at=stamp,
    )


def _row() -> dict[str, str]:
    return {
        "HomeTeam": "Home Raw",
        "AwayTeam": "Away Raw",
        "FTHG": "2",
        "FTAG": "1",
        "HTHG": "1",
        "HTAG": "0",
        "HS": "14",
        "AS": "8",
        "HST": "6",
        "AST": "3",
        "HC": "7",
        "AC": "2",
        "HF": "10",
        "AF": "13",
        "HY": "2",
        "AY": "4",
        "HR": "0",
        "AR": "1",
        "Referee": "Ref Example",
        "AvgCH": "2.00",
        "AvgCD": "3.50",
        "AvgCA": "4.00",
    }


def test_played_row_updates_existing_fixture_without_touching_kickoff() -> None:
    match = _match()
    kickoff = match.kickoff_utc
    changed_at = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)
    rows = parse_played_rows([_row()])

    changed = apply_result_rows(
        [match],
        rows,
        {"Home Raw": 10, "Away Raw": 20},
        updated_at=changed_at,
    )

    assert changed == 1
    assert match.kickoff_utc == kickoff
    assert match.status == MatchStatus.FT
    assert (match.home_goals, match.away_goals) == (2, 1)
    assert (match.home_goals_ht, match.away_goals_ht) == (1, 0)
    assert (match.home_shots, match.away_shots) == (14, 8)
    assert (match.home_shots_on_tgt, match.away_shots_on_tgt) == (6, 3)
    assert (match.home_corners, match.away_corners) == (7, 2)
    assert (match.home_fouls, match.away_fouls) == (10, 13)
    assert (match.home_yellows, match.away_yellows) == (2, 4)
    assert (match.home_reds, match.away_reds) == (0, 1)
    assert match.referee == "Ref Example"
    assert match.closing_p_home == Decimal("0.4828")
    assert match.closing_p_draw == Decimal("0.2759")
    assert match.closing_p_away == Decimal("0.2414")
    assert match.updated_at == changed_at


def test_repeated_result_refresh_is_a_no_op() -> None:
    match = _match()
    rows = parse_played_rows([_row()])
    aliases = {"Home Raw": 10, "Away Raw": 20}
    first_time = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)
    second_time = datetime(2026, 9, 29, 11, 0, tzinfo=timezone.utc)

    assert apply_result_rows([match], rows, aliases, updated_at=first_time) == 1
    first_md5 = season_matches_md5([match])
    assert apply_result_rows([match], rows, aliases, updated_at=second_time) == 0

    assert season_matches_md5([match]) == first_md5
    assert match.updated_at == first_time


def test_ns_becomes_ft_without_overwriting_an_existing_ft_score() -> None:
    new_result = _match()
    existing_result = Match(
        id=102,
        competition_id=2,
        season_id=1734,
        matchday=1,
        kickoff_utc=datetime(2026, 8, 16, 19, 30, tzinfo=timezone.utc),
        home_team_id=30,
        away_team_id=40,
        status=MatchStatus.FT,
        home_goals=4,
        away_goals=3,
        home_goals_ht=2,
        away_goals_ht=1,
        source="laliga_official",
        ingested_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    existing_row = _row() | {
        "HomeTeam": "Existing Home",
        "AwayTeam": "Existing Away",
        "FTHG": "4",
        "FTAG": "3",
        "HTHG": "2",
        "HTAG": "1",
    }

    changed = apply_result_rows(
        [new_result, existing_result],
        parse_played_rows([_row(), existing_row]),
        {
            "Home Raw": 10,
            "Away Raw": 20,
            "Existing Home": 30,
            "Existing Away": 40,
        },
        updated_at=datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc),
    )

    assert changed == 2
    assert new_result.status == MatchStatus.FT
    assert (new_result.home_goals, new_result.away_goals) == (2, 1)
    assert (existing_result.home_goals, existing_result.away_goals) == (4, 3)
    assert (existing_result.home_goals_ht, existing_result.away_goals_ht) == (2, 1)


def test_existing_ft_score_change_is_blocked_before_mutation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    match = _match()
    match.status = MatchStatus.FT
    match.home_goals = 4
    match.away_goals = 3
    match.home_goals_ht = 2
    match.away_goals_ht = 1

    with pytest.raises(ScoreChangeError, match="--allow-score-changes"):
        apply_result_rows(
            [match],
            parse_played_rows([_row()]),
            {"Home Raw": 10, "Away Raw": 20},
            updated_at=datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc),
        )

    assert (match.home_goals, match.away_goals) == (4, 3)
    assert (match.home_goals_ht, match.away_goals_ht) == (2, 1)
    assert capsys.readouterr().out == (
        "existing FT score changes: 1\n"
        "match id=101 old=(4, 3, 2, 1) new=(2, 1, 1, 0)\n"
    )


def test_existing_ft_score_change_can_be_explicitly_allowed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    match = _match()
    match.status = MatchStatus.FT
    match.home_goals = 4
    match.away_goals = 3
    match.home_goals_ht = 2
    match.away_goals_ht = 1

    changed = apply_result_rows(
        [match],
        parse_played_rows([_row()]),
        {"Home Raw": 10, "Away Raw": 20},
        updated_at=datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc),
        allow_score_changes=True,
    )

    assert changed == 1
    assert (match.home_goals, match.away_goals) == (2, 1)
    assert (match.home_goals_ht, match.away_goals_ht) == (1, 0)
    assert capsys.readouterr().out == (
        "existing FT score changes: 1\n"
        "match id=101 old=(4, 3, 2, 1) new=(2, 1, 1, 0)\n"
    )


def test_unmapped_result_team_fails_loudly() -> None:
    with pytest.raises(UnknownTeamError, match="Away Raw"):
        apply_result_rows(
            [_match()],
            parse_played_rows([_row()]),
            {"Home Raw": 10},
            updated_at=datetime.now(timezone.utc),
        )
