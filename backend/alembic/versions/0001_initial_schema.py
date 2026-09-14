"""Initial schema — all tables from Part 4 of the build plan.

Revision ID: 0001
Revises:
Create Date: 2026-09-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Enums ---
    op.execute("""
        CREATE TYPE match_status AS ENUM (
            'NS','1H','HT','2H','ET','BT','P','FT','AET','PEN',
            'PST','CANC','ABD','SUSP','INT','TBD','AWD','WO'
        )
    """)

    # --- Competitions ---
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.Text(), unique=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("country", sa.Text(), nullable=False),
        sa.Column("tier", sa.SmallInteger(), nullable=False),
        sa.Column("api_football_id", sa.Integer(), unique=True, nullable=True),
    )

    # --- Seasons ---
    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id"), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("start_year", sa.SmallInteger(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("is_current", sa.Boolean(), default=False),
        sa.UniqueConstraint("competition_id", "start_year"),
    )

    # --- Teams ---
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("canonical_name", sa.Text(), unique=True, nullable=False),
        sa.Column("short_name", sa.Text(), nullable=True),
        sa.Column("api_football_id", sa.Integer(), unique=True, nullable=True),
        sa.Column("founded", sa.SmallInteger(), nullable=True),
        sa.Column("venue_name", sa.Text(), nullable=True),
        sa.Column("logo_url", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # --- Team Aliases ---
    op.create_table(
        "team_aliases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("raw_name", sa.Text(), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.UniqueConstraint("source", "raw_name"),
    )

    # --- Matches ---
    op.create_table(
        "matches",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id"), nullable=False),
        sa.Column("season_id", sa.Integer(), sa.ForeignKey("seasons.id"), nullable=False),
        sa.Column("matchday", sa.SmallInteger(), nullable=True),
        sa.Column("kickoff_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("status", sa.Enum("NS","1H","HT","2H","ET","BT","P","FT","AET","PEN","PST","CANC","ABD","SUSP","INT","TBD","AWD","WO", name="match_status"), default="NS"),
        sa.Column("minute", sa.SmallInteger(), nullable=True),
        sa.Column("home_goals", sa.SmallInteger(), sa.CheckConstraint("home_goals BETWEEN 0 AND 20"), nullable=True),
        sa.Column("away_goals", sa.SmallInteger(), sa.CheckConstraint("away_goals BETWEEN 0 AND 20"), nullable=True),
        sa.Column("home_goals_ht", sa.SmallInteger(), nullable=True),
        sa.Column("away_goals_ht", sa.SmallInteger(), nullable=True),
        sa.Column("home_shots", sa.SmallInteger(), nullable=True),
        sa.Column("away_shots", sa.SmallInteger(), nullable=True),
        sa.Column("home_shots_on_tgt", sa.SmallInteger(), nullable=True),
        sa.Column("away_shots_on_tgt", sa.SmallInteger(), nullable=True),
        sa.Column("home_corners", sa.SmallInteger(), nullable=True),
        sa.Column("away_corners", sa.SmallInteger(), nullable=True),
        sa.Column("home_fouls", sa.SmallInteger(), nullable=True),
        sa.Column("away_fouls", sa.SmallInteger(), nullable=True),
        sa.Column("home_yellows", sa.SmallInteger(), nullable=True),
        sa.Column("away_yellows", sa.SmallInteger(), nullable=True),
        sa.Column("home_reds", sa.SmallInteger(), nullable=True),
        sa.Column("away_reds", sa.SmallInteger(), nullable=True),
        sa.Column("referee", sa.Text(), nullable=True),
        sa.Column("closing_p_home", sa.Numeric(5, 4), nullable=True),
        sa.Column("closing_p_draw", sa.Numeric(5, 4), nullable=True),
        sa.Column("closing_p_away", sa.Numeric(5, 4), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("home_team_id <> away_team_id", name="no_self_play"),
        sa.UniqueConstraint("season_id", "home_team_id", "away_team_id"),
        sa.UniqueConstraint("source", "external_id"),
    )
    op.create_index("idx_matches_kickoff", "matches", ["kickoff_utc"])
    op.create_index("idx_matches_season_status", "matches", ["season_id", "status"])
    op.create_index("idx_matches_home_time", "matches", ["home_team_id", "kickoff_utc"])
    op.create_index("idx_matches_away_time", "matches", ["away_team_id", "kickoff_utc"])
    op.create_index("idx_matches_live", "matches", ["status"])

    # --- Match Events ---
    op.create_table(
        "match_events",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("match_id", sa.BigInteger(), sa.ForeignKey("matches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("minute", sa.SmallInteger(), nullable=False),
        sa.Column("extra_minute", sa.SmallInteger(), nullable=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("player_id", sa.Integer(), nullable=True),
        sa.Column("assist_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.UniqueConstraint("match_id", "minute", "event_type", "player_id", "detail"),
    )

    # --- Players ---
    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("api_football_id", sa.Integer(), unique=True, nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("nationality", sa.Text(), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("position", sa.Text(), nullable=True),
    )

    # --- Player Season Stats ---
    op.create_table(
        "player_season_stats",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("season_id", sa.Integer(), sa.ForeignKey("seasons.id"), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("appearances", sa.SmallInteger(), nullable=True),
        sa.Column("lineups", sa.SmallInteger(), nullable=True),
        sa.Column("minutes", sa.Integer(), nullable=True),
        sa.Column("goals", sa.SmallInteger(), nullable=True),
        sa.Column("assists", sa.SmallInteger(), nullable=True),
        sa.Column("shots", sa.SmallInteger(), nullable=True),
        sa.Column("shots_on_target", sa.SmallInteger(), nullable=True),
        sa.Column("passes", sa.Integer(), nullable=True),
        sa.Column("pass_accuracy", sa.Numeric(5, 2), nullable=True),
        sa.Column("yellows", sa.SmallInteger(), nullable=True),
        sa.Column("reds", sa.SmallInteger(), nullable=True),
        sa.Column("rating", sa.Numeric(4, 2), nullable=True),
        sa.UniqueConstraint("player_id", "team_id", "season_id", "snapshot_date", name="uq_player_season_snapshot"),
    )
    op.create_index("idx_pss_lookup", "player_season_stats", ["season_id", "team_id", sa.text("snapshot_date DESC")])
    op.create_index("idx_pss_goals", "player_season_stats", ["season_id", sa.text("goals DESC")])

    # --- Standings Snapshots ---
    op.create_table(
        "standings_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("season_id", sa.Integer(), sa.ForeignKey("seasons.id"), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of_matchday", sa.SmallInteger(), nullable=True),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("played", sa.SmallInteger(), nullable=True),
        sa.Column("won", sa.SmallInteger(), nullable=True),
        sa.Column("drawn", sa.SmallInteger(), nullable=True),
        sa.Column("lost", sa.SmallInteger(), nullable=True),
        sa.Column("goals_for", sa.SmallInteger(), nullable=True),
        sa.Column("goals_against", sa.SmallInteger(), nullable=True),
        sa.Column("points", sa.SmallInteger(), nullable=True),
        sa.Column("form", sa.Text(), nullable=True),
        sa.UniqueConstraint("season_id", "computed_at", "team_id"),
    )
    op.create_index("idx_standings_latest", "standings_snapshots", ["season_id", sa.text("computed_at DESC")])

    # --- Model Versions ---
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("git_sha", sa.Text(), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("train_start", sa.Date(), nullable=False),
        sa.Column("train_end", sa.Date(), nullable=False),
        sa.Column("n_train_matches", sa.Integer(), nullable=False),
        sa.Column("hyperparameters", JSONB(), nullable=False),
        sa.Column("eval_metrics", JSONB(), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("is_production", sa.Boolean(), default=False),
        sa.UniqueConstraint("name", "version"),
    )

    # --- Predictions ---
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("match_id", sa.BigInteger(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("p_home", sa.Numeric(6, 5), nullable=False),
        sa.Column("p_draw", sa.Numeric(6, 5), nullable=False),
        sa.Column("p_away", sa.Numeric(6, 5), nullable=False),
        sa.Column("expected_home_goals", sa.Numeric(4, 2), nullable=True),
        sa.Column("expected_away_goals", sa.Numeric(4, 2), nullable=True),
        sa.Column("score_matrix", JSONB(), nullable=True),
        sa.Column("feature_hash", sa.Text(), nullable=False),
        sa.CheckConstraint("abs(p_home + p_draw + p_away - 1) < 0.001", name="probs_sum_to_one"),
        sa.UniqueConstraint("match_id", "model_version_id", "as_of"),
    )
    op.create_index("idx_predictions_match", "predictions", ["match_id"])

    # --- Simulation Runs ---
    op.create_table(
        "simulation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("season_id", sa.Integer(), sa.ForeignKey("seasons.id"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("n_simulations", sa.Integer(), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=False),
        sa.Column("as_of_matchday", sa.SmallInteger(), nullable=False),
        sa.Column("forced_results", JSONB(), nullable=True),
        sa.Column("results", JSONB(), nullable=False),
    )
    op.create_index("idx_simruns_latest", "simulation_runs", ["season_id", sa.text("run_at DESC")])

    # --- API Request Log ---
    op.create_table(
        "api_request_log",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("params", JSONB(), nullable=True),
        sa.Column("status_code", sa.SmallInteger(), nullable=True),
        sa.Column("quota_remaining", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
    )
    op.create_index("idx_api_log_day", "api_request_log", ["requested_at"])

    # --- Data Freshness ---
    op.create_table(
        "data_freshness",
        sa.Column("source", sa.Text(), primary_key=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("rows_affected", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("data_freshness")
    op.drop_table("api_request_log")
    op.drop_table("simulation_runs")
    op.drop_table("predictions")
    op.drop_table("model_versions")
    op.drop_table("standings_snapshots")
    op.drop_table("player_season_stats")
    op.drop_table("players")
    op.drop_table("match_events")
    op.drop_table("matches")
    op.drop_table("team_aliases")
    op.drop_table("teams")
    op.drop_table("seasons")
    op.drop_table("competitions")
    op.execute("DROP TYPE match_status")