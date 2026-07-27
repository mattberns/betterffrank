"""Preseason situation features, keyed (season, gsis_id), seasons 2011-2026.

One row per season-t ADP-pool player (QB/RB/WR/TE, from data/processed/adp.parquet,
non-null gsis_id). FOUR independent, individually null-safe blocks: a season whose
source is missing gets nulls for that block's columns, never an error. Every feature
is computable BEFORE season t kicks off. Timing rule per feature:

  BLOCK 1 -- snap share (source: t-1 REG offensive snap counts). t-1 outcome,
             known before season-t drafts.
    snap_share        mean weekly offense_pct across the player's t-1 REG games.
    snap_share_l4f4   mean of last 4 games minus mean of first 4 games (by week
                      order); null when < 6 t-1 games (mirrors opp velocity).
    snap_share_max4   mean of the 4 highest weekly offense_pct values (usage
                      ceiling); null when < 4 t-1 games.
  BLOCK 2 -- injury history (source: t-1 and t-2 REG injury reports + weekly
             reserve-list rosters). Past outcomes, concluded by preseason t.
             0 (never null) when the player simply had no listings; both
             sources cover every target season.
    inj_weeks_listed_l2y  # distinct (season, week) over t-2..t-1 where the
                          player was EITHER listed Questionable/Doubtful/Out on
                          the game-status report OR on a reserve list
                          (roster status RES/PUP).
    inj_soft_tissue_l2y   QDO-listed weeks restricted to soft-tissue keywords
                          in report_primary_injury OR practice_primary_injury.
    inj_recurrence        1 if the same lowercased report_primary_injury value
                          appears on QDO-listed rows in BOTH t-1 and t-2.

             TWO measurement rules, both load-bearing (2026-07-27) -- do not
             "simplify" back to report_status non-null:
             (1) Only Questionable/Doubtful/Out count. The NFL abolished the
                 "Probable" designation after 2015 (~2,500 listings/season),
                 and 2016+ files carry ~3,000 practice-only rows/season with
                 NULL report_status (pre-2016: ~200). Counting non-null status
                 made target seasons <=2017 measure ~2x what 2018+ measured --
                 the tune window was graded on a different variable than the
                 test window (pool mean 6.2 vs 3.1; QDO-only is level, 2.5-3.4
                 across all seasons).
             (2) Reserve-list weeks are unioned in because IR players drop OFF
                 the weekly injury report: the most severe injuries otherwise
                 produce the LOWEST counts (CMC 2024: 13 games missed, 12 RES
                 weeks, only 3 report-listed weeks -- less than a nagging
                 hamstring). RES/PUP only: INA (game-day inactive) exists only
                 2020+ and includes healthy scratches; PUP is folded into RES
                 before 2016. Known residual, reported not hidden: league IR
                 usage genuinely rose ~2016 (RES rows 1.9k -> 4.3k-5.4k); that
                 is roster-behavior reality, not taxonomy.
  BLOCK 3 -- contracts (source: historical_contracts, filtered year_signed <= t).
             Preseason-known. Players with no contract row: nulls.
    apy_cap_pct       APY as % of the cap on the most recent contract signed <= t.
    contract_year     1 if year_signed + years == t (final contract season), else 0.
    rookie_deal_yr    t - draft_year + 1 when on the first contract
                      (year_signed == draft_year) and that value in 1..4, else 0.
  BLOCK 4 -- Vegas win totals (source: data/raw/vegas/win_totals.parquet, OPTIONAL;
             joined on the player's season-t ADP team). A preseason line.
    vegas_wins        team's season-t Vegas win total.
    vegas_wins_delta  vegas_wins minus the team's t-1 actual REG win count
                      (from data/raw/context/games.csv).
    Both columns are all-null (with a printed note) when win_totals.parquet is
    absent; the join activates automatically once the file exists.

Leakage: BLOCK 1 uses t-1 only; BLOCK 2 uses t-1/t-2; BLOCK 3 filters
year_signed <= t; BLOCK 4 is a preseason line vs a t-1 actual. Never season t.

Quirks:
  * snap_counts_2012.parquet is empty (nflverse snap data has content 2013+),
    so target seasons 2011-2013 get null snap features (t-1 in 2010-2012).
  * pfr_player_id -> gsis via db_playerids.csv; fantasy-position match rate is
    > 99% every season with data (printed in main()).
  * 2025 team volume / season-level files trade players to one recent_team; the
    snap block instead sums a player's weekly rows across all teams they played.

Build: uv run python -m bff.situation_features

Raw data fetch commands (nflverse releases; run from repo root):
    for y in $(seq 2012 2025); do curl -sL -o data/raw/snaps/snap_counts_$y.parquet \\
      https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_$y.parquet; done
    for y in $(seq 2010 2025); do curl -sL -o data/raw/injuries/injuries_$y.parquet \\
      https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_$y.parquet; done
    curl -sL -o data/raw/contracts/historical_contracts.parquet \\
      https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.parquet
    # win_totals.parquet is created separately (Vegas lines); BLOCK 4 stays null until it exists.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bff.context_data import norm_team

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
SNAPS = RAW / "snaps"
INJURIES = RAW / "injuries"
CONTRACTS_FILE = RAW / "contracts" / "historical_contracts.parquet"
VEGAS_FILE = RAW / "vegas" / "win_totals.parquet"
GAMES_FILE = RAW / "context" / "games.csv"

SEASONS = range(2011, 2027)  # target seasons t
FANTASY_POS = ["QB", "RB", "WR", "TE"]

MIN_GAMES_L4F4 = 6  # snap_share_l4f4 null below this (mirrors opp velocity guard)
MIN_GAMES_MAX4 = 4  # snap_share_max4 null below this

# Game-status designations that count as an injury listing. The only era-stable
# subset: "Probable" existed 2010-2015 only, and null-status practice-only rows
# explode after 2015 (see BLOCK 2 docstring). 2024's stray "Note" status is
# deliberately excluded.
QDO = ["Questionable", "Doubtful", "Out"]

# Reserve-list roster statuses that count as injury weeks (IR players drop off
# the game-status report). RES/PUP only -- INA is 2020+-only and includes
# healthy scratches (see BLOCK 2 docstring).
RESERVE_STATUS = ["RES", "PUP"]

# Soft-tissue keywords (lowercase, substring match on report/practice primary injury).
SOFT_TISSUE = [
    "hamstring", "groin", "calf", "quad", "quadricep", "achilles",
    "soft tissue", "adductor", "hip flexor", "oblique",
]


# --------------------------------------------------------------------------- pool
def build_pool() -> pl.DataFrame:
    """Season-t ADP pool: one row per (season, gsis_id), QB/RB/WR/TE, 2011-2026."""
    adp = pl.read_parquet(PROC / "adp.parquet")
    pool = (
        adp.filter(
            pl.col("season").is_between(min(SEASONS), max(SEASONS))
            & pl.col("position").is_in(FANTASY_POS)
            & pl.col("gsis_id").is_not_null()
        )
        .with_columns(pl.col("season").cast(pl.Int32))
        .with_columns(
            pl.when(pl.col("team").is_null() | (pl.col("team") == "FA"))
            .then(None)
            .otherwise(norm_team("team"))
            .alias("team")
        )
        .select("season", "gsis_id", "name", "position", "team")
    )
    assert pool.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return pool


def player_ids() -> pl.DataFrame:
    """pfr_id -> gsis_id crosswalk, dedup on pfr_id (keep first)."""
    return (
        pl.read_csv(RAW / "db_playerids.csv", null_values=["NA"], infer_schema_length=10000)
        .select("pfr_id", "gsis_id")
        .filter(pl.col("pfr_id").is_not_null())
        .unique(subset=["pfr_id"], keep="first")
    )


# ------------------------------------------------------------------- BLOCK 1: snaps
def load_snaps() -> pl.DataFrame:
    """All REG offensive-snap rows with content, 2012-2025 (2012 file is empty)."""
    frames = []
    for y in range(2012, 2026):
        f = SNAPS / f"snap_counts_{y}.parquet"
        if not f.exists():
            continue
        df = pl.read_parquet(f).filter(pl.col("game_type") == "REG")
        if df.height == 0:
            continue
        frames.append(
            df.select(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                "pfr_player_id",
                "position",
                pl.col("offense_pct").cast(pl.Float64),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={"season": pl.Int32, "week": pl.Int32, "pfr_player_id": pl.String,
                    "position": pl.String, "offense_pct": pl.Float64}
        )
    return pl.concat(frames)


def snap_features(snaps: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """Per (season t, gsis_id) snap features (source season shifted t-1 -> t)."""
    mapped = snaps.join(
        ids, left_on="pfr_player_id", right_on="pfr_id", how="left"
    ).drop_nulls("gsis_id")
    pct = pl.col("offense_pct")
    agg = mapped.group_by("season", "gsis_id").agg(
        pl.len().alias("_g"),
        pct.mean().alias("snap_share"),
        (pct.sort_by("week").tail(4).mean() - pct.sort_by("week").head(4).mean())
        .alias("snap_share_l4f4"),
        pct.sort(descending=True).head(4).mean().alias("snap_share_max4"),
    )
    agg = agg.with_columns(
        pl.when(pl.col("_g") >= MIN_GAMES_L4F4).then(pl.col("snap_share_l4f4"))
        .otherwise(None).alias("snap_share_l4f4"),
        pl.when(pl.col("_g") >= MIN_GAMES_MAX4).then(pl.col("snap_share_max4"))
        .otherwise(None).alias("snap_share_max4"),
    )
    return agg.with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season")).select(
        "season", "gsis_id", "snap_share", "snap_share_l4f4", "snap_share_max4"
    )


def snap_match_report(snaps: pl.DataFrame, ids: pl.DataFrame) -> None:
    """Per source season, share of fantasy-position snap rows mapping to a gsis_id."""
    fp = snaps.filter(pl.col("position").is_in(FANTASY_POS))
    j = fp.join(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
    rep = (
        j.group_by("season")
        .agg(
            pl.len().alias("rows"),
            pl.col("gsis_id").is_not_null().mean().alias("match_rate"),
        )
        .sort("season")
    )
    print("\n== snap pfr->gsis match rate (fantasy-pos rows, source season) ==")
    for r in rep.to_dicts():
        flag = "  <-- BELOW 90%" if r["match_rate"] < 0.90 else ""
        print(f"  {r['season']} (-> target {r['season'] + 1}): rows {r['rows']:5d}  "
              f"match {r['match_rate']:6.2%}{flag}")


# ---------------------------------------------------------------- BLOCK 2: injuries
def load_injuries() -> pl.DataFrame:
    """All REG injury-report rows, 2010-2025 (has gsis_id directly)."""
    frames = []
    for y in range(2010, 2026):
        f = INJURIES / f"injuries_{y}.parquet"
        if not f.exists():
            continue
        df = pl.read_parquet(f).filter(pl.col("game_type") == "REG")
        frames.append(
            df.select(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                "gsis_id",
                "report_status",
                "report_primary_injury",
                "practice_primary_injury",
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={"season": pl.Int32, "week": pl.Int32, "gsis_id": pl.String,
                    "report_status": pl.String, "report_primary_injury": pl.String,
                    "practice_primary_injury": pl.String}
        )
    return pl.concat(frames).filter(pl.col("gsis_id").is_not_null())


def _soft_tissue_mask() -> pl.Expr:
    rp = pl.col("report_primary_injury").fill_null("").str.to_lowercase()
    pp = pl.col("practice_primary_injury").fill_null("").str.to_lowercase()
    mask = pl.lit(False)
    for kw in SOFT_TISSUE:
        mask = mask | rp.str.contains(kw, literal=True) | pp.str.contains(kw, literal=True)
    return mask


def _spread_to_targets(df: pl.DataFrame, valcol: str) -> pl.DataFrame:
    """Attribute each source-season count to target = source+1 (t-1) and source+2
    (t-2), then sum per (target, gsis_id). Distinct (season, week) counts across the
    two source seasons never collide, so the sum is the distinct l2y count."""
    y1 = df.with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))
    y2 = df.with_columns((pl.col("season") + 2).cast(pl.Int32).alias("season"))
    return (
        pl.concat([y1, y2])
        .filter(pl.col("season").is_between(min(SEASONS), max(SEASONS)))
        .group_by("season", "gsis_id")
        .agg(pl.col(valcol).sum())
    )


def load_reserve_weeks() -> pl.DataFrame:
    """Distinct (season, gsis_id, week) on a reserve list (RES/PUP), REG weeks,
    source seasons 2010-2025, from the weekly roster snapshots."""
    frames = []
    for y in range(2010, 2026):
        f = RAW / "context" / f"roster_weekly_{y}.parquet"
        if not f.exists():
            continue
        df = pl.read_parquet(f, columns=["season", "week", "gsis_id", "status", "game_type"])
        frames.append(
            df.filter(
                (pl.col("game_type") == "REG")
                & pl.col("status").is_in(RESERVE_STATUS)
                & pl.col("gsis_id").is_not_null()
            ).select(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                "gsis_id",
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={"season": pl.Int32, "week": pl.Int32, "gsis_id": pl.String}
        )
    return pl.concat(frames).unique()


def injury_features(inj: pl.DataFrame, reserve: pl.DataFrame) -> pl.DataFrame:
    """Per (season t, gsis_id): l2y listing/soft-tissue counts + recurrence flag.
    A listed week = QDO game status OR reserve-list roster status (see the
    BLOCK 2 docstring for why both rules are load-bearing)."""
    listed = inj.filter(pl.col("report_status").is_in(QDO))

    listed_weeks = pl.concat([
        listed.select("season", "gsis_id", "week"),
        reserve.select("season", "gsis_id", "week"),
    ]).unique()
    per_src_weeks = listed_weeks.group_by("season", "gsis_id").agg(
        pl.col("week").n_unique().alias("inj_weeks_listed_l2y")
    )
    per_src_soft = (
        listed.filter(_soft_tissue_mask())
        .group_by("season", "gsis_id")
        .agg(pl.col("week").n_unique().alias("inj_soft_tissue_l2y"))
    )
    weeks = _spread_to_targets(per_src_weeks, "inj_weeks_listed_l2y")
    soft = _spread_to_targets(per_src_soft, "inj_soft_tissue_l2y")

    # recurrence: same lowercased report_primary_injury present on QDO-listed
    # rows in both t-1 and t-2 (practice-only rows carry a primary injury with
    # null status in 2016+ only -- counting them is era-asymmetric)
    distinct_inj = (
        listed.filter(pl.col("report_primary_injury").is_not_null())
        .select(
            "season", "gsis_id",
            pl.col("report_primary_injury").str.to_lowercase().alias("inj_lower"),
        )
        .unique()
    )
    y1 = distinct_inj.with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))
    y2 = distinct_inj.with_columns((pl.col("season") + 2).cast(pl.Int32).alias("season"))
    recur = (
        y1.join(y2, on=["season", "gsis_id", "inj_lower"], how="inner")
        .filter(pl.col("season").is_between(min(SEASONS), max(SEASONS)))
        .select("season", "gsis_id")
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("inj_recurrence"))
    )

    return (
        weeks.join(soft, on=["season", "gsis_id"], how="full", coalesce=True)
        .join(recur, on=["season", "gsis_id"], how="full", coalesce=True)
    )


# --------------------------------------------------------------- BLOCK 3: contracts
def contract_features(pool_gsis: pl.DataFrame) -> pl.DataFrame:
    """Per (season t, gsis_id): most recent contract signed <= t, for pool players.
    Ties on year_signed -> max apy_cap_pct -> min otc_id (deterministic)."""
    c = (
        pl.read_parquet(CONTRACTS_FILE)
        .filter(pl.col("gsis_id").is_not_null())
        .join(pool_gsis, on="gsis_id", how="inner")
        .select(
            "gsis_id",
            pl.col("year_signed").cast(pl.Int32),
            pl.col("years").cast(pl.Int32),
            pl.col("apy_cap_pct").cast(pl.Float64),
            pl.col("draft_year").cast(pl.Int32),
            pl.col("otc_id").cast(pl.Int64),
        )
    )
    seasons = pl.DataFrame({"season": list(SEASONS)}, schema={"season": pl.Int32})
    sel = (
        c.join(seasons, how="cross")
        .filter(pl.col("year_signed") <= pl.col("season"))
        .sort(
            ["year_signed", "apy_cap_pct", "otc_id"],
            descending=[True, True, False],
            nulls_last=True,
        )
        .group_by("season", "gsis_id", maintain_order=True)
        .first()
    )
    rd = pl.col("season") - pl.col("draft_year") + 1
    return sel.with_columns(
        ((pl.col("year_signed") + pl.col("years")) == pl.col("season"))
        .fill_null(False).cast(pl.Int8).alias("contract_year"),
        pl.when((pl.col("year_signed") == pl.col("draft_year")) & rd.is_between(1, 4))
        .then(rd).otherwise(0).cast(pl.Int32).alias("rookie_deal_yr"),
    ).select("season", "gsis_id", "apy_cap_pct", "contract_year", "rookie_deal_yr")


# ------------------------------------------------------------------ BLOCK 4: vegas
def _team_actual_wins() -> pl.DataFrame:
    """Per (season, team) REG win count from games.csv (tie = 0.5)."""
    g = pl.read_csv(GAMES_FILE, infer_schema_length=10000).filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    home = g.select(
        "season",
        norm_team("home_team").alias("team"),
        pl.when(pl.col("home_score") > pl.col("away_score")).then(1.0)
        .when(pl.col("home_score") == pl.col("away_score")).then(0.5)
        .otherwise(0.0).alias("win"),
    )
    away = g.select(
        "season",
        norm_team("away_team").alias("team"),
        pl.when(pl.col("away_score") > pl.col("home_score")).then(1.0)
        .when(pl.col("away_score") == pl.col("home_score")).then(0.5)
        .otherwise(0.0).alias("win"),
    )
    return (
        pl.concat([home, away])
        .group_by("season", "team")
        .agg(pl.col("win").sum().alias("actual_wins"))
    )


def vegas_features(pool: pl.DataFrame) -> pl.DataFrame:
    """Per (season, gsis_id) Vegas columns via the player's season-t ADP team.
    Null (with a note) until data/raw/vegas/win_totals.parquet exists."""
    base = pool.select("season", "gsis_id")
    if not VEGAS_FILE.exists():
        print("\n[BLOCK 4] data/raw/vegas/win_totals.parquet not found -> "
              "vegas_wins / vegas_wins_delta emitted as all-null.")
        return base.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("vegas_wins"),
            pl.lit(None, dtype=pl.Float64).alias("vegas_wins_delta"),
        )
    wt = pl.read_parquet(VEGAS_FILE).select(
        pl.col("season").cast(pl.Int32),
        norm_team("team").alias("team"),
        pl.col("vegas_wins").cast(pl.Float64),
    )
    prev_wins = _team_actual_wins().select(
        (pl.col("season") + 1).cast(pl.Int32).alias("season"), "team",
        pl.col("actual_wins").alias("_prev_wins"),
    )
    team_veg = (
        wt.join(prev_wins, on=["season", "team"], how="left")
        .with_columns((pl.col("vegas_wins") - pl.col("_prev_wins")).alias("vegas_wins_delta"))
        .select("season", "team", "vegas_wins", "vegas_wins_delta")
    )
    return (
        pool.join(team_veg, on=["season", "team"], how="left")
        .select("season", "gsis_id", "vegas_wins", "vegas_wins_delta")
    )


# ----------------------------------------------------------------------- assemble
def build() -> pl.DataFrame:
    pool = build_pool()
    ids = player_ids()

    snaps = load_snaps()
    snap_feat = snap_features(snaps, ids)
    inj_feat = injury_features(load_injuries(), load_reserve_weeks())
    con_feat = contract_features(pool.select("gsis_id").unique())
    veg_feat = vegas_features(pool)

    out = (
        pool.join(snap_feat, on=["season", "gsis_id"], how="left")
        .join(inj_feat, on=["season", "gsis_id"], how="left")
        .join(con_feat, on=["season", "gsis_id"], how="left")
        .join(veg_feat, on=["season", "gsis_id"], how="left")
    )
    # Injury files cover every target season, so a pool player with no listing is a
    # true 0, not missing data. (If the injury source were absent the join above
    # would leave these null -- block stays null-safe.)
    out = out.with_columns(
        pl.col("inj_weeks_listed_l2y").fill_null(0).cast(pl.Int32),
        pl.col("inj_soft_tissue_l2y").fill_null(0).cast(pl.Int32),
        pl.col("inj_recurrence").fill_null(0).cast(pl.Int8),
    )
    out = out.sort("season", "gsis_id")
    assert out.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    assert out["season"].min() >= 2011 and out["season"].max() <= 2026
    return out


FEATURE_COLS = [
    "snap_share", "snap_share_l4f4", "snap_share_max4",
    "inj_weeks_listed_l2y", "inj_soft_tissue_l2y", "inj_recurrence",
    "apy_cap_pct", "contract_year", "rookie_deal_yr",
    "vegas_wins", "vegas_wins_delta",
]

BLOCKS = {
    "BLOCK 1 snaps": ["snap_share", "snap_share_l4f4", "snap_share_max4"],
    "BLOCK 2 injuries": ["inj_weeks_listed_l2y", "inj_soft_tissue_l2y", "inj_recurrence"],
    "BLOCK 3 contracts": ["apy_cap_pct", "contract_year", "rookie_deal_yr"],
    "BLOCK 4 vegas": ["vegas_wins", "vegas_wins_delta"],
}


def coverage_report(out: pl.DataFrame) -> None:
    for label, df in [
        ("2015-2025", out.filter(pl.col("season").is_between(2015, 2025))),
        ("2026", out.filter(pl.col("season") == 2026)),
    ]:
        print(f"\n== coverage {label} (n={df.height}) ==")
        for block, cols in BLOCKS.items():
            print(f"  {block}")
            for c in cols:
                nn = 1.0 - df[c].null_count() / max(df.height, 1)
                mean = df[c].cast(pl.Float64).mean()
                ms = f"{mean:9.4f}" if mean is not None else "      nan"
                print(f"    {c:22s} non-null {nn:6.1%}  mean {ms}")


def coverage_by_season(out: pl.DataFrame) -> None:
    print("\n== per-season non-null rate by block ==")
    rows = []
    for s in range(2011, 2027):
        df = out.filter(pl.col("season") == s)
        rec = {"season": s, "n": df.height}
        for block, cols in BLOCKS.items():
            key = block.split()[-1]  # snaps / injuries / contracts / vegas
            rec[key] = 1.0 - df[cols[0]].null_count() / max(df.height, 1)
        rows.append(rec)
    rep = pl.DataFrame(rows)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(rep.with_columns(
            [pl.col(c).round(3) for c in ("snaps", "injuries", "contracts", "vegas")]
        ))


def sanity(out: pl.DataFrame) -> None:
    def show(season, name_sub, cols):
        row = out.filter((pl.col("season") == season) & pl.col("name").str.contains(name_sub))
        print(f"\n{season} {name_sub}:")
        with pl.Config(tbl_cols=-1, tbl_width_chars=220):
            print(row.select(["name", "position", "team"] + cols))

    show(2026, "Saquon Barkley", ["snap_share", "snap_share_max4", "snap_share_l4f4",
                                  "apy_cap_pct", "contract_year", "rookie_deal_yr"])
    show(2025, "Ja'Marr Chase", ["snap_share", "apy_cap_pct", "contract_year"])
    show(2019, "Saquon Barkley", ["apy_cap_pct", "rookie_deal_yr", "contract_year"])
    show(2024, "Christian McCaffrey", ["inj_weeks_listed_l2y", "inj_soft_tissue_l2y",
                                       "inj_recurrence", "snap_share"])
    # IR-blindness regression check: CMC missed 13 games in 2024 (RES weeks 2-8,
    # 14-18) but had only 3 QDO-listed weeks; the union must push his 2026
    # l2y count well into double digits.
    show(2026, "Christian McCaffrey", ["inj_weeks_listed_l2y", "inj_soft_tissue_l2y",
                                       "inj_recurrence"])
    # 2026 rookie-deal coverage note: the 2026 draft class has null gsis_id in the
    # contracts source (ids not yet assigned), so those rows are dropped and the
    # 2026 draftees get NULL contract features. The 2025 draft class shows
    # rookie_deal_yr == 2 for the 2026 season (year 2 of a 4-year rookie deal).
    s26 = out.filter(pl.col("season") == 2026)
    print("\n2026 contract coverage note:")
    print(f"  null apy_cap_pct (= 2026 draft class, gsis not yet in contracts): "
          f"{s26.filter(pl.col('apy_cap_pct').is_null()).height}")
    print(f"  rookie_deal_yr==2 (2025 draft class, 2nd year): "
          f"{s26.filter(pl.col('rookie_deal_yr') == 2).height}")
    show(2026, "Jeanty", ["apy_cap_pct", "rookie_deal_yr", "contract_year"])


def main() -> None:
    out = build()
    out.write_parquet(PROC / "situation_features.parquet")
    print(f"wrote {out.height} rows, seasons {out['season'].min()}-{out['season'].max()}, "
          f"{len(FEATURE_COLS)} features")
    snap_match_report(load_snaps(), player_ids())
    coverage_report(out)
    coverage_by_season(out)
    sanity(out)


if __name__ == "__main__":
    main()
