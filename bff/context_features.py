"""Preseason context features, keyed (season, gsis_id), seasons 2011-2026.

One row per season-t ADP-pool player (QB/RB/WR/TE, from data/processed/adp.parquet).
Every feature is computable BEFORE season t kicks off. Timing rule per feature:

    team_missing                 season-t ADP team is null/FA (preseason fact).
    vacated_target_share         t-1 team volume x season-t roster membership.
    vacated_carry_share          same.
    vacated_rec_fp_share         same (receiving PPR points instead of raw volume).
    arriving_vet_usage           season-t roster arrivals (preseason) x their t-1 usage.
    draft_competition            April-t draft picks (preseason), excludes the player.
    qb_change                    expected season-t QB (preseason ADP) vs primary t-1 QB.
    qb_quality_delta             both QBs' season-(t-1) ppg_ppr (past outcomes).
    qb_rookie                    expected season-t QB has no t-1 ppg (preseason + past).
    qb_expected_missing          team has no QB in the season-t ADP pool (preseason).
    coach_change                 week-1 coach season t (preseason-known) vs season t-1.
    team_fp_prior(_z)            season-(t-1) team total PPR points (past outcome).
    team_pass_fp_share_prior     t-1 team receiving-PPR / total-PPR (past outcome).
    team_pass_rate_prior         t-1 team targets/(targets+carries) (past outcome).
    returning_target_competition t-1 target share of same-pos teammates still on the
                                 season-t roster (t-1 outcome x preseason roster).
    returning_carry_competition  same with carries.
    depth_rank_adp               rank of ADP among own team's same-position pool
                                 players in season t (preseason market fact).
    is_rookie                    drafted April t / draft_year == t (static fact).
    is_new_team                  season-t ADP team != most recent (t-3..t-1) team of
                                 record from past volume (preseason + past outcomes).

Caveats (weighed by audit):
  * Season-t roster membership (vacated_*, arriving_vet_usage, returning_*)
    uses ctx_rosters_week1 — each team's week-1 REG roster (statuses CUT/RET
    excluded; IR/PUP/2020-opt-outs kept as members, since those designations
    are usually preseason-known and priced into ADP). 2026 is the July
    offseason snapshot, slightly earlier-informed than backtest week-1
    snapshots; week-1 rosters also embed late-August cutdown news that summer
    ADP predates. Both are preseason facts, not outcome leakage.
  * t-1 membership (arrivals' prevros) and the 2026 draftee gsis backfill stay
    on ctx_rosters (season t-1 is concluded by preseason t; backfill is 2026-
    only). ctx_rosters is a last-observed-team table — do NOT use it for
    season-t membership.
  * 2025 team volume attributes traded players to a single recent_team.
  * position groups map FB/HB -> RB.

Sanity cases (verified in main(); results printed each run):
  1. 2025 NYJ Garrett Wilson: expected 2025 QB Justin Fields != primary 2024 QB
     Aaron Rodgers -> qb_change=1, qb_quality_delta = Fields'24ppg - Rodgers'24ppg.
  2. 2023 ATL Tyler Allgeier: Bijan Robinson drafted pick 8 ->
     draft_competition = 1/log2(1+8) ~= 0.3155; Bijan's own row excludes self.
  3. 2025 SEA Jaxon Smith-Njigba: DK Metcalf traded + Tyler Lockett released ->
     large vacated_target_share; Cooper Kupp arriving -> arriving_vet_usage > 0.

Build: uv run python -m bff.context_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

SEASONS = range(2011, 2027)
FANTASY_POS = ["QB", "RB", "WR", "TE"]
POS_GROUP_MAP = {"HB": "RB", "FB": "RB"}


def pos_group(col: str = "position") -> pl.Expr:
    return pl.col(col).replace(POS_GROUP_MAP)


def norm_name(col: str = "name") -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z ]", "")
        .str.replace_all(r"\b(jr|sr|ii|iii|iv|v)$", "")
        .str.strip_chars()
    )


def load_inputs() -> dict[str, pl.DataFrame]:
    adp = pl.read_parquet(PROC / "adp.parquet")
    rosters = pl.read_parquet(PROC / "ctx_rosters.parquet").with_columns(
        pos_group().alias("pos_group")
    )
    rosters_w1 = pl.read_parquet(PROC / "ctx_rosters_week1.parquet").with_columns(
        pos_group().alias("pos_group")
    )
    vol = pl.read_parquet(PROC / "ctx_team_volume.parquet").with_columns(
        pos_group().alias("pos_group")
    )
    draft = pl.read_parquet(PROC / "ctx_draft_picks.parquet").with_columns(
        pos_group().alias("pos_group")
    )
    teamqb = pl.read_parquet(PROC / "ctx_team_qb.parquet")
    coaches = pl.read_parquet(PROC / "ctx_week1_coaches.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    pids = pl.read_csv(RAW / "db_playerids.csv", null_values=["NA"], infer_schema_length=10000)
    return dict(
        adp=adp, rosters=rosters, rosters_w1=rosters_w1, vol=vol, draft=draft,
        teamqb=teamqb, coaches=coaches, actuals=actuals, pids=pids,
    )


def build_pool(adp: pl.DataFrame) -> pl.DataFrame:
    """Season-t ADP pool: one row per (season, gsis_id), QB/RB/WR/TE, 2011-2026."""
    pool = (
        adp.filter(
            pl.col("season").is_between(2011, 2026)
            & pl.col("position").is_in(FANTASY_POS)
            & pl.col("gsis_id").is_not_null()
        )
        .with_columns(
            pl.when(pl.col("team").is_null() | (pl.col("team") == "FA"))
            .then(None)
            .otherwise(pl.col("team"))
            .alias("team")
        )
        .with_columns(pl.col("team").is_null().cast(pl.Int8).alias("team_missing"))
        .select("season", "gsis_id", "name", "position", "team", "adp", "team_missing")
    )
    assert pool.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return pool


def player_prior_usage(vol: pl.DataFrame) -> pl.DataFrame:
    """Per (season, gsis_id): season-total target/carry share (summed across team
    stints for traded players) and team of record (stint with most PPR points)."""
    return (
        vol.sort("fp_ppr", descending=True)
        .group_by("season", "gsis_id")
        .agg(
            (pl.col("targets") / pl.col("team_targets")).sum().alias("target_share"),
            (pl.col("carries") / pl.col("team_carries")).sum().alias("carry_share"),
            pl.col("team").first().alias("vol_team"),
        )
    )


def vacated_by_team(vol: pl.DataFrame, rosters_w1: pl.DataFrame) -> pl.DataFrame:
    """Per (season t, team): share of t-1 team targets/carries/receiving PPR that
    belonged to players NOT on the team's season-t week-1 roster.
    Timing: t-1 outcomes x season-t week-1 membership (preseason)."""
    prev = vol.with_columns((pl.col("season") + 1).alias("season_t")).filter(
        pl.col("season_t").is_in(list(SEASONS))
    )
    ros = rosters_w1.select(
        pl.col("season").alias("season_t"), "team", "gsis_id",
        pl.lit(1).alias("on_roster_t"),
    )
    j = prev.join(ros, on=["season_t", "team", "gsis_id"], how="left").with_columns(
        pl.col("on_roster_t").is_null().cast(pl.Float64).alias("departed")
    )
    return j.group_by("season_t", "team").agg(
        ((pl.col("targets") * pl.col("departed")).sum() / pl.col("team_targets").first())
        .alias("vacated_target_share"),
        ((pl.col("carries") * pl.col("departed")).sum() / pl.col("team_carries").first())
        .alias("vacated_carry_share"),
        ((pl.col("rec_fp_ppr") * pl.col("departed")).sum() / pl.col("team_rec_fp_ppr").first())
        .alias("vacated_rec_fp_share"),
    ).rename({"season_t": "season"})


def arrivals(rosters_w1: pl.DataFrame, rosters: pl.DataFrame,
             usage: pl.DataFrame) -> pl.DataFrame:
    """Rows: (season t, team, pos_group, gsis_id, vet_usage) for players on the
    season-t WEEK-1 roster who were not on that team's t-1 roster, with prior
    t-1 NFL usage. vet_usage = t-1 carry share for RB group, t-1 target share
    for WR/TE. Timing: season-t week-1 membership (preseason) x t-1 roster
    (concluded by preseason t) x t-1 usage (past)."""
    cur = rosters_w1.filter(
        pl.col("season").is_in(list(SEASONS)) & pl.col("pos_group").is_in(FANTASY_POS)
    ).select("season", "team", "gsis_id", "pos_group")
    prevros = rosters.select(
        (pl.col("season") + 1).alias("season"), "team", "gsis_id", pl.lit(1).alias("was_here")
    )
    new = (
        cur.join(prevros, on=["season", "team", "gsis_id"], how="left")
        .filter(pl.col("was_here").is_null())
        .drop("was_here")
    )
    u = usage.select(
        (pl.col("season") + 1).alias("season"), "gsis_id", "target_share", "carry_share"
    )
    return (
        new.join(u, on=["season", "gsis_id"], how="inner")
        .with_columns(
            pl.when(pl.col("pos_group") == "RB")
            .then(pl.col("carry_share"))
            .otherwise(pl.col("target_share"))
            .alias("vet_usage")
        )
        .select("season", "team", "pos_group", "gsis_id", "vet_usage")
    )


def draft_rows(draft: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """April-t draft picks at fantasy positions, with 2026 gsis_ids backfilled from
    the 2026 roster snapshot (nflverse 2026 draft ids are placeholders).
    Timing: the April-t draft is a preseason fact for season t."""
    d = draft.filter(pl.col("pos_group").is_in(FANTASY_POS)).with_columns(
        norm_name().alias("nn")
    )
    r26 = (
        rosters.filter((pl.col("season") == 2026) & pl.col("pos_group").is_in(FANTASY_POS))
        .with_columns(norm_name().alias("nn"))
        .unique(subset=["team", "nn", "pos_group"], keep="first")
        .select("team", "nn", "pos_group", pl.col("gsis_id").alias("gsis_bf"))
    )
    d = d.join(
        r26.with_columns(pl.lit(2026).alias("season")),
        on=["season", "team", "nn", "pos_group"],
        how="left",
    ).with_columns(pl.coalesce("gsis_id", "gsis_bf").alias("gsis_id"))
    return d.select("season", "team", "pos_group", "pick", "gsis_id", "nn")


def build(inputs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    adp, rosters, vol = inputs["adp"], inputs["rosters"], inputs["vol"]
    rosters_w1 = inputs["rosters_w1"]
    pool = build_pool(adp)
    usage = player_prior_usage(vol)

    feat = pool.with_columns(norm_name().alias("nn"))

    # --- 1. vacated shares (season-t team; rookies naturally use the team they join)
    feat = feat.join(vacated_by_team(vol, rosters_w1), on=["season", "team"], how="left")

    # --- 2. arriving veteran competition (same position group, excluding self)
    arr = arrivals(rosters_w1, rosters, usage)
    arr_team = arr.group_by("season", "team", "pos_group").agg(
        pl.col("vet_usage").sum().alias("arr_sum")
    )
    arr_self = arr.select(
        "season", "team", "gsis_id", pl.col("vet_usage").alias("arr_self")
    )
    feat = (
        feat.with_columns(pos_group().alias("pos_group"))
        .join(arr_team, on=["season", "team", "pos_group"], how="left")
        .join(arr_self, on=["season", "team", "gsis_id"], how="left")
        .with_columns(
            (pl.col("arr_sum").fill_null(0.0) - pl.col("arr_self").fill_null(0.0))
            .clip(lower_bound=0.0)
            .alias("arriving_vet_usage")
        )
        .drop("arr_sum", "arr_self")
    )

    # --- 3. draft competition (highest April-t pick at position group, excl. self)
    dr = draft_rows(inputs["draft"], rosters)
    dc = (
        feat.select("season", "gsis_id", "nn", "team", "pos_group")
        .join(dr, on=["season", "team", "pos_group"], how="inner", suffix="_dr")
        .filter(
            (pl.col("gsis_id_dr").is_null() | (pl.col("gsis_id_dr") != pl.col("gsis_id")))
            & (pl.col("nn_dr") != pl.col("nn"))  # name fallback for unmatched ids
        )
        .group_by("season", "gsis_id")
        .agg(pl.col("pick").min().alias("best_pick"))
        .with_columns(
            (1.0 / (1.0 + pl.col("best_pick")).log(2)).alias("draft_competition")
        )
        .drop("best_pick")
    )
    feat = feat.join(dc, on=["season", "gsis_id"], how="left").with_columns(
        pl.col("draft_competition").fill_null(0.0)
    )

    # --- 4. QB change / quality delta (pass catchers)
    tq = inputs["teamqb"]
    exp_t = tq.select(
        "season", "team", "expected_qb_gsis", "expected_qb_name"
    )
    prim_prev = tq.select(
        (pl.col("season") + 1).alias("season"), "team",
        "primary_qb_gsis", "primary_qb_name",
    )
    qb_ppg_prev = inputs["actuals"].select(
        (pl.col("season") + 1).alias("season"),
        pl.col("gsis_id"),
        pl.col("ppg_ppr"),
    )
    feat = (
        feat.join(exp_t, on=["season", "team"], how="left")
        .join(prim_prev, on=["season", "team"], how="left")
        .join(
            qb_ppg_prev.rename({"gsis_id": "expected_qb_gsis", "ppg_ppr": "new_qb_ppg"}),
            on=["season", "expected_qb_gsis"], how="left",
        )
        .join(
            qb_ppg_prev.rename({"gsis_id": "primary_qb_gsis", "ppg_ppr": "old_qb_ppg"}),
            on=["season", "primary_qb_gsis"], how="left",
        )
    )
    is_pc = pl.col("position").is_in(["RB", "WR", "TE"])
    has_exp = pl.col("expected_qb_gsis").is_not_null()
    changed = has_exp & pl.col("primary_qb_gsis").is_not_null() & (
        pl.col("expected_qb_gsis") != pl.col("primary_qb_gsis")
    )
    feat = feat.with_columns(
        (is_pc & ~has_exp & pl.col("team").is_not_null())
        .cast(pl.Int8).alias("qb_expected_missing"),
        (is_pc & changed).cast(pl.Int8).alias("qb_change"),
        (is_pc & changed & pl.col("new_qb_ppg").is_null())
        .cast(pl.Int8).alias("qb_rookie"),
        pl.when(is_pc & changed & pl.col("new_qb_ppg").is_not_null())
        .then(pl.col("new_qb_ppg") - pl.col("old_qb_ppg").fill_null(0.0))
        .otherwise(0.0)
        .alias("qb_quality_delta"),
    )

    # --- 5. coach change
    co = inputs["coaches"]
    co_j = co.join(
        co.select((pl.col("season") + 1).alias("season"), "team",
                  pl.col("week1_coach").alias("prev_coach")),
        on=["season", "team"], how="inner",
    ).select(
        "season", "team",
        (pl.col("week1_coach") != pl.col("prev_coach")).cast(pl.Int8).alias("coach_change"),
    )
    feat = feat.join(co_j, on=["season", "team"], how="left")

    # --- 6. team context (t-1 team strength + returning same-pos competition)
    team_prev = (
        vol.group_by("season", "team")
        .agg(
            pl.col("team_fp_ppr").first(),
            pl.col("team_rec_fp_ppr").first(),
            pl.col("team_targets").first(),
            pl.col("team_carries").first(),
        )
        .with_columns(
            (pl.col("team_rec_fp_ppr") / pl.col("team_fp_ppr")).alias("team_pass_fp_share_prior"),
            (pl.col("team_targets") / (pl.col("team_targets") + pl.col("team_carries")))
            .alias("team_pass_rate_prior"),
            ((pl.col("team_fp_ppr") - pl.col("team_fp_ppr").mean().over("season"))
             / pl.col("team_fp_ppr").std().over("season")).alias("team_fp_prior_z"),
        )
        .select(
            (pl.col("season") + 1).alias("season"), "team",
            pl.col("team_fp_ppr").alias("team_fp_prior"),
            "team_pass_fp_share_prior", "team_pass_rate_prior", "team_fp_prior_z",
        )
    )
    feat = feat.join(team_prev, on=["season", "team"], how="left")

    # returning same-position-group teammates' t-1 usage on the season-t
    # week-1 roster
    ret = (
        vol.filter(pl.col("pos_group").is_in(FANTASY_POS))
        .with_columns((pl.col("season") + 1).alias("season_t"))
        .join(
            rosters_w1.select(pl.col("season").alias("season_t"), "team", "gsis_id",
                              pl.lit(1).alias("on_roster_t")),
            on=["season_t", "team", "gsis_id"], how="inner",
        )
        .select(
            pl.col("season_t").alias("season"), "team", "pos_group", "gsis_id",
            (pl.col("targets") / pl.col("team_targets")).alias("t_share"),
            (pl.col("carries") / pl.col("team_carries")).alias("c_share"),
        )
    )
    ret_team = ret.group_by("season", "team", "pos_group").agg(
        pl.col("t_share").sum().alias("ret_t_sum"),
        pl.col("c_share").sum().alias("ret_c_sum"),
    )
    ret_self = ret.group_by("season", "team", "gsis_id").agg(
        pl.col("t_share").sum().alias("self_t"),
        pl.col("c_share").sum().alias("self_c"),
    )
    feat = (
        feat.join(ret_team, on=["season", "team", "pos_group"], how="left")
        .join(ret_self, on=["season", "team", "gsis_id"], how="left")
        .with_columns(
            (pl.col("ret_t_sum").fill_null(0.0) - pl.col("self_t").fill_null(0.0))
            .clip(lower_bound=0.0).alias("returning_target_competition"),
            (pl.col("ret_c_sum").fill_null(0.0) - pl.col("self_c").fill_null(0.0))
            .clip(lower_bound=0.0).alias("returning_carry_competition"),
        )
        .drop("ret_t_sum", "ret_c_sum", "self_t", "self_c")
    )

    # --- 7. depth rank by ADP within (season, team, position) in the pool
    feat = feat.with_columns(
        pl.when(pl.col("team").is_not_null())
        .then(pl.col("adp").rank("ordinal").over("season", "team", "position"))
        .otherwise(None)
        .cast(pl.Int32)
        .alias("depth_rank_adp")
    )

    # --- 8a. is_rookie: draft_year == t (crosswalk) or picked in the April-t draft
    pids = (
        inputs["pids"]
        .filter(pl.col("gsis_id").is_not_null())
        .select("gsis_id", "draft_year")
        .unique(subset=["gsis_id"], keep="first")
    )
    drafted_t = dr.filter(pl.col("gsis_id").is_not_null()).select(
        "season", "gsis_id", pl.lit(1).alias("drafted_t")
    ).unique()
    feat = (
        feat.join(pids, on="gsis_id", how="left")
        .join(drafted_t, on=["season", "gsis_id"], how="left")
        .with_columns(
            ((pl.col("draft_year") == pl.col("season")) | (pl.col("drafted_t") == 1))
            .fill_null(False).cast(pl.Int8).alias("is_rookie")
        )
        .drop("draft_year", "drafted_t")
    )

    # --- 8b. is_new_team: season-t team differs from most recent (t-1..t-3) team
    # of record in past volume. Rookies (no prior volume) get 0; is_rookie carries it.
    prior_team = None
    for lag in (1, 2, 3):
        u = usage.select(
            (pl.col("season") + lag).alias("season"), "gsis_id",
            pl.col("vol_team").alias(f"vt{lag}"),
        )
        feat = feat.join(u, on=["season", "gsis_id"], how="left")
    feat = feat.with_columns(
        pl.coalesce("vt1", "vt2", "vt3").alias("prior_team")
    ).with_columns(
        (
            pl.col("prior_team").is_not_null()
            & pl.col("team").is_not_null()
            & (pl.col("prior_team") != pl.col("team"))
        ).cast(pl.Int8).alias("is_new_team")
    ).drop("vt1", "vt2", "vt3", "prior_team")

    # --- impute 0 + team_missing flag for team-dependent numerics
    zero_fill = [
        "vacated_target_share", "vacated_carry_share", "vacated_rec_fp_share",
        "arriving_vet_usage", "qb_quality_delta",
        "returning_target_competition", "returning_carry_competition",
        "team_fp_prior", "team_fp_prior_z", "team_pass_fp_share_prior",
        "team_pass_rate_prior",
    ]
    feat = feat.with_columns([pl.col(c).fill_null(0.0) for c in zero_fill])
    feat = feat.with_columns(pl.col("depth_rank_adp").fill_null(0).cast(pl.Int32))
    feat = feat.with_columns(
        pl.col("coach_change").fill_null(0).cast(pl.Int8),
        pl.col("qb_change").fill_null(0), pl.col("qb_rookie").fill_null(0),
        pl.col("qb_expected_missing").fill_null(0),
    )

    out_cols = [
        "season", "gsis_id", "name", "position", "team", "team_missing",
        "vacated_target_share", "vacated_carry_share", "vacated_rec_fp_share",
        "arriving_vet_usage", "draft_competition",
        "qb_change", "qb_quality_delta", "qb_rookie", "qb_expected_missing",
        "coach_change",
        "team_fp_prior", "team_fp_prior_z", "team_pass_fp_share_prior",
        "team_pass_rate_prior",
        "returning_target_competition", "returning_carry_competition",
        "depth_rank_adp", "is_rookie", "is_new_team",
    ]
    out = feat.select(out_cols).sort("season", "gsis_id")
    assert out.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return out


def coverage_report(out: pl.DataFrame) -> None:
    feats = [c for c in out.columns if c not in ("season", "gsis_id", "name", "position", "team")]
    for label, df in [
        ("2015-2025", out.filter(pl.col("season").is_between(2015, 2025))),
        ("2026", out.filter(pl.col("season") == 2026)),
    ]:
        print(f"\n== coverage {label} (n={df.height}) ==")
        for c in feats:
            nn = 1.0 - df[c].null_count() / df.height
            nz = (df[c].fill_null(0) != 0).sum() / df.height
            print(f"  {c:32s} non-null {nn:5.1%}  nonzero {nz:5.1%}  mean {df[c].cast(pl.Float64).mean():8.4f}")


def sanity(out: pl.DataFrame) -> None:
    def show(season, name_sub, cols):
        row = out.filter((pl.col("season") == season) & pl.col("name").str.contains(name_sub))
        print(f"\n{season} {name_sub}:")
        print(row.select(["name", "team"] + cols))

    show(2025, "Garrett Wilson", ["qb_change", "qb_quality_delta", "qb_rookie"])
    show(2023, "Allgeier", ["draft_competition", "vacated_carry_share"])
    show(2023, "Bijan", ["draft_competition", "is_rookie", "arriving_vet_usage"])
    show(2025, "Smith-Njigba", ["vacated_target_share", "arriving_vet_usage",
                                "returning_target_competition", "depth_rank_adp"])


def main() -> None:
    inputs = load_inputs()
    out = build(inputs)
    out.write_parquet(PROC / "context_features.parquet")
    print(f"wrote {out.height} rows, seasons {out['season'].min()}-{out['season'].max()}")
    coverage_report(out)
    sanity(out)


if __name__ == "__main__":
    main()
