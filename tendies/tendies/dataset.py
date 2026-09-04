"""Assemble boards, the VORP curve, and the replayed training examples.

This is the layer that turns "1,070 rows of who-picked-whom" into "for each
pick, the set of players that were actually choosable and the state of the
board and the roster at that moment". Everything the model sees comes from
here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from . import attrs as attrs_mod
from . import board as board_mod
from . import league as league_mod
from . import rookies as rookies_mod
from . import vorp as vorp_mod
from .config import (
    ACTUALS_PARQUET, CONTEXT_RAW_DIR, CURVE_ADP_PARQUET, DEFAULT_ADP_PARQUET,
    ECR_HALF_PARQUET, ECR_PARQUET, ECR_SOURCE_BY_SEASON, ESPN_CACHE, FP_RAW_DIR,
    PRESEASON_SOURCES, PROCESSED, REALTIME_ADP_SEASONS, ROSTERS_WEEK1_PARQUET,
)
from .replay import Board


def seasons_available(raw_dir: Path = FP_RAW_DIR) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in raw_dir.glob("half_*.json"))


# Franchise relocations and publisher spellings, for the contemporaneity check
# only. ctx_rosters_week1 says JAX/LV/LAC/LAR where the boards say JAC/OAK/SD/
# STL; without this Jacksonville alone reads as a season-wide mismatch.
TEAM_ALIAS = {"JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LAR",
              "LA": "LAR", "ARZ": "ARI", "BLT": "BAL", "HST": "HOU", "CLV": "CLE"}

# 2018's half capture is dated 2018-09-06, the morning of kickoff, by which time
# Le'Veon Bell's holdout was public and priced: he sits at half rank 18 against
# PPR rank 2 on the Sep-1 capture the parent repo holds. That is a real
# preseason board disagreeing with an earlier one, not hindsight, and the
# contemporaneity check (0.997) confirms the board is of its season.
ANCHOR_EXEMPT = frozenset({2018})


def _read_ecr(path: Path) -> pl.DataFrame | None:
    """One normalisation point. The two series were written by different
    builders and disagree on integer width (ecr_half carries season as Int64,
    ecr as Int32), so normalise on read and let nothing downstream depend on
    which file a season came from."""
    if not Path(path).exists():
        return None
    df = pl.read_parquet(path)
    if "source" not in df.columns:
        df = df.with_columns(source=pl.lit(None, dtype=pl.Utf8))
    return df.with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("gsis_id").cast(pl.Utf8),
        pl.col("ecr_rank").cast(pl.Int32),
        pl.col("source").cast(pl.Utf8),
    )


def load_ecr(
    seasons: list[int] | None = None,
    half_path: Path = ECR_HALF_PARQUET,
    ppr_path: Path = ECR_PARQUET,
    pin: dict[int, str] = ECR_SOURCE_BY_SEASON,
    on_undeclared: str = "fail",
) -> tuple[pl.DataFrame | None, dict[int, str]]:
    """Expert consensus rank per season, from the series PINNED in config.

    The board this prices is half-PPR, so its expert ranks should be too:
    scored against a full-PPR ECR, the half-PPR 2026 board shows a mean rank
    gap of 10.8 where the matched half-PPR series shows 8.3, and the extra 2.5
    is a scoring-format artifact that would read as expert-vs-market edge.

    This function used to take half wherever half existed. That is why the
    2026-09-01 rebuild of ecr_half.parquet repriced every training season with
    no code change and no output (see ECR_SOURCE_BY_SEASON). It now resolves
    each season against the pin and raises rather than improvising:

    * a season absent from the pin raises (or falls back to ppr with a warning
      under ``on_undeclared="ppr"``, or is taken as pinned-half-if-present and
      the run marked tainted under ``"use"``);
    * a season pinned to half whose rows do not carry a preseason ``source``
      raises, because an admissible board and an available board are different
      things;
    * a pin pointing at a series with no rows for that season raises. A broken
      pin is a bug to fix, not an invitation to fall back.
    """
    if on_undeclared not in ("fail", "ppr", "use"):
        raise ValueError(f"on_undeclared must be fail|ppr|use, got {on_undeclared!r}")
    half, ppr = _read_ecr(half_path), _read_ecr(ppr_path)
    pool = [s for s in (half, ppr) if s is not None]
    if not pool:
        return None, {}
    if seasons is None:
        seasons = sorted({int(s) for df in pool for s in df["season"].unique().to_list()})

    frames, src = [], {}
    for season in sorted(int(s) for s in seasons):
        tag = pin.get(season)
        if tag is None:
            if on_undeclared == "fail":
                raise ValueError(
                    f"season {season} is not declared in ECR_SOURCE_BY_SEASON. "
                    "Run `python -m tendies ecr-check` and pin it explicitly; "
                    "an undeclared season is how the 2026-09-01 silent switch "
                    "happened."
                )
            if on_undeclared == "ppr":
                print(f"  WARNING: {season} undeclared in ECR_SOURCE_BY_SEASON, using ppr")
                tag = "ppr"
            else:
                tag = "half" if half is not None and half.filter(
                    pl.col("season") == season).height else "ppr"
                print(f"  TAINTED: {season} undeclared, taking {tag} anyway")
        df = half if tag == "half" else ppr
        sub = df.filter(pl.col("season") == season) if df is not None else None
        if sub is None or not sub.height:
            raise ValueError(
                f"season {season} is pinned to '{tag}' but that series has no rows "
                f"for it ({half_path if tag == 'half' else ppr_path})"
            )
        if tag == "half":
            got = set(sub["source"].unique().to_list())
            bad = {s for s in got if s not in PRESEASON_SOURCES}
            if bad:
                raise ValueError(
                    f"season {season} is pinned to 'half' but its rows carry "
                    f"source={sorted(bad)}, which is not preseason "
                    f"({sorted(PRESEASON_SOURCES)}). A revised board must never "
                    "price a season; see ECR_SOURCE_BY_SEASON."
                )
        frames.append(sub.select("season", "gsis_id", "ecr_rank"))
        src[season] = tag

    if src:
        by: dict[str, list[int]] = {}
        for s, t in sorted(src.items()):
            by.setdefault(t, []).append(s)
        print("  ECR source (pinned): " + " · ".join(
            f"{t} {_runs(v)}" for t, v in sorted(by.items())))
    return (pl.concat(frames) if frames else None), src


def check_ecr(
    half_path: Path = ECR_HALF_PARQUET,
    ppr_path: Path = ECR_PARQUET,
    rosters_path: Path = ROSTERS_WEEK1_PARQUET,
    seasons: list[int] | None = None,
) -> tuple[list[str], list[str]]:
    """Audit the half-PPR series against evidence, rather than trusting its own
    `source` label. Returns (failures, warnings).

    The workhorse is CONTEMPORANEITY. FantasyPros restates a historical page
    under current team identity, so a board pulled from the live ?year= URL
    lists today's rosters: measured 2026-09-01, the revised seasons agree with
    that season's week-1 roster on 0.4-5% of rows while a real capture agrees on
    95.5-99.7%. Nothing else separates the two cleanly — Spearman against the
    PPR twin reads 0.96-0.996 on BOTH, which is why it is only a warning here.
    """
    half, ppr = _read_ecr(half_path), _read_ecr(ppr_path)
    fails: list[str] = []
    warns: list[str] = []
    if half is None:
        return [f"no half series at {half_path}"], warns
    # An UNPLAYED season cannot be tested for contemporaneity and does not need
    # to be. The check asks "was this board written before or after the season
    # happened"; for a season that has not happened there is no after, and no
    # settled roster to compare against either (the 2026 week-1 snapshot is
    # itself a projection, and a third of the board is unsigned). Its
    # protection is that `live_preseason` is attested against kickoff upstream.
    played = set()
    if Path(ACTUALS_PARQUET).exists():
        played = {int(s) for s in
                  pl.read_parquet(ACTUALS_PARQUET, columns=["season"])["season"].unique()}
    rosters = (
        pl.read_parquet(rosters_path).select(
            pl.col("season").cast(pl.Int32), "gsis_id", pl.col("team").alias("team_ros")
        ).unique(subset=["season", "gsis_id"])
        if Path(rosters_path).exists() else None
    )
    if rosters is None:
        warns.append(f"no rosters at {rosters_path}; contemporaneity NOT checked")
    seasons = seasons or sorted(int(s) for s in half["season"].unique().to_list())

    print(f"{'season':>6} {'source':>18} {'rows':>5} {'gsis':>5} {'team_ok':>7} "
          f"{'top5<=15':>8} {'RB24':>7} {'rho':>5} {'FA':>5}")
    for season in seasons:
        hs = half.filter(pl.col("season") == season)
        if not hs.height:
            fails.append(f"{season}: no rows in the half series")
            continue
        src = ", ".join(sorted({s or "?" for s in hs["source"].unique().to_list()}))
        pins = ECR_SOURCE_BY_SEASON.get(season)
        unplayed = played and season not in played

        # A1 identity
        dup = hs.drop_nulls("gsis_id").group_by("gsis_id").len().filter(pl.col("len") > 1)
        if dup.height:
            fails.append(f"{season}: {dup.height} gsis_id values appear twice")
        ranks = hs["ecr_rank"].to_list()
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            warns.append(f"{season}: ecr_rank is not a dense 1..n ordinal")
        # A2/A3 coverage
        gsis = float(hs["gsis_id"].is_not_null().mean())
        if hs.height < 150:
            fails.append(f"{season}: {hs.height} rows, floor is 150")
        # Rookies have no gsis id until they play, so an unplayed season sits
        # lower by construction (2026 reads 0.894 against 0.98-1.00 elsewhere).
        gsis_floor = 0.85 if unplayed else 0.90
        if gsis < gsis_floor:
            fails.append(f"{season}: gsis coverage {gsis:.3f}, floor {gsis_floor:.2f}")
        # A4 contemporaneity
        team_ok = None
        if rosters is not None and "team" in hs.columns:
            j = hs.join(rosters, on=["season", "gsis_id"], how="inner")
            if j.height >= 50:
                team_ok = float(j.select(
                    (pl.col("team").replace(TEAM_ALIAS)
                     == pl.col("team_ros").replace(TEAM_ALIAS)).mean()
                ).item())
                if pins == "half" and team_ok < 0.90:
                    if unplayed:
                        warns.append(
                            f"{season}: team agreement {team_ok:.1%}, but the season is "
                            "unplayed so there is no settled roster to check against"
                        )
                    else:
                        fails.append(
                            f"{season}: pinned half but only {team_ok:.1%} of rows carry "
                            "that season's team; this is a live-revised board, not a capture"
                        )
            else:
                warns.append(f"{season}: only {j.height} roster matches, team check skipped")
        # A5/A6/warnings against the PPR twin
        ps = ppr.filter(pl.col("season") == season) if ppr is not None else None
        worst = rb_h = rb_p = rho = None
        if ps is not None and ps.height:
            top5 = [g for g in ps.sort("ecr_rank").head(5)["gsis_id"].to_list() if g]
            hit = hs.filter(pl.col("gsis_id").is_in(top5))
            worst = int(hit["ecr_rank"].max()) if hit.height else None
            if pins == "half" and worst is not None and worst > 15 and season not in ANCHOR_EXEMPT:
                fails.append(
                    f"{season}: a PPR top-5 player sits at half rank {worst} (floor 15)"
                )
            rb_h = hs.sort("ecr_rank").head(24).filter(pl.col("position") == "RB").height
            rb_p = ps.sort("ecr_rank").head(24).filter(pl.col("position") == "RB").height
            if pins == "half" and rb_h < rb_p - 2:
                fails.append(
                    f"{season}: {rb_h} RBs in the half top-24 against {rb_p} in PPR; "
                    "half-PPR should be MORE RB-friendly, not less"
                )
            shared = hs.join(ps.select("gsis_id", pl.col("ecr_rank").alias("r_ppr")),
                             on="gsis_id", how="inner").drop_nulls(["ecr_rank", "r_ppr"])
            if shared.height > 30:
                rho = float(np.corrcoef(
                    shared["ecr_rank"].rank().to_numpy(),
                    shared["r_ppr"].rank().to_numpy())[0, 1])
                if rho < 0.90:
                    warns.append(f"{season}: rho {rho:.3f} against the PPR twin")
        fa = float((hs["team"] == "FA").mean()) if "team" in hs.columns else None
        if fa is not None and fa > 0.20 and pins == "half" and not unplayed:
            warns.append(f"{season}: {fa:.0%} of rows list team FA")
        print(f"{season:>6} {src:>18} {hs.height:>5} {gsis:>5.3f} "
              f"{'—' if team_ok is None else f'{team_ok:>7.3f}'} "
              f"{'—' if worst is None else f'{worst:>8d}'} "
              f"{'—' if rb_h is None else f'{rb_h:>3d}/{rb_p:<3d}'} "
              f"{'—' if rho is None else f'{rho:>5.3f}'} "
              f"{'—' if fa is None else f'{fa:>5.2f}'}")
    return fails, warns


def _runs(seasons: list[int]) -> str:
    """[2020,2021,2022,2026] -> '2020-2022,2026'; [] -> 'none'."""
    if not seasons:
        return "none"      # a build of ONLY real-time seasons leaves the AVG side empty
    out, start, prev = [], seasons[0], seasons[0]
    for s in seasons[1:] + [None]:
        if s == prev + 1:
            prev = s
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = s
    return ",".join(out)


def build_boards(
    seasons: list[int],
    raw_dir: Path = FP_RAW_DIR,
    skill_path: Path = DEFAULT_ADP_PARQUET,
    ecr: pl.DataFrame | None = None,
    context_dir: Path = CONTEXT_RAW_DIR,
    realtime_seasons: frozenset[int] = REALTIME_ADP_SEASONS,
) -> pl.DataFrame:
    """Every season's board, stacked. K and D/ST included.

    A season in `realtime_seasons` prices off the board's REAL-TIME column
    instead of AVG (see config.REALTIME_ADP_SEASONS). Announced rather than
    silent: which series priced which season is the thing that went wrong on
    2026-09-01 with ECR, and an ADP switch is the same class of change.
    """
    skill = pl.read_parquet(skill_path) if Path(skill_path).exists() else None
    if ecr is None:
        ecr, _ = load_ecr(seasons)
    maps = rookies_mod.entry_year_map(context_dir, max(seasons))
    births = attrs_mod.birth_date_map(context_dir, max(seasons))
    rt = sorted(set(seasons) & set(realtime_seasons))
    if rt:
        print(f"  ADP source: real-time {_runs(rt)} · AVG "
              f"{_runs([s for s in seasons if s not in realtime_seasons])}")
    frames = [
        board_mod.build(s, raw_dir, skill, ecr, maps, births, ACTUALS_PARQUET,
                        realtime=s in realtime_seasons)
        for s in seasons
    ]
    boards = pl.concat(frames)
    rookies_mod.check(boards)
    attrs_mod.check(boards)
    return boards


def curve_history(
    actuals_path: Path = ACTUALS_PARQUET,
    adp_path: Path = CURVE_ADP_PARQUET,
    ecr: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Expert slot -> realised half-PPR points over the parent repo's full
    market history (2012-). Deeper than this league's seven boards on purpose:
    each slot then averages ~13 seasons instead of ~6, which is the difference
    between a usable curve and noise. Still `season < t` filtered downstream.

    ECR is joined on so the slots are indexed by EXPERT rank — the same
    quantity `attach_vorp` looks a player up by. Without it the curve silently
    falls back to market rank (`slot_key`'s fallback for unranked players), and
    the curve would answer a different question from the one being asked of it.
    `require_ecr=True` makes that failure loud instead.

    The expert ranks are MIXED-FORMAT by season, and deliberately: half-PPR for
    every season with a preseason capture (2018, 2020-2025) and full-PPR for the
    six whose only half board is a live revision (see ECR_SOURCE_BY_SEASON).
    Each season contributes its own slot -> points mapping, so the mix is
    heterogeneity across seasons, not a format error inside one; the target is
    `pts_half` throughout either way.
    """
    boards = pl.read_parquet(adp_path)
    if ecr is None:
        ecr, _ = load_ecr()
    if ecr is not None and "ecr_rank" not in boards.columns:
        boards = boards.join(
            ecr.drop_nulls("gsis_id").unique(subset=["season", "gsis_id"], keep="first"),
            on=["season", "gsis_id"], how="left",
        )
    return vorp_mod.market_slot_points(boards, pl.read_parquet(actuals_path))


def attach_vorp(
    boards: pl.DataFrame,
    actuals_path: Path = ACTUALS_PARQUET,
    ecr: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Walk-forward VORP: each season's board is priced by a curve built only
    from earlier seasons, so no board carries its own outcome."""
    hist = curve_history(actuals_path, ecr=ecr)
    vorp_mod.check_depth(hist, min(int(s) for s in boards["season"].unique().to_list()))
    out = []
    for season in sorted(boards["season"].unique().to_list()):
        curves, ses = vorp_mod.build_curve_full(hist, season)
        out.append(
            vorp_mod.attach_vorp(
                boards.filter(pl.col("season") == season), curves, ses
            )
        )
    return pl.concat(out)


def load_picks(league_id: int, processed: Path = PROCESSED) -> pl.DataFrame:
    return pl.read_parquet(processed / f"draft_picks_{league_id}.parquet")


def link(boards: pl.DataFrame, picks: pl.DataFrame) -> pl.DataFrame:
    """Attach board `pid` to every pick, season by season."""
    out = []
    for season in sorted(picks["season"].unique().to_list()):
        b = boards.filter(pl.col("season") == season)
        p = picks.filter(pl.col("season") == season)
        if b.is_empty():
            continue
        out.append(board_mod.link_picks(b, p))
    return pl.concat(out, how="diagonal_relaxed")


def league_configs(
    seasons: list[int], picks: pl.DataFrame, cache: Path | None = None
) -> dict[int, league_mod.LeagueConfig]:
    """Per-season rules, with `rounds` taken from the observed draft rather
    than from roster size (2019-20 ran 16 rounds, 2021+ run 15)."""
    cache = cache or (ESPN_CACHE / str(picks["league_id"][0]))
    rounds = {
        int(r["season"]): int(r["rounds"])
        for r in picks.group_by("season").agg(rounds=pl.col("round").max()).to_dicts()
    }
    known = sorted(int(p.stem.split("_")[1]) for p in cache.glob("league_*.json"))
    newest = max(known) if known else None
    cfgs = {}
    for s in seasons:
        fallback = None if s in known else newest
        try:
            cfgs[s] = league_mod.load(cache, s, rounds.get(s), fallback=fallback)
        except FileNotFoundError:
            continue
    return cfgs


def boards_by_season(boards: pl.DataFrame) -> dict[int, Board]:
    return {
        int(s): Board.from_frame(boards.filter(pl.col("season") == s))
        for s in boards["season"].unique().to_list()
    }


def build_all(league_id: int, seasons: list[int] | None = None,
              on_undeclared: str = "fail"):
    """Everything downstream needs: boards, linked picks, league configs."""
    seasons = seasons or seasons_available()
    ecr, _ = load_ecr(on_undeclared=on_undeclared)
    picks = load_picks(league_id)
    # prev_owner is a LEAGUE fact, so it needs the picks; everything else on the
    # board is a market fact and is built without them.
    boards = attrs_mod.attach_prev_owner(build_boards(seasons, ecr=ecr), picks)
    boards = attach_vorp(boards, ecr=ecr)
    linked = link(boards, picks)
    cfgs = league_configs(sorted(linked["season"].unique().to_list()), picks)
    return boards, linked, cfgs


def write_boards(boards: pl.DataFrame, processed: Path = PROCESSED) -> Path:
    """The one artifact downstream code reads. Committed on purpose: the raw
    FantasyPros captures are git-ignored and behind a login, so this file is
    what makes the pipeline reproducible without a re-scrape."""
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / "boards.parquet"
    boards.write_parquet(path)
    return path
