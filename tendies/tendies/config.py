"""Paths and defaults. Everything league-specific is a parameter, not a constant."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent                       # the tendies/ project root
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

AUTH_DIR = ROOT / ".auth"
COOKIE_FILE = AUTH_DIR / "espn_cookies.json"
PROFILE_DIR = AUTH_DIR / "chrome-profile"

# The league this was built for; every entry point takes --league-id, so other
# leagues need no code change.
DEFAULT_LEAGUE_ID = 20482930
DEFAULT_FIRST_SEASON = 2018
DEFAULT_LAST_SEASON = 2026

# Market ADP boards, built by the parent betterffrank repo (bff/fp_adp.py):
# FantasyPros, which replaced FantasyFootballCalculator as the ADP source
# because FFC's boards are pruned at the publisher (its 2022 half-PPR board is
# 117 players deep and omits Austin Ekeler outright). FantasyPros half-PPR
# runs 246-426 deep per season, so it covers every pick in a 10-team draft.
# Required columns: season, name, position, adp, adp_rank (gsis_id when present).
_PROCESSED = ROOT.parent / "data" / "processed"
_RAW = ROOT.parent / "data" / "raw"

# Raw FantasyPros captures. Read directly (not through the parent's curated
# parquet) because that build drops K and D/ST, which are 13% of this league's
# picks. See board.py.
FP_RAW_DIR = _RAW / "fp_adp"
ACTUALS_PARQUET = _PROCESSED / "actuals.parquet"
# RAW nflverse WEEKLY stats, player_stats_<season>.parquet (2025 is a differently
# named special case, same as in the parent repo). Read only by streaming.py,
# which needs week-by-week scores that the season-total actuals cannot supply.
STATS_RAW_DIR = _RAW / "stats"
# Week-1 roster snapshot, used only by `ecr-check` to test whether a board
# lists that season's teams or today's. See dataset.check_ecr.
ROSTERS_WEEK1_PARQUET = _PROCESSED / "ctx_rosters_week1.parquet"

# RAW nflverse rosters, roster_<season>.parquet, 2010-2026. The processed
# ctx_rosters* tables drop the bio block; these keep `entry_year`, which is the
# only source that resolves rookie status for the CURRENT class (see
# rookies.py for why gsis ids and FantasyPros ids both fail there).
CONTEXT_RAW_DIR = _RAW / "context"

# Expert consensus rank. The board is half-PPR, so the expert ranks that price
# it should be half-PPR too -- comparing a half-PPR ADP against a full-PPR ECR
# manufactures an "edge" everywhere receptions matter, which is most of the
# board.
ECR_HALF_PARQUET = _PROCESSED / "ecr_half.parquet"
ECR_PARQUET = _PROCESSED / "ecr.parquet"

# WHICH SERIES PRICES WHICH SEASON IS A PIN, NOT A PREFERENCE.
#
# It used to be "prefer half wherever it exists". On 2026-09-01 at 09:28
# ecr_half.parquet was rebuilt from 2026-only to 2013-2026 and every training
# season silently repriced -- walk-forward top-1 0.2053 -> 0.2086, 535 of 906
# picks moved -- with no code change, nothing printed, and docs/index.html
# re-rendered on top of it 39 seconds later. `load_ecr` now REFUSES an
# undeclared season, so a parquet gaining rows can never reroute the model
# again.
#
# A season pinned to "half" must ALSO carry a preseason `source` in the parquet
# (PRESEASON_SOURCES below); the pin says which series, the data says whether
# that series is admissible, and both have to agree.
#
# Why 2012-2017 and 2019 stay on ppr: their only half boards come from
# FantasyPros' live ?year= pages, which serve the LATEST-REVISION board rather
# than a frozen pre-draft snapshot. bff/fp_ecr.check_preseason measures those
# at +0.0201 mean Spearman against outcomes (7 of 8 seasons, 2021 alone
# +0.0929) -- the same size as the parent model's entire published edge over
# ADP -- and their team columns give it away: 0.004-0.05 agreement with
# ctx_rosters_week1 where a real capture scores 0.955-0.997. The ppr series for
# those seasons is preseason-attested, so this trades a bounded format error
# for the removal of an unbounded hindsight bias.
ECR_SOURCE_BY_SEASON: dict[int, str] = {
    **{s: "ppr" for s in range(2012, 2018)},   # half boards are live-revised
    2018: "half",                              # Wayback capture 2018-09-06, kickoff morning
    # 2019's only half capture is 2019-09-13 -- the page's own header says
    # "Week 2", eight days after the 2019-09-05 opener, so it has seen Week 1
    # results. Checked 2026-09-01: the archive holds nothing for that page in
    # August or early September 2019, so this is a permanent gap, not a fetch
    # to retry. 2019 is never scored (folds are 2020-2025); it is training data
    # whose share shrinks each fold, and the format gap it carries is a
    # 0.95-0.99 rank correlation rather than a different quantity.
    2019: "ppr",
    **{s: "half" for s in range(2020, 2027)},  # Wayback preseason 2020-25, live preseason 2026
}

# A half board is admissible only if the parquet says it was captured before
# kickoff. `source` is written by bff/fp_ecr.py; a file without that column
# cannot serve any half-pinned season at all.
#
# `live_preseason_filtered` is a live pull made before kickoff where the page's
# own filters/league settings were set by hand first (bff.fp_ecr
# --manual-filter). This gate is about HINDSIGHT -- whether the board has seen
# results -- and a filtered board has not; it is preseason by the same
# check_preseason assert as an unfiltered one. What it is not is identical in
# COMPOSITION to the default published consensus that prices the training
# seasons, which is a train/serve question the pin does not police: the raw
# payload records the row counts and control diff either side of the pause.
PRESEASON_SOURCES = frozenset(
    {"wayback_preseason", "live_preseason", "live_preseason_filtered"}
)

# Deep market history for the VORP curve only (2012-2026). Not the draft board:
# the curve needs many seasons per within-position slot, and this league has
# only seven.
#
# This is the last full-PPR input, and it stays that way until the half board
# is deep enough. Its job here is to supply the ROW UNIVERSE and a tiebreak
# among unranked players; the slots are ECR-indexed and the points are
# pts_half, so the format cost is a 0.95-0.99 rank correlation. The cost of
# swapping today is not: fp_adp_half covers 2018-2026, which would leave the
# fold-2020 curve TWO prior seasons instead of eight and move RB slot 1 from
# 226.8 points to 298.9. vorp.check_depth enforces the floors, so when the half
# board reaches 2014 this becomes a one-line change that the assert either
# admits or refuses.
CURVE_ADP_PARQUET = _PROCESSED / "fp_adp_ppr.parquet"

# WHICH SEASONS PRICE OFF REAL-TIME ADP. Same shape as ECR_SOURCE_BY_SEASON and
# for the same reason: a pin, so a board cannot change series without a diff.
#
# The FantasyPros board carries a trailing REAL-TIME column beside AVG. AVG is a
# mean pick over the whole sampling window; real-time is a dense rank over only
# the recent drafts, so it prices news that the season-long average is still
# averaging away. Measured 2026-09-02: Josh Jacobs AVG 48.0 (rank 49), real-time
# 96, half-PPR ECR 152 -- the experts and the recent market agree, and AVG is
# the outlier. Drafting off AVG here means reaching ~50 picks early.
#
# ONLY the live season, for two reasons. The column exists only on the 2025 and
# 2026 captures, so a blanket switch would reprice two seasons of a nine-season
# series and leave the rest on AVG; and 2025 is COMPLETE, so its page's "recent
# drafts" are drafts happening today, not in its preseason -- in-season
# information in a preseason board, which is the trap ECR_SOURCE_BY_SEASON
# exists to keep out. Training seasons stay on AVG permanently.
#
# Mirrors bff.fp_adp.REALTIME_SEASONS; the parent's `--build --realtime` writes
# the parquet, this pin covers the raw board the draft page is built from. Keep
# the two in step or the site's board and its gsis join come off different
# columns.
REALTIME_ADP_SEASONS = frozenset({2026})

# tendies' own outputs
ESPN_CACHE = RAW / "espn"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
WEB = PKG / "web"
DEFAULT_ADP_PARQUET = _PROCESSED / "fp_adp_half.parquet"

# Extra boards, each attached in its own `adp_*_<tag>` columns and consulted in
# THIS ORDER for adp_rank_filled. FantasyPros PPR is here for the half-vs-PPR
# comparison; the superseded FFC half board is here because FantasyPros' own
# historical pages drop the occasional real player (Austin Ekeler is on
# neither 2025 FantasyPros board, at ADP 90 on FFC's).
#
# The FFC entry pointed at `adp_half.parquet` until 2026-09-01, a path nothing
# has ever written; `bff/adp_ffc.py` writes `adp_ffc.parquet`. The board is
# still absent from disk, so this fallback is inert either way -- but now it is
# inert against the right filename, and cli.py says what that costs instead of
# printing "skipped".
DEFAULT_FALLBACK_BOARDS = [
    ("ppr", _PROCESSED / "fp_adp_ppr.parquet"),
    ("ffc", _PROCESSED / "adp_ffc.parquet"),
]
