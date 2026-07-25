"""Curate the preseason player-prop boards into gsis-keyed tables.

Inputs:  data/raw/props/season_props.parquet  (bff.props_wayback)
         data/raw/db_playerids.csv            (crosswalk)
Outputs: data/processed/season_props.parquet  -- LONG, one row per
             (season, market, player): odds + implied/no-vig probability +
             board position, gsis_id where it could be resolved.
         data/processed/props_features.parquet -- WIDE, one row per
             (season, gsis_id): no-vig probability per market, zero-filled,
             plus coverage flags. This is the join-ready table.

NOTHING here is wired into bff/model.py. The columns are not in FEATURES and
not in CANDIDATE_BLOCKS; selection (tune window 2012-2017 only, via
bff/select_features.py) has not been run on them.

Board depth is wildly uneven across seasons (23 players priced for 2012
rushing yards, 122 for 2021 receiving yards), so raw implied probability is
NOT comparable season-to-season: a deeper board splits the same 100% across
more names. `novig_prob` divides each player's implied probability by the
board's total, which removes both the vig and the depth effect and makes a
season's board a proper distribution over its own field. `board_rank` /
`board_n` keep the raw shape available. A player absent from a board is a
market judgment ("not a contender"), so the wide table zero-fills and carries
`has_prop` / `n_markets` so a model can tell "priced at 0.4%" from "unpriced".

Build: uv run python -m bff.props
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "props" / "season_props.parquet"
RAW_IDS = ROOT / "data" / "raw" / "db_playerids.csv"
ROSTERS_W1 = PROC / "ctx_rosters_week1.parquet"
GAMES = ROOT / "data" / "raw" / "context" / "games.csv"
ADP = PROC / "adp.parquet"
ECR = PROC / "ecr.parquet"
OUT_LONG = PROC / "season_props.parquet"
OUT_WIDE = PROC / "props_features.parquet"

POSITIONS = ["QB", "RB", "WR", "TE"]

# Which crosswalk positions a market can plausibly price. Used only to
# disambiguate duplicate names, never to drop a matched player.
MARKET_POS = {
    "pass_yds": ["QB"],
    "pass_td": ["QB"],
    "rush_yds": ["RB", "QB", "WR"],
    "rush_td": ["RB", "QB", "WR"],
    "rec_yds": ["WR", "TE", "RB"],
    "rec_td": ["WR", "TE", "RB"],
    "mvp": POSITIONS,
    "oroy": POSITIONS,
    "comeback": POSITIONS,
}
MARKET_ORDER = ["mvp", "pass_yds", "pass_td", "rush_yds", "rush_td",
                "rec_yds", "rec_td", "oroy", "comeback"]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# sportsoddshistory spellings -> crosswalk names, post-norm_name. The site
# writes some names without punctuation ("AJ Green") and uses a few era
# nicknames; norm_name already strips ".'-" so only real differences appear.
_ALIASES = {
    "chad ochocinco": "chad johnson",
    "cadillac williams": "carnell williams",
    # the site's own misspellings, corrected to the crosswalk spelling. Only
    # QB/RB/WR/TE names are worth fixing; linemen and defenders reach these
    # boards via oroy/comeback and stay unmatched on purpose.
    "ezekiel elliot": "ezekiel elliott",
    "mitch trubisky": "mitchell trubisky",
    "alshon jeffrey": "alshon jeffery",
    "ryan matthews": "ryan mathews",
    "devante adams": "davante adams",
    "matt stafford": "matthew stafford",
    "robby anderson": "robbie anderson",
    "kyle juszcyk": "kyle juszczyk",
    "ezekial elliot": "ezekiel elliott",
    "cordarelle patterson": "cordarrelle patterson",
    "isiah crowell": "isaiah crowell",
    "elijiah moore": "elijah moore",
    "elijiah mitchell": "elijah mitchell",
    "deshaun jackson": "desean jackson",
    "joshua kelly": "joshua kelley",
    "darrell williams": "darrel williams",
    "kinbrell thompkins": "kenbrell thompkins",
    "derrius heywardbey": "darrius heywardbey",
    "irv smith j": "irv smith",
    "david johhnson": "david johnson",
    "marquez valdezscantling": "marquez valdesscantling",
    "jakobi myers": "jakobi meyers",
    "devin achane": "devon achane",
    "micahel crabtree": "michael crabtree",
    "michael pennix": "michael penix",
    "cameron ward": "cam ward",
}

# Board entries that are not players. They carry real probability mass and so
# STAY in the no-vig denominator (dropping them would inflate every real
# player's share), but they never resolve to a gsis_id.
_NON_PLAYERS = {"field", "the field", "any other player", "other"}
# NOTE: no alias is needed for suffix-only differences (Robert Griffin III,
# Odell Beckham Jr, Steve Smith Sr) -- norm_name already strips suffixes, which
# is also what CREATES the Steve Smith / Mike Williams style collisions that
# `resolve_by_season` exists to break.


def norm_name(name: str) -> str:
    """Fourth norm_name variant in the repo (adp/ecr/crosswalk have their own);
    deliberately NOT unified with them -- see CLAUDE.md fragility list."""
    s = name.lower()
    # the site disambiguates same-name players with a parenthetical team
    # ("Josh Allen (BUF)"); drop it and let the season identity do that job
    s = re.sub(r"\([^)]*\)", " ", s)
    # Curly quotes fold to the straight one FIRST. The cross-capture
    # reconciliation (2026-07-24) found the source serving "Ja’Marr Chase" with
    # U+2019 on some captures and "Ja'Marr Chase" with U+0027 on others; only
    # the straight form was being stripped, so a curly capture would have
    # produced "ja’marr chase", missed the crosswalk, and silently dropped a
    # top-5 player. Shipped captures happen to use U+0027, so this changes no
    # current output -- it removes the landmine.
    s = s.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    s = re.sub(r"[.'\-]", "", s)
    parts = [p for p in s.split() if p not in _SUFFIXES]
    out = " ".join(parts)
    return _ALIASES.get(out, out)


def implied_prob(odds: pl.Expr) -> pl.Expr:
    """American moneyline -> implied probability (vig included)."""
    return (
        pl.when(odds > 0)
        .then(100.0 / (odds.cast(pl.Float64) + 100.0))
        .otherwise(-odds.cast(pl.Float64) / (-odds.cast(pl.Float64) + 100.0))
    )


def load_crosswalk() -> pl.DataFrame:
    return (
        pl.read_csv(RAW_IDS, null_values=["NA"])
        .filter(pl.col("gsis_id").is_not_null())
        .with_columns(
            pl.col("name").map_elements(norm_name, return_dtype=pl.Utf8)
            .alias("norm_name")
        )
        .select("norm_name", "position", "gsis_id")
    )


def _unambiguous(df: pl.DataFrame, out: str) -> pl.DataFrame:
    return (
        df.group_by("season", "norm_name")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias(out))
        .filter(pl.col("n") == 1)
        .select("season", "norm_name", out)
    )


def season_identity() -> pl.DataFrame:
    """(season, norm_name) -> gsis_id for players who demonstrably existed in
    that season, from PRESEASON-safe tables only: that season's ADP and ECR
    boards first, then the week-1 roster snapshot (CLAUDE.md: the leakage-safe
    membership table). No outcome table is touched, so this resolves identity
    without importing season-t results.

    This is what breaks the collisions norm_name creates: several Mike
    Williamses, Steve Smith vs Steve Smith Sr, Alex Smith the QB vs Alex Smith
    the TE, David Johnson the ARI back vs David Johnson the PIT tight end.

    The FANTASY board is tried before the roster because that is the tiebreaker
    for names the roster cannot split: in 2016 both David Johnsons were on
    week-1 rosters, but only the Arizona back is on the ADP/ECR board, and a
    futures market pricing "David Johnson" for receiving touchdowns means that
    one. Roster-only names (deep-bench oroy longshots) still resolve at the
    second step."""
    fantasy, roster = [], []
    for path, col in ((ADP, "name"), (ECR, "player")):
        if path.exists():
            fantasy.append(
                pl.read_parquet(path)
                .filter(pl.col("gsis_id").is_not_null())
                .select(pl.col("season").cast(pl.Int32),
                        pl.col(col).alias("name"), "gsis_id")
            )
    if ROSTERS_W1.exists():
        roster.append(
            pl.read_parquet(ROSTERS_W1)
            .filter(pl.col("gsis_id").is_not_null())
            .select(pl.col("season").cast(pl.Int32), "name", "gsis_id")
        )
    empty = pl.DataFrame(
        schema={"season": pl.Int32, "norm_name": pl.Utf8, "gsis_id": pl.Utf8}
    )
    def prep(frames: list[pl.DataFrame]) -> pl.DataFrame:
        if not frames:
            return empty
        return pl.concat(frames).with_columns(
            pl.col("name").map_elements(norm_name, return_dtype=pl.Utf8)
            .alias("norm_name")
        )
    fan = _unambiguous(prep(fantasy), "gsis_fantasy")
    ros = _unambiguous(prep(roster), "gsis_roster")
    return (
        fan.join(ros, on=["season", "norm_name"], how="full", coalesce=True)
        .with_columns(
            pl.coalesce("gsis_fantasy", "gsis_roster").alias("gsis_season")
        )
        .select("season", "norm_name", "gsis_season")
    )


def attach_gsis(props: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """Four-step match, most specific first. The props source carries no id and
    no position, so identity comes from the season first, then from the
    market's plausible position set."""
    pos_map = pl.DataFrame(
        {"market": list(MARKET_POS), "position": list(MARKET_POS.values())}
    ).explode("position")

    # 0) (season, norm_name) among players who were in the league that season
    props = props.join(season_identity(), on=["season", "norm_name"], how="left")

    # 1) (norm_name, position) restricted to the market's plausible positions,
    #    accepted only when it resolves to a single gsis_id.
    by_np = (
        ids.group_by("norm_name", "position")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("g"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "position", "g")
    )
    cand = (
        props.select("season", "market", "norm_name").unique()
        .join(pos_map, on="market", how="left")
        .join(by_np, on=["norm_name", "position"], how="inner")
        .group_by("season", "market", "norm_name")
        .agg(pl.col("g").n_unique().alias("n"), pl.col("g").first().alias("gsis_np"))
        .filter(pl.col("n") == 1)
        .select("season", "market", "norm_name", "gsis_np")
    )
    props = props.join(cand, on=["season", "market", "norm_name"], how="left")

    # 2) (norm_name, position) over the four fantasy positions, unambiguous
    by_off = (
        ids.filter(pl.col("position").is_in(POSITIONS))
        .group_by("norm_name")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("gsis_off"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "gsis_off")
    )
    props = props.join(by_off, on="norm_name", how="left")

    # 3) norm_name alone, unambiguous across every position
    by_n = (
        ids.group_by("norm_name")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("gsis_n"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "gsis_n")
    )
    props = props.join(by_n, on="norm_name", how="left")

    # crosswalk position for the resolved id (reporting only)
    pos_of = (
        ids.filter(pl.col("position").is_in(POSITIONS))
        .sort("gsis_id", "position")  # keep="first" needs a defined order
        .unique(subset=["gsis_id"], keep="first", maintain_order=True)
        .select("gsis_id", pl.col("position").alias("cw_position"))
    )
    return (
        props.with_columns(
            pl.coalesce("gsis_season", "gsis_np", "gsis_off", "gsis_n")
            .alias("gsis_id"),
            pl.when(pl.col("gsis_season").is_not_null()).then(pl.lit("season"))
            .when(pl.col("gsis_np").is_not_null()).then(pl.lit("name_pos"))
            .when(pl.col("gsis_off").is_not_null()).then(pl.lit("name_off"))
            .when(pl.col("gsis_n").is_not_null()).then(pl.lit("name"))
            .otherwise(pl.lit("unmatched")).alias("match_via"),
        )
        .drop("gsis_season", "gsis_np", "gsis_off", "gsis_n")
        .join(pos_of, on="gsis_id", how="left")
    )


def dedupe_by_player(props: pl.DataFrame) -> pl.DataFrame:
    """One quote per player per board, keyed on gsis_id rather than name string.

    The SOURCE sometimes lists the same player twice under two spellings with
    two different prices — verified case: the 2015 comeback board carries
    "Micahel Crabtree" +6600 and "Michael Crabtree" +10000, a site data-entry
    error. `bff.props_wayback` cannot catch it (the name strings differ) and the
    `_ALIASES` fix is what surfaces it, by mapping both onto one gsis_id.

    Resolution: keep the SHORTEST price (highest implied probability), which is
    the conventional best-available-price reading, with a deterministic name
    tiebreak. This runs BEFORE novig_prob / board_rank / board_n are computed,
    so the duplicate is also removed from the board denominator — otherwise the
    phantom row would dilute every other player's no-vig share on that board
    (2015 comeback: board_n 51 -> 50).

    Unmatched rows keep their name-level identity; there is no id to dedupe on,
    and they never reach the wide table anyway."""
    keyed = props.filter(pl.col("gsis_id").is_not_null())
    rest = props.filter(pl.col("gsis_id").is_null())
    keyed = (
        keyed.sort(["season", "market", "gsis_id", "implied_prob", "player"],
                   descending=[False, False, False, True, False])
        .unique(subset=["season", "market", "gsis_id"], keep="first",
                maintain_order=True)
    )
    return pl.concat([keyed, rest], how="vertical")


def build_long(ids: pl.DataFrame) -> pl.DataFrame:
    raw = pl.read_parquet(RAW)
    props = raw.with_columns(
        pl.col("player").map_elements(norm_name, return_dtype=pl.Utf8)
        .alias("norm_name"),
        implied_prob(pl.col("odds_american")).alias("implied_prob"),
    )
    props = attach_gsis(props, ids).with_columns(
        pl.col("norm_name").is_in(list(_NON_PLAYERS)).alias("is_field")
    )
    props = dedupe_by_player(props)
    return (
        props.with_columns(
            (pl.col("implied_prob") / pl.col("implied_prob").sum()
             .over("season", "market")).alias("novig_prob"),
            pl.col("odds_american").rank("ordinal")
            .over("season", "market").cast(pl.Int32).alias("board_rank"),
            pl.len().over("season", "market").cast(pl.Int32).alias("board_n"),
        )
        .select("season", "market", "player", "norm_name", "gsis_id",
                "match_via", "is_field", "cw_position", "odds_american",
                "implied_prob", "novig_prob", "board_rank", "board_n",
                "as_of", "source_url")
        .sort("season", "market", "board_rank")
    )


def build_wide(long: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, gsis_id): no-vig probability per market, zero-filled
    where the player was not on that board, plus coverage columns.

    Zero-fill is the right default here (an unpriced player is one the market
    did not consider a contender), but it is NOT the same statement as a 0.4%
    quote, so `has_prop` / `n_markets` / `market_seasons` travel with it. A
    season whose board was never captured has `market_seasons` = 0 for that
    market, which is how a consumer tells a missing SOURCE from a missing
    PLAYER -- the distinction the vegas block never had."""
    matched = long.filter(pl.col("gsis_id").is_not_null())
    wide = (
        matched.group_by("season", "gsis_id", "market")
        .agg(pl.col("novig_prob").max())  # a player appears once per board
        .pivot(on="market", index=["season", "gsis_id"], values="novig_prob")
    )
    present = [m for m in MARKET_ORDER if m in wide.columns]
    wide = wide.with_columns(
        [pl.col(m).fill_null(0.0).alias(f"prop_{m}") for m in present]
    ).drop(present)
    cols = [f"prop_{m}" for m in present]
    wide = wide.with_columns(
        pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int32) for c in cols])
        .alias("prop_n_markets"),
    ).with_columns(
        (pl.col("prop_n_markets") > 0).alias("has_prop"),
    )
    # per-season source coverage: how many of the nine boards exist at all
    cover = (
        long.group_by("season").agg(
            pl.col("market").n_unique().cast(pl.Int32).alias("prop_market_seasons")
        )
    )
    return wide.join(cover, on="season", how="left").sort("season", "gsis_id")


def check_join(long: pl.DataFrame, wide: pl.DataFrame) -> None:
    """Every matched quote must survive into the wide table, and no two quotes
    may collapse into one cell.

    The raw fetch has a row census (`bff.props_wayback.audit_page`) but this
    step had none, so a bad alias mapping two different players onto one
    gsis_id, or a pivot that silently dropped a market, would not have been
    noticed. Checked as an identity, not a spot check: the number of distinct
    (season, market, gsis_id) triples in the long table must equal the number of
    non-zero prop cells in the wide table."""
    matched = long.filter(pl.col("gsis_id").is_not_null())
    triples = matched.select("season", "market", "gsis_id").unique().height
    cols = [c for c in wide.columns
            if c.startswith("prop_") and c not in
            ("prop_n_markets", "prop_market_seasons")]
    cells = int(sum((wide[c] > 0).sum() for c in cols))
    assert triples == cells, (
        f"join lost or merged quotes: {triples} distinct (season, market, gsis) "
        f"in the long table but {cells} non-zero cells in the wide table"
    )
    # a gsis_id must not receive two different players' quotes in one board
    dup = (
        matched.group_by("season", "market", "gsis_id")
        .agg(pl.col("player").n_unique().alias("n"), pl.col("player").unique())
        .filter(pl.col("n") > 1)
    )
    assert dup.height == 0, f"two players collapsed onto one gsis_id:\n{dup}"
    print(f"join check {triples} matched quotes -> {cells} wide cells, "
          f"no gsis collisions: OK")


def main() -> None:
    ids = load_crosswalk()
    long = build_long(ids)
    long.write_parquet(OUT_LONG)
    wide = build_wide(long)
    wide.write_parquet(OUT_WIDE)
    check_join(long, wide)

    print(f"wrote {OUT_LONG} ({long.height} rows)")
    print(f"wrote {OUT_WIDE} ({wide.height} rows, "
          f"{len([c for c in wide.columns if c.startswith('prop_')])} prop cols)\n")

    print("board depth and gsis match rate by season:")
    per = (
        long.group_by("season").agg(
            pl.col("market").n_unique().alias("markets"),
            pl.len().alias("quotes"),
            pl.col("gsis_id").is_not_null().sum().alias("matched"),
        ).sort("season")
    )
    for r in per.iter_rows(named=True):
        rate = r["matched"] / r["quotes"]
        flag = "" if rate >= 0.97 else "   <- CHECK"
        print(f"  {r['season']}  markets={r['markets']}/9  quotes={r['quotes']:>4}  "
              f"gsis={r['matched']:>4}/{r['quotes']:<4} ({rate:.1%}){flag}")

    print("\nhow each quote resolved to a gsis_id:")
    for r in (long.group_by("match_via").agg(pl.len().alias("n"))
              .sort("n", descending=True).iter_rows(named=True)):
        print(f"  {r['match_via']:<10} {r['n']:>4}  ({r['n']/long.height:.1%})")

    print("\nquotes by market:")
    bym = long.group_by("market").agg(
        pl.col("season").n_unique().alias("seasons"),
        pl.len().alias("quotes"),
        pl.col("gsis_id").is_null().sum().alias("unmatched"),
    ).sort("quotes", descending=True)
    for r in bym.iter_rows(named=True):
        print(f"  {r['market']:<10} seasons={r['seasons']:>2}  "
              f"quotes={r['quotes']:>4}  unmatched={r['unmatched']}")

    nf = long.filter(pl.col("is_field"))
    if nf.height:
        print(f"\nnon-player board entries kept in the no-vig denominator "
              f"({nf.height}):")
        for r in nf.iter_rows(named=True):
            print(f"  {r['season']} {r['market']:<9} {r['player']:<8} "
                  f"{r['odds_american']:+d}  {r['novig_prob']:.1%} of the board")

    miss = (
        long.filter(pl.col("gsis_id").is_null() & ~pl.col("is_field"))
        .group_by("player").agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    if miss.height:
        print(f"\nunmatched players ({miss.height} distinct):")
        for r in miss.head(25).iter_rows(named=True):
            print(f"  {r['player']:<28} x{r['n']}")

    # every board is a proper distribution over its own field
    tot = long.group_by("season", "market").agg(pl.col("novig_prob").sum())
    assert (tot["novig_prob"] - 1.0).abs().max() < 1e-9, tot
    print("\nsanity every (season, market) no-vig board sums to 1.0: OK")
    check_leakage(long)


def check_leakage(long: pl.DataFrame) -> None:
    """Assert every board's as-of date is no later than its season's Week 1
    kickoff DATE. In practice all fourteen land exactly ON opener day: the site
    posts its archived preseason board dated the morning of the Thursday opener
    and labels it "prior to the start of the season", which bff.props_wayback
    requires verbatim before accepting a capture. Zero regular-season games had
    been played, so this satisfies the walk-forward rule -- but it is the
    tightest margin of any input in the repo, so it is checked, not assumed."""
    if not GAMES.exists():
        print("games.csv absent — Week 1 leakage check SKIPPED")
        return
    w1 = (
        pl.read_csv(GAMES, infer_schema_length=None)
        .filter((pl.col("week") == 1) & (pl.col("game_type") == "REG"))
        .group_by("season")
        .agg(pl.col("gameday").min().alias("week1"))
        .with_columns(pl.col("week1").str.to_date(),
                      pl.col("season").cast(pl.Int32))
    )
    j = (
        long.group_by("season").agg(pl.col("as_of").max().alias("as_of"))
        .join(w1, on="season", how="inner")
        .with_columns(
            (pl.col("week1") - pl.col("as_of")).dt.total_days().alias("lead_days")
        )
        .sort("season")
    )
    bad = j.filter(pl.col("lead_days") < 0)
    assert bad.height == 0, f"board dated AFTER Week 1 kickoff:\n{bad}"
    same = j.filter(pl.col("lead_days") == 0).height
    print(f"leakage check {j.height} seasons: all boards <= Week 1 kickoff date "
          f"({same} dated exactly on opener day, "
          f"max lead {j['lead_days'].max()}d): OK")


if __name__ == "__main__":
    main()
