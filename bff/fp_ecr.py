"""Live FantasyPros ECR (PPR cheatsheet) for the CURRENT preseason.

    https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php

Reuses the signed-in Playwright session from `bff.fp_adp` (same publisher, same
auth, same Cloudflare constraints -- see that module's docstring for the login
routes). Caches the raw board to `data/raw/fp_ecr/ppr_<season>.json`, then
surgically refreshes that season's rows inside `data/processed/ecr.parquet`.

**Why surgical instead of a full `bff.ecr` rebuild.** `bff/ecr.py` builds
2012-2020 from `data/raw/db_fpecr_wayback.parquet` and 2021-2025 from
`db_fpecr.parquet`, both git-ignored commercial archives that are ABSENT from
this checkout, so `bff.ecr` cannot run at all here. `ecr.parquet` is the
committed artifact and is the only copy of those seasons. This module therefore
replaces ONE season's rows and leaves every other season byte-identical, which
it asserts before writing. It is not a substitute for `bff.ecr`; if the raw
archives ever come back, rebuild through `bff.ecr` and re-apply this on top.

Rank convention matches `bff.ecr` exactly: the board's displayed overall RK is
the `ecr` value, and `ecr_rank` is an ordinal re-rank within the season with the
same `["season", "ecr", "player"]` sort, so ties break reproducibly.

One thing this gets that the CSV export never had: the live rows carry a real
`fp-id`, so 2026 resolves to gsis ids id-first (the
`FantasyPros_2026_Draft_ALL_Rankings.csv` export has no id column at all, and
2026 currently resolves by name only).

Leakage: a rankings board is a preseason artifact only while it IS preseason.
`check_preseason` asserts the pull date is strictly before that season's Week 1
kickoff and refuses to write otherwise -- pulling this page in-season would
write in-season information into a preseason feature.

    uv run python -m bff.fp_ecr --fetch            # cache the live board
    uv run python -m bff.fp_ecr --build            # splice into ecr.parquet
    uv run python -m bff.fp_ecr --fetch --build --season 2026
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from bff.ecr import norm_name
from bff.fp_adp import CROSSWALK, _new_context, _require_state

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fp_ecr"
ECR = ROOT / "data" / "processed" / "ecr.parquet"
GAMES = ROOT / "data" / "raw" / "context" / "games.csv"

URL = "https://www.fantasypros.com/nfl/rankings/{slug}.php?year={season}"
SLUGS = {"ppr": "ppr-cheatsheets", "half": "half-point-ppr-cheatsheets",
         "std": "cheatsheets"}
POSITIONS = ("QB", "RB", "WR", "TE")
MIN_ROWS = 150

EXTRACT_JS = r"""() => {
  const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
  const t = document.querySelector("#ranking-table");
  if (!t) return null;
  const headers = [...t.querySelectorAll("thead th")].map((x) => norm(x.innerText));
  const rows = [...t.querySelectorAll("tbody tr")]
    .filter((tr) => tr.querySelector("a.fp-player-link"))
    .map((tr) => {
      const a = tr.querySelector("a.fp-player-link");
      const m = (a.className || "").match(/fp-id-(\d+)/);
      return {
        fp_id: m ? m[1] : null,
        name: norm(a.getAttribute("fp-player-name") || a.innerText),
        cells: [...tr.querySelectorAll("td")].map((td) => norm(td.innerText)),
      };
    });
  return {headers, rows};
}"""

_TEAM_RE = re.compile(r"\(([A-Z]{2,3})\)\s*$")
_POS_RE = re.compile(r"^([A-Z]+)(\d+)?$")


def week1(season: int) -> date | None:
    if not GAMES.exists():
        return None
    w = (
        pl.read_csv(GAMES, infer_schema_length=None)
        .filter((pl.col("week") == 1) & (pl.col("game_type") == "REG")
                & (pl.col("season") == season))
        .select(pl.col("gameday").min())
    )
    return None if not w.height or w.item() is None else date.fromisoformat(w.item())


def check_preseason(season: int, pulled: date,
                    allow_revised: bool = False) -> None:
    """A rankings page is preseason ONLY while the season has not kicked off."""
    w1 = week1(season)
    if w1 is None:
        print(f"  no Week 1 date for {season} in games.csv — preseason check SKIPPED")
        return
    if pulled < w1:
        print(f"  LIVE board, preseason OK: pulled {pulled}, {season} Week 1 is "
              f"{w1} ({(w1 - pulled).days}d lead)")
        return
    # COMPLETED season: `?year=` serves the board as it stands TODAY, which is
    # NOT the frozen pre-draft consensus. Measured 2026-09-01 against the
    # contemporaneous Wayback captures this repo already holds:
    #   2018  Le'Veon Bell  stored #2  ->  live #15
    # Bell held out all of 2018. The Sep-1-2018 capture ranks him #2 (as the
    # market did); the demotion to #15 happened AFTER the draft window, once
    # the holdout became a season-long absence. Every other 2018 top-12 player
    # matches within one slot, and 2019/2020/2022 show no comparable drift
    # (top-12 spearman 0.958-1.000, busts still ranked high -- Barkley #1 in
    # 2019, Taylor #3 in 2022), so the revision is narrow but real.
    #
    # MEASURED, 2026-09-01. Spearman of each board's top 150 against actual
    # season PPR points, stored-preseason vs live-?year=, 2018-2025:
    #   2018 +0.0010  2019 +0.0159  2020 +0.0382  2021 +0.0929
    #   2022 +0.0081  2023 +0.0114  2024 +0.0027  2025 -0.0098
    #   mean +0.0201, positive in 7 of 8 seasons
    # The live boards score 0.32-0.53 in absolute terms, so they are NOT
    # post-season or rest-of-season rankings -- they are preseason-SHAPED
    # boards carrying post-draft revisions. (A hindsight ranking would sit
    # north of 0.8.) But the bias is upward: retroactively demoting a bust
    # makes the board look MORE accurate than the real preseason board was.
    #
    # Why that is disqualifying rather than merely untidy: +0.0201 is the SAME
    # SIZE as the model's entire published edge over ADP (+0.0201). Using
    # these as the ECR benchmark would inject a bias as large as the effect
    # being measured, and 2021 alone (+0.0929) is 4.6x the headline result. As
    # an anchor it leaks post-draft information directly. So a completed
    # season is fetched only under an explicit flag and must never feed a
    # scored anchor or baseline.
    assert allow_revised, (
        f"{season} is COMPLETE (Week 1 was {w1}); its ?year= page is the "
        f"LATEST-REVISION board, not a frozen pre-draft snapshot -- see the "
        f"Le'Veon Bell case in this function. Pass --allow-revised if you want "
        f"it for DESCRIPTIVE use only; it must not feed a scored anchor."
    )
    print(f"  REVISED board for completed season {season} — descriptive use "
          f"only, NOT preseason-clean (see check_preseason)")


def cmd_fetch(season: int, fmt: str, headless: bool) -> None:
    _require_state()
    from playwright.sync_api import sync_playwright

    RAW.mkdir(parents=True, exist_ok=True)
    url = URL.format(slug=SLUGS[fmt], season=season)
    with sync_playwright() as pw:
        ctx, close = _new_context(pw, headless=headless)
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(5000)
        prev, stable = -1, 0
        for _ in range(40):
            d = page.evaluate(EXTRACT_JS)
            n = len(d["rows"]) if d else 0
            stable = stable + 1 if n == prev else 0
            if stable >= 2:
                break
            prev = n
            page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            page.wait_for_timeout(500)
        d = page.evaluate(EXTRACT_JS)
        title = page.title()
        close()
    if not d or len(d["rows"]) < MIN_ROWS:
        sys.exit(f"got {0 if not d else len(d['rows'])} rows (< {MIN_ROWS}) — "
                 f"session gated or page changed; not cached")
    # the page title carries the season it is actually serving
    m = re.search(r"\b(20\d{2})\b", title)
    assert m and int(m.group(1)) == season, (
        f"page title says {m.group(1) if m else '?'}, expected {season}: {title!r}"
    )
    payload = {
        "source": "fantasypros",
        "kind": "ecr",
        "format": fmt,
        "season": season,
        "url": url,
        "title": title,
        "pulled": datetime.now(timezone.utc).date().isoformat(),
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headers": d["headers"],
        "n_rows": len(d["rows"]),
        "players": d["rows"],
    }
    out = RAW / f"{fmt}_{season}.json"
    out.write_text(json.dumps(payload))
    print(f"cached {len(d['rows'])} rows -> {out.relative_to(ROOT)}")
    print(f"  title: {title}")


def build_rows(season: int, fmt: str, allow_revised: bool = False) -> pl.DataFrame:
    src = RAW / f"{fmt}_{season}.json"
    if not src.exists():
        sys.exit(f"no cached board at {src.relative_to(ROOT)} — run --fetch")
    d = json.loads(src.read_text())
    check_preseason(season, date.fromisoformat(d["pulled"]), allow_revised)
    idx = {h.upper(): i for i, h in enumerate(d["headers"])}
    # header row carries leading blank/sentiment columns; the data cells start
    # at RK, so locate by offset from RK rather than trusting header indices
    off = idx.get("RK", 0)
    rows = []
    for r in d["players"]:
        c = r["cells"]
        if not c:
            continue
        try:
            rk = float(c[0])
        except ValueError:
            continue
        who = c[2] if len(c) > 2 else ""
        tm = _TEAM_RE.search(who)
        pos_cell = c[3] if len(c) > 3 else ""
        pm = _POS_RE.match(pos_cell)
        pos = pm.group(1) if pm else pos_cell
        if pos not in POSITIONS:
            continue
        rows.append(
            {
                "season": season,
                "player": r["name"],
                "position": pos,
                "team": tm.group(1) if tm else None,
                "ecr": rk,
                "fantasypros_id": r["fp_id"],
            }
        )
    df = pl.DataFrame(rows).with_columns(
        pl.col("player").map_elements(norm_name, return_dtype=pl.Utf8).alias("norm_name")
    )
    # identical convention to bff.ecr: sort by (season, ecr, player), ordinal rank
    df = df.sort(["season", "ecr", "player"]).with_columns(
        pl.col("ecr").rank("ordinal").over("season").cast(pl.Int32).alias("ecr_rank")
    )
    return join_gsis(df)


def join_gsis(df: pl.DataFrame) -> pl.DataFrame:
    """id-first, then the same name ladder bff.ecr uses."""
    ids = pl.read_csv(CROSSWALK, null_values=["NA"]).with_columns(
        pl.col("fantasypros_id").cast(pl.Utf8),
        pl.col("name").map_elements(norm_name, return_dtype=pl.Utf8).alias("norm_name"),
    )
    by_fp = (
        ids.filter(pl.col("fantasypros_id").is_not_null()
                   & pl.col("gsis_id").is_not_null())
        .select("fantasypros_id", pl.col("gsis_id").alias("g_fp"))
        .unique(subset=["fantasypros_id"], keep="first")
    )
    by_era = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .join(df.select("norm_name", "position", "season").unique(),
              on=["norm_name", "position"], how="inner")
        .filter(pl.col("draft_year").is_null()
                | (pl.col("draft_year") <= pl.col("season")))
        .group_by("norm_name", "position", "season")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("g_era"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "position", "season", "g_era")
    )
    by_n = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .group_by("norm_name")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("g_n"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "g_n")
    )
    out = (
        df.join(by_fp, on="fantasypros_id", how="left")
        .join(by_era, on=["norm_name", "position", "season"], how="left")
        .join(by_n, on="norm_name", how="left")
        .with_columns(pl.coalesce("g_fp", "g_era", "g_n").alias("gsis_id"))
        .drop("g_fp", "g_era", "g_n")
    )
    # bff.ecr's invariant: (season, gsis_id) unique, dedupe by best rank.
    #
    # Two DIFFERENT FantasyPros ids can resolve to one gsis_id through the name
    # fallbacks -- 2019 has "Mike Davis" as both RB (fpid 13943) and WR (fpid
    # 12133). CLAUDE.md records this exact case and the rule: dedupe at the id
    # level, keep the best rank, and do NOT position-gate the name fallback
    # (it legitimately rescues cross-position listings).
    #
    # Do the dedupe DIRECTLY. An earlier version built a keep-set and joined it
    # back on (season, gsis_id) -- the very key it had deduped by -- which
    # marks BOTH rows as keepable and is a silent no-op.
    id_rows = (
        out.filter(pl.col("gsis_id").is_not_null())
        .sort(["ecr", "player"])
        .unique(subset=["season", "gsis_id"], keep="first", maintain_order=True)
    )
    out = pl.concat([id_rows, out.filter(pl.col("gsis_id").is_null())])
    # dropping a row leaves a hole in ecr_rank, so re-rank AFTER the dedupe;
    # `ecr` (the board's displayed rank) is untouched, ecr_rank is ordinal.
    out = out.sort(["season", "ecr", "player"]).with_columns(
        pl.col("ecr").rank("ordinal").over("season").cast(pl.Int32).alias("ecr_rank")
    )
    return out.select(
        "season", "player", "norm_name", "position", "team",
        "ecr", "ecr_rank", "fantasypros_id", "gsis_id",
    )


def out_path(fmt: str) -> Path:
    """Only the PPR board is the model's anchor. Every other format goes to its
    own artifact — splicing half-PPR ranks into `ecr.parquet` would silently
    replace the anchor the whole pipeline reads with a different scoring
    format's ranking, which is unrecoverable here (see the module docstring:
    `bff.ecr` cannot rebuild in this checkout)."""
    return ECR if fmt == "ppr" else ECR.with_name(f"ecr_{fmt}.parquet")


def cmd_build(season: int, fmt: str, allow_revised: bool = False) -> None:
    new = build_rows(season, fmt, allow_revised)
    # PROVENANCE. A live ?year= pull of a COMPLETED season is a revised board
    # (post-draft demotions, +0.0201 of hindsight); a pull made while the
    # season is still preseason is clean. Downstream must be able to tell,
    # so the label travels with the rows -- see bff.ecr_half_wayback.
    _d = json.loads((RAW / f"{fmt}_{season}.json").read_text())
    _pulled = date.fromisoformat(_d["pulled"])
    _w1 = week1(season)
    new = new.with_columns(
        pl.lit("live_preseason" if (_w1 is None or _pulled < _w1)
               else "live_revised").alias("source")
    )
    dest = out_path(fmt)
    if not dest.exists():
        new.write_parquet(dest)
        print(f"\n{season} {fmt} ECR: created {dest.relative_to(ROOT)} "
              f"({new.height} rows, gsis "
              f"{new['gsis_id'].is_not_null().mean():.1%})")
        print("  top-12: " + ", ".join(
            f"{r['ecr_rank']}.{r['player']}"
            for r in new.head(12).iter_rows(named=True)))
        return
    old = pl.read_parquet(dest)
    prior = old.filter(pl.col("season") == season)
    others = old.filter(pl.col("season") != season)

    new = new.select(old.columns).with_columns(
        [pl.col(c).cast(old.schema[c]) for c in old.columns]
    )
    merged = pl.concat([others, new]).sort(["season", "ecr_rank"])

    # every other season must come through untouched
    assert merged.filter(pl.col("season") != season).equals(
        old.filter(pl.col("season") != season).sort(["season", "ecr_rank"])
    ), "non-target seasons changed — refusing to write"

    print(f"\n{season} ECR: {prior.height} rows -> {new.height} rows")
    g_old = prior["gsis_id"].is_not_null().mean() if prior.height else 0.0
    g_new = new["gsis_id"].is_not_null().mean()
    print(f"  gsis coverage {g_old:.1%} -> {g_new:.1%}"
          f"  (fantasypros_id present on {new['fantasypros_id'].is_not_null().mean():.1%})")

    j = (
        prior.select("gsis_id", pl.col("ecr_rank").alias("was"))
        .drop_nulls("gsis_id")
        .join(new.select("gsis_id", "player", pl.col("ecr_rank").alias("now"))
              .drop_nulls("gsis_id"), on="gsis_id", how="inner")
        .with_columns((pl.col("was") - pl.col("now")).alias("move"))
    )
    print(f"  top-12 now: " + ", ".join(
        f"{r['ecr_rank']}.{r['player']}" for r in new.head(12).iter_rows(named=True)))
    big = j.sort(pl.col("move").abs(), descending=True).head(10)
    print("  biggest moves (+ = rose since the stored board):")
    for r in big.iter_rows(named=True):
        print(f"    {r['player']:24s} {r['was']:3d} -> {r['now']:3d}  {r['move']:+d}")

    merged.write_parquet(dest)
    print(f"\nwrote {dest.relative_to(ROOT)} ({merged.height} rows)")
    print("  NOTE: preds/rankings are now stale — rerun bff.model --season "
          f"{season} (and the full chain if backtests matter).")


def _seasons(arg: str | None, default: int) -> tuple[int, ...]:
    if not arg:
        return (default,)
    out: list[int] = []
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--seasons", default=None,
                    help="e.g. 2018-2026 or 2018,2022 (overrides --season)")
    ap.add_argument("--format", default="ppr", choices=sorted(SLUGS))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--allow-revised", action="store_true",
                    help="permit COMPLETED seasons, whose boards carry "
                         "post-draft revisions (descriptive use only)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch a season already cached")
    a = ap.parse_args()
    seasons = _seasons(a.seasons, a.season)

    if a.fetch:
        for yr in seasons:
            dest = RAW / f"{a.format}_{yr}.json"
            if dest.exists() and not a.refresh:
                print(f"{a.format} {yr}: cached, skipping (--refresh to force)")
                continue
            try:
                cmd_fetch(yr, a.format, a.headless)
            except SystemExit as e:
                print(f"{a.format} {yr}: {e}")
            except Exception as e:
                print(f"{a.format} {yr}: FAILED {type(e).__name__}: {e}")
    if a.build:
        for yr in seasons:
            if not (RAW / f"{a.format}_{yr}.json").exists():
                continue
            cmd_build(yr, a.format, a.allow_revised)
    if not (a.fetch or a.build):
        ap.print_help()


if __name__ == "__main__":
    main()
