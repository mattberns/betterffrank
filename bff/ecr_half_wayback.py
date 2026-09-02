"""Preseason HALF-PPR ECR 2018-2025 from Wayback captures of the half-PPR
draft cheatsheet. The half-PPR counterpart to `bff/ecr_wayback.py`.

    uv run python -m bff.ecr_half_wayback                 # fetch + parse + report
    uv run python -m bff.ecr_half_wayback --build         # write ecr_half.parquet

**Why this module exists.** The live `?year=` rankings pages (`bff/fp_ecr.py`)
are LATEST-REVISION boards, not frozen pre-draft snapshots: measured against
this repo's contemporaneous PPR captures, they predict outcomes +0.0201 mean
Spearman BETTER than the real preseason boards did (7 of 8 seasons; 2021
+0.0929), because busts get retroactively demoted. Le'Veon Bell sits at #2 on
the Sep-1-2018 capture and #15 on today's 2018 page. +0.0201 is the size of the
model's entire published edge over ADP, so revised boards cannot serve as an
anchor or a benchmark. Contemporaneous captures can.

**These captures self-certify, which is stronger than dating them by capture
time.** Each page embeds an `ecrData` JSON blob carrying its own metadata:

    "type": "Draft Half PPR", "ranking_type_name": "draft",
    "scoring": "HALF", "week": "0", "year": "<season>",
    "last_updated": "<M/DD>", "total_experts": <n>

`week: "0"` IS the draft board (weeks 1+ are in-season boards), and
`last_updated` is the board's own revision date. `check_leakage` compares that
date -- not the Wayback timestamp -- against the season's Week 1 kickoff, which
is the honest test: a capture taken after kickoff of a board last updated
before it is still a preseason board, and a capture taken before kickoff of a
board updated that morning is still preseason. Both edge cases occur here.

TWO capture families, dispatched on CONTENT not year (see `extract`):
2021-2025 embed `ecrData`; **2018-2020 do NOT** -- they are the
`table id="rank-data"` / `mpb-player-<fpid>` layout, which
`bff.ecr_wayback.parse_any` already parses for the PPR series and is reused
here rather than duplicated. The HTML family carries no embedded date and no
expert count, so those seasons can only be dated by their capture timestamp,
which is weaker. `rank_ecr` (JSON) and the displayed rank (HTML) are the same
quantity, so `ecr` follows the same convention as every other ECR source.

Capture set supplied by the user 2026-09-01 (see CAPTURES). Timestamps vs that
season's Week 1 kickoff -- read these before trusting a season:

    2018  cap 09-06 12:03Z  W1 09-06  same-day, pre-kickoff (Thu 20:20 ET)
    2019  cap 09-13 21:29Z  W1 09-05  **EIGHT DAYS AFTER KICKOFF**
    2020  cap 08-27 03:31Z  W1 09-10  clean
    2021  cap 09-07 00:22Z  W1 09-09  clean
    2022  cap 08-30 11:39Z  W1 09-08  clean
    2023  cap 09-04 22:50Z  W1 09-07  clean
    2024  cap 09-05 18:05Z  W1 09-05  same-day, pre-kickoff (14:05 ET)
    2025  cap 09-01 01:14Z  W1 09-04  clean

2019 is EXCLUDED and stays excluded: its capture is 8 days post-kickoff AND
it is in the HTML family, so it carries no board date to appeal to -- there is
no way to establish it as preseason. Week 1 2019 results were already known
when that page was archived. `--allow-postweek1` can force it in for
descriptive use; it must never anchor or benchmark anything.

Verified side-effect worth recording: the REVISED 2019-and-earlier boards had
looked anomalous (the live 2018 half board shows Antonio Brown #1 and only 6
RBs in its top 24, the wrong direction for half-PPR). The contemporaneous 2018
capture reads Gurley / Brown / D.Johnson -- RB-led, as half-PPR should be. The
anomaly was an artifact of the revisions, not of the format.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import polars as pl

from bff.ecr import norm_name
from bff.fp_ecr import join_gsis, week1

ROOT = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT / "data" / "raw" / "wayback_half"
OUT = ROOT / "data" / "processed" / "ecr_half.parquet"
PAGE = "https://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php"

CAPTURES = {
    2018: "20180906120301",
    2019: "20190913212924",
    2020: "20200827033122",
    2021: "20210907002238",
    2022: "20220830113943",
    2023: "20230904225001",
    2024: "20240905180525",
    2025: "20250901011445",
}

POSITIONS = ("QB", "RB", "WR", "TE")
MIN_PLAYERS = 200
_ECR_RE = re.compile(r"ecrData\s*=\s*(\{.*?\});?\s*(?:var|let|const|</script>)", re.S)


def fetch(season: int) -> Path | None:
    """Cache one capture. archive.org rate-limits hard (HTTP 429); a refused
    fetch leaves NO file so a later run retries rather than parsing a stub."""
    out = HTML_DIR / f"half_{season}.html"
    if out.exists() and out.stat().st_size > 50_000:
        return out
    import requests

    url = f"https://web.archive.org/web/{CAPTURES[season]}/{PAGE}"
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        print(f"  {season}: fetch failed {type(e).__name__}")
        return None
    if r.status_code != 200 or len(r.content) < 50_000 or "429 Too Many" in r.text:
        print(f"  {season}: HTTP {r.status_code}, {len(r.content)}B — not cached")
        return None
    out.write_text(r.text, encoding="utf-8")
    print(f"  {season}: cached {len(r.content)}B")
    return out


def capture_path(season: int) -> Path | None:
    """Accept .html or .htm — a browser "Save as" may use either."""
    for ext in (".html", ".htm"):
        p = HTML_DIR / f"half_{season}{ext}"
        if p.exists() and p.stat().st_size > 50_000:
            return p
    return None


def extract(season: int, path: Path) -> tuple[dict, pl.DataFrame]:
    """Two capture FAMILIES, dispatched on content, never on year:

      JSON (2021-2025): the page embeds `ecrData`, which self-certifies as the
      draft half-PPR board (`week:0`, `scoring:HALF`) and carries its own
      `last_updated`. Preferred: the metadata makes the leakage check exact.

      HTML TABLE (2018-2020): `table id="rank-data"` with
      `mpb-player-<fpid>` rows -- the SAME layout `bff.ecr_wayback` already
      parses for the PPR series, so that parser is reused rather than
      duplicated. No embedded metadata, so these can only be dated by their
      capture timestamp.
    """
    html = path.read_text(encoding="utf-8", errors="replace")
    if "ecrData" in html:
        meta, players = _parse_json(season, html)
        rows = [
            {
                "season": season,
                "player": p["player_name"],
                "position": p.get("player_position_id"),
                "team": p.get("player_team_id") or None,
                "ecr": float(p["rank_ecr"]),
                "fantasypros_id": str(p["player_id"]) if p.get("player_id") else None,
            }
            for p in players
            if p.get("rank_ecr") is not None
            and p.get("player_position_id") in POSITIONS
        ]
        return meta, pl.DataFrame(rows)

    from bff.ecr_wayback import parse_any

    df = parse_any(html, season)
    assert df.height >= MIN_PLAYERS, f"{season}: only {df.height} rows parsed"
    # `pos` carries the positional rank ("RB1"); strip it to the bare position
    df = df.with_columns(
        pl.col("pos").str.replace(r"\d+$", "").alias("position")
    ).filter(pl.col("position").is_in(POSITIONS))
    rows = df.select(
        pl.col("season").cast(pl.Int64),
        "player",
        "position",
        pl.col("team"),
        "ecr",
        pl.col("id").alias("fantasypros_id"),
    )
    meta = {
        "type": "html-table (no embedded metadata)",
        "ranking_type_name": "draft",
        "year": str(season),
        "week": "0",
        "scoring": "HALF",
        "last_updated": None,
        "count": df.height,
        "total_experts": None,
    }
    return meta, rows


def _parse_json(season: int, html: str) -> tuple[dict, list[dict]]:
    """Extract ecrData and assert it is the DRAFT HALF-PPR board for `season`."""
    m = _ECR_RE.search(html)
    assert m, f"{season}: ecrData present but not matched"
    d = json.loads(m.group(1))
    meta = {
        k: d.get(k)
        for k in ("type", "ranking_type_name", "year", "week", "scoring",
                  "position_id", "last_updated", "count", "total_experts")
    }
    # the payload identifies itself; refuse anything that is not the draft board
    assert meta["scoring"] == "HALF", f"{season}: scoring={meta['scoring']!r}"
    assert meta["week"] in ("0", 0), f"{season}: week={meta['week']!r} (0 = draft)"
    assert meta["ranking_type_name"] == "draft", f"{season}: {meta}"
    assert str(meta["year"]) == str(season), f"{season}: year={meta['year']!r}"
    players = d.get("players", [])
    assert len(players) >= MIN_PLAYERS, f"{season}: only {len(players)} players"
    return meta, players


def check_leakage(season: int, meta: dict) -> tuple[bool, str]:
    """Prefer the BOARD's own last_updated; fall back to the CAPTURE timestamp
    for the HTML-table family, which embeds no date. Returns (ok, message)."""
    w1 = week1(season)
    lu = meta.get("last_updated")
    if w1 is None:
        return True, "no Week 1 date — UNCHECKED"
    if not lu:
        ts = CAPTURES[season]
        cap = date(int(ts[0:4]), int(ts[4:6]), int(ts[6:8]))
        lead = (w1 - cap).days
        if lead < 0:
            return False, (f"no board date; CAPTURE {cap} is {-lead}d AFTER "
                           f"Week 1 ({w1}) — cannot establish preseason")
        tag = ("capture same-day as opener, "
               f"{ts[8:10]}:{ts[10:12]}Z vs a night kickoff" if lead == 0
               else f"capture {lead}d before W1")
        return True, f"no board date; {tag}"
    mm, dd = (int(x) for x in str(lu).split("/")[:2])
    # boards are dated within their own season; Jan/Feb would mean next year
    upd = date(season if mm >= 3 else season + 1, mm, dd)
    lead = (w1 - upd).days
    if lead < 0:
        return False, (f"board last_updated {upd} is {-lead}d AFTER Week 1 "
                       f"({w1}) — IN-SEASON, not preseason")
    tag = "same-day (pre-kickoff)" if lead == 0 else f"{lead}d before W1"
    return True, f"last_updated {upd}, {tag}"


def finalize(season: int, rows: pl.DataFrame) -> pl.DataFrame:
    """Normalized rows -> the ecr_half.parquet schema, same rank convention as
    every other ECR source here (displayed overall rank is `ecr`, then an
    ordinal re-rank with the ["season", "ecr", "player"] sort)."""
    df = rows.with_columns(
        pl.col("player").map_elements(norm_name, return_dtype=pl.Utf8).alias("norm_name")
    )
    df = df.sort(["season", "ecr", "player"]).with_columns(
        pl.col("ecr").rank("ordinal").over("season").cast(pl.Int32).alias("ecr_rank")
    )
    return join_gsis(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true",
                    help="write the parsed seasons into ecr_half.parquet")
    ap.add_argument("--allow-postweek1", action="store_true",
                    help="include a season whose BOARD postdates Week 1 "
                         "(not preseason — descriptive use only)")
    a = ap.parse_args()

    print("fetching captures (cached after first success):")
    frames, skipped = [], []
    for season in sorted(CAPTURES):
        p = capture_path(season) or fetch(season)
        if p is None:
            skipped.append((season, "no capture on disk (rate-limited?)"))
            continue
        meta, rows = extract(season, p)
        ok, msg = check_leakage(season, meta)
        flag = "" if ok else "   <-- EXCLUDED"
        if not ok and a.allow_postweek1:
            flag = "   <-- INCLUDED under --allow-postweek1 (NOT preseason)"
        exp = meta["total_experts"] or "?"
        print(f"  {season}: {rows.height:4d} players, {exp} experts, "
              f"{msg}{flag}")
        if ok or a.allow_postweek1:
            frames.append(finalize(season, rows))
        else:
            skipped.append((season, msg))

    if not frames:
        sys.exit("nothing parsed")
    new = pl.concat(frames)
    print("\nparsed preseason half-PPR ECR:")
    for r in (new.group_by("season").agg(
            pl.len().alias("n"),
            pl.col("gsis_id").is_not_null().mean().round(4).alias("g"))
            .sort("season").iter_rows(named=True)):
        t = new.filter((pl.col("season") == r["season"]) & (pl.col("ecr_rank") <= 150))
        print(f"  {r['season']}: n={r['n']:4d} gsis_all={r['g']:.1%} "
              f"gsis_top150={t['gsis_id'].is_not_null().mean():.1%}  "
              f"#1={new.filter((pl.col('season')==r['season'])&(pl.col('ecr_rank')==1))['player'][0]}")
    if skipped:
        print("\nNOT included:")
        for s, why in skipped:
            print(f"  {s}: {why}")

    if not a.build:
        print("\n(dry run — pass --build to write ecr_half.parquet)")
        return

    seasons = sorted(new["season"].unique().to_list())
    # PROVENANCE COLUMN, not optional. This table will hold a MIX: seasons
    # recovered from contemporaneous captures (preseason-clean) alongside
    # seasons still coming from the live ?year= pages (revision-contaminated,
    # +0.0201 of hindsight). A consumer must be able to tell them apart --
    # silently blending them is the failure this repo has already been bitten
    # by on the ADP side. 2026 from the live page IS clean, because 2026 is
    # still preseason as of the pull.
    new = new.with_columns(pl.lit("wayback_preseason").alias("source"))
    if OUT.exists():
        old = pl.read_parquet(OUT)
        if "source" not in old.columns:
            old = old.with_columns(
                pl.when(pl.col("season") == 2026)
                .then(pl.lit("live_preseason"))
                .otherwise(pl.lit("live_revised"))
                .alias("source")
            )
        new = new.select(old.columns).with_columns(
            [pl.col(c).cast(old.schema[c]) for c in old.columns])
        keep = old.filter(~pl.col("season").is_in(seasons))
        merged = pl.concat([keep, new]).sort(["season", "ecr_rank"])
        # quantify what the revised boards had been claiming
        print("\nreplacing REVISED rows with PRESEASON rows:")
        for s in seasons:
            a_ = old.filter(pl.col("season") == s).select(
                "gsis_id", pl.col("ecr_rank").alias("rev")).drop_nulls()
            b_ = new.filter(pl.col("season") == s).select(
                "gsis_id", pl.col("ecr_rank").alias("pre"), "player").drop_nulls()
            j = a_.join(b_, on="gsis_id", how="inner")
            if not j.height:
                print(f"  {s}: (no prior rows)")
                continue
            d = (j["rev"] - j["pre"]).abs()
            worst = j.with_columns((pl.col("rev") - pl.col("pre")).alias("m")).sort(
                pl.col("m").abs(), descending=True).head(1)
            w = worst.iter_rows(named=True).__next__()
            print(f"  {s}: n={j.height:3d} mean|move|={d.mean():5.1f} "
                  f"max={d.max():3d}  biggest: {w['player']} "
                  f"revised {w['rev']} vs preseason {w['pre']}")
    else:
        merged = new.sort(["season", "ecr_rank"])
    merged.write_parquet(OUT)
    print(f"\nwrote {OUT.relative_to(ROOT)} ({merged.height} rows)")


if __name__ == "__main__":
    main()
