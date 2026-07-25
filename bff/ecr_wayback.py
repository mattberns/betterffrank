"""One-time backfill: extract preseason FantasyPros PPR ECR for 2012-2020 from
Wayback Machine snapshots of the overall draft cheatsheet.

Not part of the routine build. Run once:

    uv run python -m bff.ecr_wayback

It fetches each archived page to data/raw/wayback_ppr/ppr_<year>.html (cached;
re-runs parse the saved file offline), parses the ranking table, and writes
data/raw/db_fpecr_wayback.parquet in the same schema the archive branch of
bff.ecr emits (player, pos, team, ecr, id, season). bff.ecr unions that parquet.

Two page FAMILIES, two parsers, dispatched on whether the rows carry an
mpb-player class:

  MODERN (2015-2020), `parse`: every player row is
  <tr class="...mpb-player-<fpid>...">, table id="data" (2015-2017) or
  "rank-data" (2018-2020, adds a WSID column). One position-relative parser.

  EARLY (2012-2014), `parse_early`: table id="data", plain <tr>, no row class.
  Three sub-layouts, distinguished by structure rather than year:
    - 2012: pos AND team live inside the name cell's
      <span class="tiny">(RB1, HOU, 8)</span>; no player id anywhere.
    - 2013: separate Pos column, team in <small> (MIN/5)</small>; no id.
    - 2014: same as 2013 plus <a class="fp-player-link fp-id-9398">, which is
      the SAME id space as mpb-player-<fpid> (verified: LeSean McCoy is 9398 in
      both the 2014 and 2015 captures).
  2012/2013 therefore emit id=None and resolve gsis through bff.ecr's
  (norm_name, position) and norm_name fallbacks -- the same path the 2026
  FantasyPros export uses. That is a real match-rate cost: same-name players
  that are ambiguous in the crosswalk (two Steve Smith WRs) drop out rather
  than mis-resolve.

The FP player id shares db_playerids.csv's fantasypros_id space, so id-bearing
rows resolve gsis via bff.ecr's by-fantasypros_id join. `ecr` is the displayed
overall rank (td[0]); only the downstream ordinal re-rank matters and overall
rank is tie-free, so ecr_rank is fully determined and matches FantasyPros' own
ordering.

Both parsers are ROW-SCOPED and `build` asserts parsed rows == player links in
the table (the props-scrape lesson: a document-wide regex silently eats rows,
and the rows it eats are the informative ones).
"""
import re
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT / "data" / "raw" / "wayback_ppr"
OUT = ROOT / "data" / "raw" / "db_fpecr_wayback.parquet"

# Wayback snapshots (preseason PPR overall cheatsheet), all late-Aug/early-Sep
# for cross-season timing consistency -- each is that season's LATEST capture
# before its Week 1 kickoff (the rule bff.ecr uses for the archive).
# 2013 is the one exception: the mid-Aug capture is the last one archived that
# preseason (checked via the CDX index), so the series runs Sep-dated except
# for that season.
URLS = {
    2012: "https://web.archive.org/web/20120901071106/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2013: "https://web.archive.org/web/20130818132524/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2014: "https://web.archive.org/web/20140901172718/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2015: "https://web.archive.org/web/20150901072312/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2016: "https://web.archive.org/web/20160901065804/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2017: "https://web.archive.org/web/20170830160039/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2018: "https://web.archive.org/web/20180901064837/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2019: "https://web.archive.org/web/20190903001905/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    2020: "https://web.archive.org/web/20200830104840/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
}

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def fetch(year: int) -> str:
    """Return the archived HTML for a season, caching to disk on first fetch."""
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    path = HTML_DIR / f"ppr_{year}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(URLS[year], headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    path.write_text(html, encoding="utf-8")
    return html


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def _extract_table(html: str, table_id: str) -> str | None:
    """Return the full <table id=...>...</table>, balancing nested <table> tags
    (newer pages nest tables, so a lazy .*?</table> truncates)."""
    open_m = re.search(r'<table[^>]*id="%s"' % table_id, html)
    if open_m is None:
        return None
    depth = 0
    start = open_m.start()
    for m in re.finditer(r"<table\b|</table>", html[start:]):
        depth += 1 if m.group(0) == "<table" else -1
        if depth == 0:
            return html[start : start + m.end()]
    return None


def parse(html: str, year: int) -> pl.DataFrame:
    """Parse the overall ranking table into archive-schema rows."""
    table = _extract_table(html, "data") or _extract_table(html, "rank-data")
    if table is None:
        raise RuntimeError(f"{year}: no ranking table found")

    recs = []
    for fpid, body in re.findall(
        r'<tr class="[^"]*mpb-player-(\d+)[^"]*"[^>]*>(.*?)</tr>', table, re.S
    ):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        if len(tds) < 3:
            continue
        # player cell = the td.player-label (both eras); the player link is
        # /players/<name>.php (older) or /rankings/<name>.php (newer top rows).
        pidx = next((i for i, td in enumerate(tds) if "player-label" in td), None)
        if pidx is None:
            pidx = next(
                (i for i, td in enumerate(tds)
                 if re.search(r"(?:players|rankings)/[^\"]*\.php", td)),
                None,
            )
        if pidx is None or pidx + 1 >= len(tds):
            continue
        cell = tds[pidx]
        # newer pages wrap the name in <span class="full-name">; older pages
        # put it as the anchor's direct text.
        m = re.search(r'<span class="full-name">([^<]+)</span>', cell) or re.search(
            r'(?:players|rankings)/[^"]*"[^>]*>([^<]+)</a>', cell
        )
        if m is None or not m.group(1).strip():
            continue
        name = m.group(1).strip()
        pos = _strip(tds[pidx + 1])  # e.g. "RB1"
        rank = _strip(tds[0])
        if not re.fullmatch(r"\d+", rank):
            continue
        small = re.search(r'<small[^>]*>([A-Za-z]{2,3})</small>', tds[pidx])
        team = small.group(1).upper() if small else None
        recs.append(
            {"player": name, "pos": pos, "team": team,
             "ecr": float(rank), "id": fpid, "season": year}
        )

    return pl.DataFrame(
        recs,
        schema={"player": pl.Utf8, "pos": pl.Utf8, "team": pl.Utf8,
                "ecr": pl.Float64, "id": pl.Utf8, "season": pl.Int32},
    )


_PLAYER_LINK = re.compile(r'href="[^"]*/nfl/players/[^"]*\.php')


def parse_early(html: str, year: int) -> pl.DataFrame:
    """Parse the 2012-2014 cheatsheet (plain <tr>, no mpb-player row class).

    Row-scoped by construction. Pos/team come from whichever of the three
    early layouts the row matches (see module docstring); `id` is the
    fp-id-<n> when the page carries one (2014 only) and None otherwise.
    """
    table = _extract_table(html, "data")
    if table is None:
        raise RuntimeError(f"{year}: no ranking table found")

    recs = []
    for body in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        if len(tds) < 3:
            continue
        rank = _strip(tds[0])
        if not re.fullmatch(r"\d+", rank):
            continue
        pidx = next((i for i, td in enumerate(tds) if _PLAYER_LINK.search(td)), None)
        if pidx is None:
            continue
        cell = tds[pidx]
        m = re.search(r'/nfl/players/[^"]*\.php[^"]*"[^>]*>([^<]+)</a>', cell)
        if m is None or not m.group(1).strip():
            continue
        name = m.group(1).strip()

        # 2012: "(RB1, HOU, 8)" inside the name cell carries BOTH pos and team.
        tiny = re.search(r"\(([A-Za-z]{1,3}\d*),\s*([A-Za-z]{2,4}),", cell)
        if tiny is not None:
            pos, team = tiny.group(1), tiny.group(2).upper()
        else:
            # 2013/2014: Pos is its own column, team is "<small> (MIN/5)</small>"
            if pidx + 1 >= len(tds):
                continue
            pos = _strip(tds[pidx + 1])
            # free agents wrap the team in an anchor ("(<a ...>FA</a>/5)"), so
            # strip tags inside <small> before reading the team code
            sm = re.search(r"<small[^>]*>(.*?)</small>", cell, re.S)
            small = re.search(r"\(([A-Za-z]{2,4})\s*/", _strip(sm.group(1))) if sm else None
            team = small.group(1).upper() if small else None

        fpid = re.search(r"fp-id-(\d+)", cell)
        recs.append(
            {"player": name, "pos": pos, "team": team,
             "ecr": float(rank), "id": fpid.group(1) if fpid else None,
             "season": year}
        )

    return pl.DataFrame(
        recs,
        schema={"player": pl.Utf8, "pos": pl.Utf8, "team": pl.Utf8,
                "ecr": pl.Float64, "id": pl.Utf8, "season": pl.Int32},
    )


def parse_any(html: str, year: int) -> pl.DataFrame:
    """Dispatch on page family: mpb-player row classes => modern parser."""
    return parse(html, year) if "mpb-player-" in html else parse_early(html, year)


def expected_rows(html: str) -> int:
    """How many rows a correct parse must return -- the scrape regression check.

    Counted PER ROW, not per link or per id: a free agent renders his team as a
    second /nfl/players/ anchor ("FA"), so five 2013 rows hold two links each
    and a raw link count over-counts.

    Team defences are excluded on both sides. They have no player page (early
    era) and no player-label cell (modern era), so neither parser emits them --
    correctly, since bff.ecr keeps only QB/RB/WR/TE. That accounts for the
    entire modern-era gap: 26-32 rows per season, every one a DST, verified
    2026-07-25.
    """
    table = _extract_table(html, "data") or _extract_table(html, "rank-data") or ""
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S)
    if "mpb-player-" in table:
        rows = re.findall(r'<tr class="[^"]*mpb-player-\d+[^"]*"[^>]*>(.*?)</tr>',
                          table, re.S)
    else:
        rows = [r for r in rows if _PLAYER_LINK.search(r)]
    return sum(1 for r in rows if not re.search(r"\bDST\d*\b", r))


def build() -> pl.DataFrame:
    frames = []
    for year in URLS:
        html = fetch(year)
        df = parse_any(html, year)
        # rows parsed must equal player rows present -- see module docstring
        expect = expected_rows(html)
        assert df.height == expect, f"{year}: parsed {df.height} rows, page has {expect}"
        skill = df.filter(
            pl.col("pos").str.replace(r"\d+$", "").is_in(["QB", "RB", "WR", "TE"])
        )
        print(f"{year}: rows={df.height} skill={skill.height} "
              f"top_rank={df['ecr'].min():.0f} "
              f"ids={df['id'].is_not_null().sum()}/{df.height}")
        frames.append(df)

    out = pl.concat(frames)
    out.write_parquet(OUT)
    print(f"wrote {OUT} ({out.height} rows, seasons {out['season'].min()}-{out['season'].max()})")
    return out


if __name__ == "__main__":
    build()
