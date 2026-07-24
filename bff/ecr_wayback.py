"""One-time backfill: extract preseason FantasyPros PPR ECR for 2015-2020 from
Wayback Machine snapshots of the overall draft cheatsheet.

Not part of the routine build. Run once:

    uv run python -m bff.ecr_wayback

It fetches each archived page to data/raw/wayback_ppr/ppr_<year>.html (cached;
re-runs parse the saved file offline), parses the ranking table, and writes
data/raw/db_fpecr_wayback.parquet in the same schema the archive branch of
bff.ecr emits (player, pos, team, ecr, id, season). bff.ecr unions that parquet.

Two page eras, both handled by one position-relative parser:
  - 2015-2017: <table id="data">
  - 2018-2020: <table id="rank-data"> (adds a WSID column)
Every player row is <tr class="...mpb-player-<fpid>..."> in both eras. The FP
player id shares db_playerids.csv's fantasypros_id space, so these rows resolve
gsis via bff.ecr's by-fantasypros_id join. `ecr` is the displayed overall rank
(td[0]); only the downstream ordinal re-rank matters and overall rank is tie-free,
so ecr_rank is fully determined and matches FantasyPros' own ordering.
"""
import re
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT / "data" / "raw" / "wayback_ppr"
OUT = ROOT / "data" / "raw" / "db_fpecr_wayback.parquet"

# Wayback snapshots (preseason PPR overall cheatsheet), all late-Aug/early-Sep
# for cross-season timing consistency.
URLS = {
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


def build() -> pl.DataFrame:
    frames = []
    for year in URLS:
        df = parse(fetch(year), year)
        skill = df.filter(
            pl.col("pos").str.replace(r"\d+$", "").is_in(["QB", "RB", "WR", "TE"])
        )
        print(f"{year}: rows={df.height} skill={skill.height} "
              f"top_rank={df['ecr'].min():.0f} distinct_ids={df['id'].n_unique()}")
        frames.append(df)

    out = pl.concat(frames)
    out.write_parquet(OUT)
    print(f"wrote {OUT} ({out.height} rows, seasons {out['season'].min()}-{out['season'].max()})")
    return out


if __name__ == "__main__":
    build()
