"""Preseason Vegas season win totals, 2010-2026, via Wayback captures of
sportsoddshistory.com.

Output: data/raw/vegas/win_totals.parquet
    season (i32), team (canonical code), vegas_wins (f64)

Source pages: https://www.sportsoddshistory.com/nfl-win/?y=YYYY&sa=nfl&t=win
(one page per season; table columns Team / Win Total / Over Odds / Under Odds
/ Week bet settled / Actual Wins / Result). Fetched through the Wayback
Machine (the live site is not reachable from this environment); HTML is
cached in data/raw/vegas/nfl_win_YYYY.html and reruns are offline.

Leakage: NONE from capture timing. Unlike an ECR snapshot (whose content
drifts with time), a win-totals HISTORY page captured after the season still
records the PRESEASON over/under line — the line itself is the preseason
artifact. Only the `Win Total` column is exported; `Actual Wins` / `Result`
are season outcomes and are deliberately dropped at parse time.

The season-t win total is a preseason fact for season t (futures lines are
posted in spring). A missing season (e.g. no Wayback capture yet) is simply
absent from the parquet; downstream (bff/situation_features.py) null-fills.

Build: uv run python -m bff.vegas_wayback
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
VEGAS = ROOT / "data" / "raw" / "vegas"

SEASONS = range(2010, 2027)

# Full franchise names as printed by sportsoddshistory -> canonical codes
# (the 32 codes used by data/processed/adp.parquet; see bff/context_data.py).
# Relocations/renames map to the modern franchise code.
TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LAR", "St. Louis Rams": "LAR",
    "St Louis Rams": "LAR",  # site prints it without the period
    "Las Vegas Raiders": "LV", "Oakland Raiders": "LV",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}

# One CDX prefix query over the whole /nfl-win/ path; per-season queries with
# an encoded "?y=" miss CDX's canonical urlkey form and return nothing.
CDX = ("https://web.archive.org/cdx/search/cdx"
       "?url=sportsoddshistory.com%2Fnfl-win%2F&matchType=prefix"
       "&output=text&fl=original,timestamp,statuscode"
       "&filter=statuscode:200&limit=5000")
UA = {"User-Agent": "betterffrank-research/1.0"}


def _get(url: str, timeout: int = 90, tries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # Wayback flakes: timeouts, 5xx
            last = e
            time.sleep(3.0 * (attempt + 1))
    raise last  # type: ignore[misc]


def find_captures() -> dict[int, list[tuple[str, str]]]:
    """Candidate Wayback captures per season, best-first: t=win pages before
    bare y= default views, newer before older. {season: [(url, ts), ...]}."""
    cands: dict[int, list[tuple[str, str]]] = {}
    for line in _get(CDX).decode().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        url, ts = parts[0], parts[1]
        m = re.search(r"[?&]y=(\d{4})", url)
        if not m:
            continue
        y = int(m.group(1))
        if y not in SEASONS:
            continue
        if "t=win" in url or "t=" not in url:
            cands.setdefault(y, []).append((url, ts))
    for y in cands:
        cands[y].sort(key=lambda ut: ("t=win" not in ut[0], ut[1]),
                      reverse=False)
        # sort ascending puts t=win first (False < True) but oldest ts first;
        # want newest within each class:
        cands[y].sort(key=lambda ut: ut[1], reverse=True)
        cands[y].sort(key=lambda ut: "t=win" not in ut[0])
    return cands


def _looks_like_win_page(raw: bytes) -> bool:
    """Real win-totals pages have the table header AND team rows. Empty
    shells (e.g. the sole 2025 capture) have the title but no rows."""
    return (len(raw) > 10_000 and b"Win Total" in raw
            and raw.count(b"Team=") >= 30)


def fetch_season(season: int, captures: dict[int, list[tuple[str, str]]]) -> Path | None:
    """Fetch (or reuse cached) HTML for one season, trying candidates
    best-first until one validates. None if no candidate is a real page."""
    cache = VEGAS / f"nfl_win_{season}.html"
    if cache.exists() and _looks_like_win_page(cache.read_bytes()):
        return cache
    for orig, ts in captures.get(season, [])[:5]:
        try:
            raw = _get(f"https://web.archive.org/web/{ts}id_/{orig}")
        except Exception:
            continue
        time.sleep(1.0)  # be polite to Wayback
        if _looks_like_win_page(raw):
            cache.write_bytes(raw)
            return cache
    return None


ROW_RE = re.compile(
    r"Team=[^\"]*\">\s*([A-Za-z0-9. ]+?)\s*</a>.*?"
    r"<td[^>]*>\s*([0-9]+(?:\.5)?)\s*</td>",
    re.S,
)


def parse_season(path: Path, season: int) -> pl.DataFrame:
    """Extract (team, win total) rows. The first numeric td after each team
    link is the preseason Win Total column; Actual Wins comes columns later
    and is never captured by this regex (single-td lookahead)."""
    html = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for name, total in ROW_RE.findall(html):
        name = re.sub(r"\s+", " ", name).strip()
        if name in TEAM_NAME_MAP:
            rows.append({
                "season": season,
                "team": TEAM_NAME_MAP[name],
                "vegas_wins": float(total),
            })
    df = pl.DataFrame(rows)
    if df.height:
        # a team can appear once per page section; keep first occurrence
        df = df.unique(subset=["season", "team"], keep="first")
    return df


def main() -> None:
    VEGAS.mkdir(parents=True, exist_ok=True)
    captures = find_captures()
    frames, missing = [], []
    for season in SEASONS:
        path = fetch_season(season, captures)
        if path is None:
            missing.append(season)
            print(f"{season}: NO capture — season will be null downstream")
            continue
        df = parse_season(path, season)
        n_expected = 32
        status = "OK" if df.height == n_expected else f"WARN ({df.height}/32 teams)"
        print(f"{season}: {df.height} teams  {status}")
        if df.height >= 28:  # tolerate a straggler row, refuse a broken parse
            frames.append(df)
        else:
            missing.append(season)
    out = (
        pl.concat(frames)
        .with_columns(pl.col("season").cast(pl.Int32))
        .sort("season", "team")
    )
    out.write_parquet(VEGAS / "win_totals.parquet")
    print(f"\nwrote {VEGAS / 'win_totals.parquet'} ({out.height} rows, "
          f"seasons {out['season'].min()}-{out['season'].max()})")
    if missing:
        print(f"missing seasons (null downstream): {missing}")
    # sanity: 2015 CAR preseason line was 8.5
    chk = out.filter((pl.col("season") == 2015) & (pl.col("team") == "CAR"))
    if chk.height:
        assert chk["vegas_wins"][0] == 8.5, chk
        print("sanity 2015 CAR == 8.5 line: OK")


if __name__ == "__main__":
    main()
