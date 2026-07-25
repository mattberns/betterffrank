"""Preseason NFL player-level futures props, 2012-2026, via Wayback captures
of sportsoddshistory.com (and its covers.com mirror).

Output: data/raw/props/season_props.parquet
    season (i32), market (str), player (str), odds_american (i32),
    as_of (date), source_url (str)

Source pages: https://www.sportsoddshistory.com/nfl-award/?y=YYYY&sa=nfl&a=CODE&o=r
(mirror: https://www.covers.com/sportsoddshistory/nfl-award/?...). One page per
(season, market); table columns Player / Odds / Result. Fetched through the
Wayback Machine (neither live host is reachable from this environment); HTML is
cached in data/raw/props/nfl_award_YYYY_CODE.html and reruns are offline.

Leakage: the SAME argument as bff/vegas_wayback.py, but enforced HARDER. Every
accepted page carries the header "As of <date> - prior to the start of the
season", so the page permanently records a PRESEASON board and capture timing
cannot leak. That attestation is REQUIRED: `_validate` rejects any capture
without it, and a (season, market) with no attested capture is simply absent
from the parquet rather than filled from an unlabelled board. The `Result`
column is a season OUTCOME and is never parsed (see iter_rows: the first
signed-integer cell in a row is Odds; Result is text).

An OUTCOME-LEAK was found and fixed here on 2026-07-24, and it is the reason
parsing is row-scoped. Eight boards carry a row for a player with `N/A` in the
Odds column and `** WINNER **` in Result: the season's winner, who was NOT on
the preseason board at all. The original document-wide regex matched such a
name, skipped the `N/A`, and took the NEXT row's price -- fabricating quotes
like "Josh Gordon +275" for 2013 receiving yards (the real +275 favourite was
Calvin Johnson). That wrote season-t OUTCOME into a preseason feature, at the
TOP of the board, for 8 player-seasons, 4 of them inside the tune window. Rows
without a priced cell now yield nothing, and `audit_page` asserts that every
player row resolves to either a quote or a recognised unpriced row.

Coverage note: the mirror renders the same table server-side, so ten boards
that had no capture on either host (mvp 2021-2025, rcv 2024, pass/oroy/comeback
2025, rushtd 2024) were recovered by asking Wayback to archive the live mirror
page -- see save_page_now() and the `--save-missing` flag. 2026 is a different
problem and NOT a fetch bug: the mirror's 2026 pages exist but their tables are
empty, because the leader markets are not posted yet.

Markets are the offense-relevant ones only; defensive/coach markets (sack, int,
tack, kick, nfldpoy, droy, nflcoy, fired) are deliberately not fetched.

Build: uv run python -m bff.props_wayback
      uv run python -m bff.props_wayback --save-missing   # see save_page_now()
"""

from __future__ import annotations

import datetime as dt
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROPS = ROOT / "data" / "raw" / "props"
OUT = PROPS / "season_props.parquet"

SEASONS = range(2012, 2027)

# site award code -> our market name. Offense only.
MARKETS = {
    "nflmvp": "mvp",
    "oroy": "oroy",
    "comeback": "comeback",
    "pass": "pass_yds",
    "passtd": "pass_td",
    "rush": "rush_yds",
    "rushtd": "rush_td",
    "rcv": "rec_yds",
    "rcvtd": "rec_td",
}

# One CDX prefix query per host over the whole /nfl-award/ path; per-URL queries
# with an encoded "?y=" miss CDX's canonical urlkey form and return nothing
# (same gotcha as bff/vegas_wayback.py).
CDX_HOSTS = [
    "sportsoddshistory.com%2Fnfl-award%2F",
    "covers.com%2Fsportsoddshistory%2Fnfl-award%2F",
]
CDX = ("https://web.archive.org/cdx/search/cdx?url={}&matchType=prefix"
       "&output=text&fl=original,timestamp,statuscode"
       "&filter=statuscode:200&limit=20000")
UA = {"User-Agent": "betterffrank-research/1.0"}

PRESEASON_RE = re.compile(
    r"As of\s+([A-Z][a-z]+ \d{1,2}, \d{4})\s*[-–]\s*prior to the start of the season",
    re.I,
)

# Parsing is ROW-SCOPED, one <tr> at a time. A single document-wide regex was
# tried first and silently lost 82 rows across 53 files (fixed 2026-07-24):
#   1. the name class excluded "'", so every apostrophe player truncated at the
#      quote ("Le'Veon Bell" -> "Le"), failed the following `&y=` and vanished.
#      ZERO apostrophe names were in the dataset -- no Ja'Marr Chase, no De'Von
#      Achane, no Le'Veon Bell -- i.e. it deleted part of the elite tier.
#   2. a stray nfl-award-player link ABOVE the table let the lazy `.*?` run
#      forward and consume the first data row's odds cell, eating the top line
#      of the board (the winner/favourite, the most informative row there is).
# Scoping to a <tr> makes both impossible: a match cannot span rows, and each
# row is matched independently of what precedes it.
TR_RE = re.compile(r"<tr\b.*?</tr>", re.S | re.I)
# `&(?:amp;)?` is load-bearing: covers.com-mirror pages come back
# Wayback-rewritten with HTML-escaped ampersands.
NAME_RE = re.compile(
    r"nfl-award-player/\?[^\"']*?[?&](?:amp;)?p=([^&]+?)&(?:amp;)?y=\d{4}", re.S)
# first signed-integer cell in the row is Odds; Result ("** WINNER **", "&nbsp;")
# never matches, and being row-scoped it can never be reached anyway.
ODDS_RE = re.compile(r"<td[^>]*>\s*([+-]\d{2,6})\s*</td>", re.S)


def iter_rows(text: str):
    """Yield (raw_name, odds_str) for each real board row."""
    for tr in TR_RE.findall(text):
        nm = NAME_RE.search(tr)
        if not nm:
            continue
        od = ODDS_RE.search(tr)
        if not od:
            continue
        yield nm.group(1), od.group(1)


def audit_page(text: str) -> tuple[int, int, int]:
    """(player rows, parsed quotes, unpriced rows). Invariant: every row that
    contains a player anchor must resolve to exactly one of the two outcomes --
    a parsed quote, or a recognised unpriced (`N/A`) row. Anything else is a
    silently lost row.

    The row census counts the LITERAL anchor string, deliberately NOT `NAME_RE`.
    A first version of this guard counted links with the same regex it was
    validating, so a bug in the name capture hid from its own audit (verified:
    re-introducing the apostrophe bug did not trip it). The check is only worth
    anything if the census is independent of the pattern under test."""
    rows = parsed = unpriced = 0
    for tr in TR_RE.findall(text):
        if "nfl-award-player" not in tr:
            continue
        rows += 1
        has_odds = ODDS_RE.search(tr) is not None
        if NAME_RE.search(tr) and has_odds:
            parsed += 1
        elif not has_odds:
            unpriced += 1
    return rows, parsed, unpriced


def _get(url: str, timeout: int = 90, tries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # Wayback flakes: 429, timeouts, 5xx
            last = e
            time.sleep(5.0 * (attempt + 1))
    raise last  # type: ignore[misc]


def find_captures() -> dict[tuple[int, str], list[tuple[str, str]]]:
    """Candidate captures per (season, award code), newest first."""
    cands: dict[tuple[int, str], list[tuple[str, str]]] = {}
    for host in CDX_HOSTS:
        try:
            text = _get(CDX.format(host)).decode()
        except Exception as e:
            print(f"CDX query failed for {host}: {e}")
            continue
        n = 0
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            url, ts = parts[0], parts[1]
            my = re.search(r"[?&]y=(\d{4})", url)
            ma = re.search(r"[?&]a=([a-z]+)", url)
            if not (my and ma):
                continue
            season, code = int(my.group(1)), ma.group(1)
            if season not in SEASONS or code not in MARKETS:
                continue
            cands.setdefault((season, code), []).append((url, ts))
            n += 1
        print(f"CDX {host.split('%2F')[0]}: {n} candidate captures")
    for key in cands:
        # newest capture first: the site backfills the preseason attestation
        # onto a season's page after the fact, so late captures are likelier
        # to carry it. Dedupe identical (url, ts).
        cands[key] = sorted(set(cands[key]), key=lambda ut: ut[1], reverse=True)
    return cands


def _validate(raw: bytes) -> str | None:
    """Return the as-of date string if this is a real, PRESEASON-attested board
    with enough rows; else None."""
    if len(raw) < 8_000:
        return None
    text = raw.decode("utf-8", errors="replace")
    m = PRESEASON_RE.search(text)
    if not m:
        return None
    if sum(1 for _ in iter_rows(text)) < 8:
        return None
    return m.group(1)


def save_page_now(season: int, code: str) -> tuple[Path, str] | None:
    """Ask Wayback to archive the LIVE mirror page, then validate what comes
    back. This is how the 10 boards with no capture on either host were
    recovered (mvp 2021-2025, rcv 2024, pass/oroy/comeback 2025, rushtd 2024):
    Save Page Now fetches server-side, so it reaches a host this environment
    cannot resolve directly.

    Opt-in only (`--save-missing`). It WRITES to a third-party public archive,
    so a normal run must never trigger it."""
    live = (f"https://www.covers.com/sportsoddshistory/nfl-award/"
            f"?y={season}&sa=nfl&a={code}")
    try:
        raw = _get(f"https://web.archive.org/save/{live}", timeout=180, tries=2)
    except Exception as e:
        print(f"  SPN {season}/{code}: failed ({e})")
        return None
    time.sleep(6.0)  # SPN is rate-limited; do not hammer it
    as_of = _validate(raw)
    if not as_of:
        return None
    cache = PROPS / f"nfl_award_{season}_{code}.html"
    cache.write_bytes(raw)
    return cache, as_of


def fetch_page(season: int, code: str,
               captures: dict[tuple[int, str], list[tuple[str, str]]],
               save_missing: bool = False
               ) -> tuple[Path, str] | None:
    """Fetch (or reuse cached) HTML for one (season, market), trying candidates
    newest-first until one validates. None if none is an attested preseason
    board."""
    cache = PROPS / f"nfl_award_{season}_{code}.html"
    if cache.exists():
        as_of = _validate(cache.read_bytes())
        if as_of:
            return cache, as_of
    for orig, ts in captures.get((season, code), [])[:6]:
        try:
            raw = _get(f"https://web.archive.org/web/{ts}id_/{orig}")
        except Exception:
            continue
        time.sleep(1.0)  # be polite to Wayback
        as_of = _validate(raw)
        if as_of:
            cache.write_bytes(raw)
            return cache, as_of
    if save_missing:
        return save_page_now(season, code)
    return None


def parse_page(path: Path, season: int, code: str, as_of: str) -> pl.DataFrame:
    """Extract (player, American odds) rows. Keeps the first (best) quote per
    player; the board is ordered by the site's `o=r` sort, not by odds."""
    text = path.read_text(encoding="utf-8", errors="replace")
    as_of_date = dt.datetime.strptime(as_of, "%B %d, %Y").date()
    # record the host the cached page actually came from; the two are the same
    # publisher (the old host 301s to the mirror) but provenance should be exact
    if "covers.com/sportsoddshistory" in text:
        src = (f"https://www.covers.com/sportsoddshistory/nfl-award/"
               f"?y={season}&sa=nfl&a={code}")
    else:
        src = (f"https://www.sportsoddshistory.com/nfl-award/"
               f"?y={season}&sa=nfl&a={code}&o=r")
    links, n_rows, unpriced = audit_page(text)
    assert links == n_rows + unpriced, (
        f"{season}/{code}: {links} player rows but {n_rows} parsed + "
        f"{unpriced} unpriced -- {links - n_rows - unpriced} row(s) lost "
        f"silently. Do NOT relax this; investigate the markup."
    )
    rows = []
    for raw_name, odds in iter_rows(text):
        name = urllib.parse.unquote_plus(raw_name).strip()
        name = re.sub(r"\s+", " ", name)
        if not name or len(name) > 40:
            continue
        rows.append({
            "season": season,
            "market": MARKETS[code],
            "player": name,
            "odds_american": int(odds),
            "as_of": as_of_date,
            "source_url": src,
        })
    df = pl.DataFrame(rows)
    if df.height:
        df = df.unique(subset=["season", "market", "player"], keep="first")
    return df


def main() -> None:
    save_missing = "--save-missing" in sys.argv
    PROPS.mkdir(parents=True, exist_ok=True)
    captures = find_captures()
    if save_missing:
        print("--save-missing: uncaptured boards will be archived via Wayback "
              "Save Page Now (writes to archive.org)")
    print()
    frames: list[pl.DataFrame] = []
    missing: list[str] = []
    for season in SEASONS:
        line = [f"{season}:"]
        for code in MARKETS:
            got = fetch_page(season, code, captures, save_missing)
            if got is None:
                missing.append(f"{season}/{MARKETS[code]}")
                line.append(f"{MARKETS[code]}=--")
                continue
            path, as_of = got
            df = parse_page(path, season, code, as_of)
            if df.height < 8:
                missing.append(f"{season}/{MARKETS[code]}")
                line.append(f"{MARKETS[code]}=!{df.height}")
                continue
            frames.append(df)
            line.append(f"{MARKETS[code]}={df.height}")
        print("  ".join(line))

    if not frames:
        raise SystemExit("no attested preseason boards found — nothing written")

    out = (
        pl.concat(frames)
        .with_columns(pl.col("season").cast(pl.Int32))
        .sort("season", "market", "odds_american", "player")
    )
    out.write_parquet(OUT)
    print(f"\nwrote {OUT} ({out.height} rows, "
          f"seasons {out['season'].min()}-{out['season'].max()}, "
          f"{out['market'].n_unique()} markets)")
    if missing:
        print(f"\nno attested preseason capture ({len(missing)}): "
              f"{', '.join(missing)}")

    # sanity: 2013 rushing-yards board had LeSean McCoy (the eventual winner)
    # priced at +2500 as a longshot — the tail pricing this source exists for.
    chk = out.filter(
        (pl.col("season") == 2013) & (pl.col("market") == "rush_yds")
        & (pl.col("player") == "LeSean McCoy")
    )
    if chk.height:
        assert chk["odds_american"][0] == 2500, chk
        print("sanity 2013 rush_yds LeSean McCoy == +2500: OK")
    # every accepted row must predate its season's Week 1
    bad = out.filter(pl.col("as_of").dt.month() > 9)
    assert bad.height == 0, bad
    assert (out["as_of"].dt.year().cast(pl.Int32) == out["season"]).all()
    print("sanity all as_of dates are in-season-year, month <= 9: OK")


if __name__ == "__main__":
    main()
