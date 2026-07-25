"""Build tidy preseason PPR ADP table from FantasyFootballCalculator raw JSON.

Output: data/processed/adp.parquet with columns
  season, name, norm_name, position, team, adp, adp_rank, times_drafted, gsis_id

Two seasons do not come from the live FFC API, both recovered from Wayback
captures and cached under data/raw/adp/ppr_<season>_wayback.json in the same
shape as the API payloads (the plain file sorts before the _wayback file, so
the recovery always wins):

  2025 -- preseason ADP is simply absent from the API.

  2012 -- the API payload EXISTS but is HALF MISSING, which is worse than
  absent because it looks fine. The API returns 93 players stopping at ADP
  156.9, with ten gaps wider than 4 ADP slots scattered from pick 34 to 157;
  2011 returns 188 (zero such gaps) and 2013 returns 191 (two). The
  contemporaneous 2012-09-02 capture of adp_ppr.php?teams=12 -- three days
  before that season's Week 1, so preseason-clean -- carries 189 players
  running to ADP 172.7, exactly in family with its neighbours. The two also
  disagree on the draft sample underneath: the API reports total_drafts 303
  yet no player on its board was drafted more than 130 times, which cannot be
  right for a board whose top pick goes 1.02; the capture has Arian Foster at
  251. FFC's stored 2012 PPR data was pruned at some point after 2012. Do NOT
  "simplify" this back to the API -- 2012 is a scored tune fold and the API
  version grades it on 91 players against ~150 everywhere else. Found and
  fixed 2026-07-25.

The 2012-era page is a different layout from the modern one, so there are two
row parsers; `parse_legacy_html` handles the 2012 table (<tr class="RB"> with
a bye column and a graph checkbox carrying the player id).
"""

from __future__ import annotations

import html as htmllib
import json
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_ADP = ROOT / "data" / "raw" / "adp"
CROSSWALK = ROOT / "data" / "raw" / "db_playerids.csv"
OUT = ROOT / "data" / "processed" / "adp.parquet"

WAYBACK_TS = "20250808113300"
WAYBACK_URL = (
    f"https://web.archive.org/web/{WAYBACK_TS}/"
    "https://fantasyfootballcalculator.com/adp/ppr"
)
WAYBACK_JSON = RAW_ADP / "ppr_2025_wayback.json"
WAYBACK_HTML = RAW_ADP / "ppr_2025_wayback.html"

# Legacy-layout recoveries: season -> (wayback timestamp, original url).
# See the module docstring for why 2012 cannot use the API payload.
LEGACY_RECOVERY = {
    2012: (
        "20120902090435",
        "http://fantasyfootballcalculator.com/adp_ppr.php?teams=12",
    ),
}
# A recovered board must be at least this deep, else the capture is a dud and
# we would silently re-introduce the truncation we are fixing.
MIN_LEGACY_PLAYERS = 150

KEEP_POS = ("QB", "RB", "WR", "TE")

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    """lowercase; strip periods/apostrophes/hyphens; strip jr/sr/ii/iii/iv/v
    suffixes; collapse whitespace. Must match the definition used elsewhere."""
    s = name.lower()
    s = re.sub(r"[.'’-]", "", s)
    parts = [p for p in s.split() if p]
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


NORM_NAME_EXPR = (
    pl.col("name")
    .str.to_lowercase()
    .str.replace_all(r"[.'’-]", "")
    .str.replace_all(r"\s+(jr|sr|ii|iii|iv|v)(\s+(jr|sr|ii|iii|iv|v))*\s*$", "")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
    .alias("norm_name")
)

_ROW_RE = re.compile(
    r'<tr class="(?P<pos>QB|RB|WR|TE|DEF|PK)">\s*'
    r'<td[^>]*>(?P<rank>\d+)</td>\s*'
    r'<td[^>]*>(?P<pick>[\d.]+)</td>\s*'
    r'<td class="adp-player-name"><a[^>]*>(?P<name>[^<]+)</a></td>\s*'
    r'<td>(?P<pos2>[A-Z]+)</td>\s*'
    r'<td>(?P<team>[A-Z]*)</td>.*?'
    r'<td class="d-none d-sm-table-cell"[^>]*>(?P<adp>[\d.]+)</td>\s*'
    r'<td class="d-none d-sm-table-cell"[^>]*>(?P<stdev>[\d.]+)</td>\s*'
    r'<td class="d-none d-sm-table-cell"[^>]*>(?P<high>[\d.]+)</td>\s*'
    r'<td class="d-none d-sm-table-cell"[^>]*>(?P<low>[\d.]+)</td>\s*'
    r'<td class="d-none d-sm-table-cell"[^>]*>(?P<td_>\d+)</td>.*?'
    r'value="(?P<pid>\d+)"',
    re.S,
)


def parse_wayback_html(html: str) -> dict:
    """Parse the archived /adp/ppr page into an API-shaped payload."""
    players = []
    for m in _ROW_RE.finditer(html):
        players.append(
            {
                "player_id": int(m["pid"]),
                "name": htmllib.unescape(m["name"]).strip(),
                "position": m["pos"],
                "team": m["team"],
                "adp": float(m["adp"]),
                "adp_formatted": m["pick"],
                "times_drafted": int(m["td_"]),
                "stdev": float(m["stdev"]),
            }
        )
    return {
        "status": "Success",
        "meta": {
            "type": "PPR",
            "teams": 12,
            "source": "wayback",
            "snapshot": WAYBACK_TS,
            "url": WAYBACK_URL,
        },
        "players": players,
    }


# 2012-era row: rank | pick | name | pos | team | bye | adp | stdev | high |
# low | times_drafted | graph checkbox (value=<ffc player id>)
_LEGACY_ROW_RE = re.compile(
    r'<tr class="(?P<pos>QB|RB|WR|TE|DEF|PK)">\s*'
    r'<td[^>]*>(?P<rank>\d+)</td>\s*'
    r'<td[^>]*>(?P<pick>[\d.]+)</td>\s*'
    r'<td[^>]*>(?P<name>[^<]+)</td>\s*'
    r'<td[^>]*>(?P<pos2>[A-Z]+)</td>\s*'
    r'<td[^>]*>(?P<team>[A-Z]*)</td>\s*'
    r'<td[^>]*>(?P<bye>\d*)</td>\s*'
    r'<td[^>]*>(?P<adp>[\d.]+)</td>\s*'
    r'<td[^>]*>(?P<stdev>[\d.]+)</td>\s*'
    r'<td[^>]*>(?P<high>[\d.]+)</td>\s*'
    r'<td[^>]*>(?P<low>[\d.]+)</td>\s*'
    r'<td[^>]*>(?P<td_>\d+)</td>.*?'
    r'value="(?P<pid>\d+)"',
    re.S,
)


def parse_legacy_html(html: str, season: int, ts: str, url: str) -> dict:
    """Parse a 2012-era adp_ppr.php page into an API-shaped payload."""
    players = [
        {
            "player_id": int(m["pid"]),
            "name": htmllib.unescape(m["name"]).strip(),
            "position": m["pos"],
            "team": m["team"],
            "adp": float(m["adp"]),
            "adp_formatted": m["pick"],
            "times_drafted": int(m["td_"]),
            "stdev": float(m["stdev"]),
        }
        for m in _LEGACY_ROW_RE.finditer(html)
    ]
    # scrape regression check, same rule as bff.ecr_wayback: a row-scoped
    # parser must not silently lose rows to a layout it did not expect
    present = len(re.findall(r'<tr class="(?:QB|RB|WR|TE|DEF|PK)">', html))
    assert len(players) == present, (
        f"{season}: parsed {len(players)} rows, page has {present}"
    )
    return {
        "status": "Success",
        "meta": {"type": "PPR", "teams": 12, "source": "wayback",
                 "snapshot": ts, "url": url},
        "players": players,
    }


def recover_legacy(season: int) -> dict | None:
    """Return a legacy-layout season payload, fetching+caching on first run."""
    ts, url = LEGACY_RECOVERY[season]
    out_json = RAW_ADP / f"ppr_{season}_wayback.json"
    if out_json.exists():
        d = json.loads(out_json.read_text())
        if d.get("players"):
            return d
    import requests

    out_html = RAW_ADP / f"ppr_{season}_wayback.html"
    if out_html.exists():
        text = out_html.read_text(encoding="utf-8", errors="replace")
    else:
        r = requests.get(f"https://web.archive.org/web/{ts}/{url}", timeout=120)
        r.raise_for_status()
        text = r.text
        out_html.write_text(text, encoding="utf-8")
    d = parse_legacy_html(text, season, ts, url)
    assert len(d["players"]) >= MIN_LEGACY_PLAYERS, (
        f"{season}: recovered only {len(d['players'])} players "
        f"(< {MIN_LEGACY_PLAYERS}); capture is unusable, do not cache it"
    )
    out_json.write_text(json.dumps(d))
    return d


def recover_2025() -> dict | None:
    """Return the 2025 payload, fetching from the Wayback Machine if needed."""
    if WAYBACK_JSON.exists():
        d = json.loads(WAYBACK_JSON.read_text())
        if d.get("players"):
            return d
    import requests

    r = requests.get(WAYBACK_URL, timeout=120)
    r.raise_for_status()
    WAYBACK_HTML.write_text(r.text)
    d = parse_wayback_html(r.text)
    if not d["players"]:
        return None
    WAYBACK_JSON.write_text(json.dumps(d))
    return d


def load_season_payloads() -> dict[int, dict]:
    """season -> API-shaped payload, skipping error stubs and future seasons."""
    payloads: dict[int, dict] = {}
    for p in sorted(RAW_ADP.glob("ppr_*.json")):
        m = re.match(r"ppr_(\d{4})(_wayback)?\.json$", p.name)
        if not m:
            continue
        season = int(m.group(1))
        d = json.loads(p.read_text())
        if d.get("status") != "Success" or not d.get("players"):
            continue
        payloads[season] = d  # wayback file sorts after plain and overrides stub
    if 2025 not in payloads:
        d = recover_2025()
        if d:
            payloads[2025] = d
    for season in LEGACY_RECOVERY:
        # the API payload for these seasons is present but truncated, so the
        # recovery is unconditional -- not a fallback for an absent season
        if payloads.get(season, {}).get("meta", {}).get("source") == "wayback":
            continue
        d = recover_legacy(season)
        if d:
            payloads[season] = d
    return payloads


def build_adp_table() -> pl.DataFrame:
    payloads = load_season_payloads()
    frames = []
    for season, d in sorted(payloads.items()):
        df = pl.DataFrame(
            [
                {
                    "season": season,
                    "name": pp["name"],
                    "position": pp["position"],
                    "team": pp.get("team"),
                    "adp": float(pp["adp"]),
                    "times_drafted": int(pp.get("times_drafted") or 0),
                }
                for pp in d["players"]
            ]
        )
        frames.append(df)
    adp = pl.concat(frames)
    adp = adp.filter(pl.col("position").is_in(KEEP_POS))
    adp = adp.with_columns(NORM_NAME_EXPR)
    adp = adp.sort(["season", "adp", "times_drafted"], descending=[False, False, True])
    adp = adp.with_columns(
        pl.int_range(1, pl.len() + 1).over("season").alias("adp_rank")
    )
    return adp.select(
        "season", "name", "norm_name", "position", "team",
        "adp", "adp_rank", "times_drafted",
    )


# FFC nickname -> crosswalk name (normalized), verified single match in crosswalk
ALIASES = {
    "hollywood brown": "marquise brown",
    "gabe davis": "gabriel davis",
    "kenny gainwell": "kenneth gainwell",
    "chig okonkwo": "chigoziem okonkwo",
    "steve johnson": "stevie johnson",
    "joshua cribbs": "josh cribbs",
    "jeff wilson": "jeffery wilson",
}


def join_gsis(adp: pl.DataFrame) -> pl.DataFrame:
    ids = pl.read_csv(CROSSWALK, null_values=["NA"])
    ids = ids.filter(pl.col("gsis_id").is_not_null()).select(
        "name", "position", "gsis_id", "draft_year"
    )
    ids = ids.with_columns(NORM_NAME_EXPR).select(
        pl.col("norm_name").alias("match_name"), "position", "gsis_id", "draft_year"
    )

    adp = adp.with_row_index("_i").with_columns(
        pl.col("norm_name").replace(ALIASES).alias("match_name")
    )

    # (match_name, position): keep candidates already drafted by that season
    # (draft_year null/0 = unknown, always kept); match if one gsis remains.
    cand = (
        adp.join(ids, on=["match_name", "position"], how="inner")
        .filter(
            pl.col("draft_year").is_null()
            | (pl.col("draft_year") <= pl.col("season"))
        )
        .group_by("_i")
        .agg(pl.col("gsis_id").n_unique().alias("n"), pl.col("gsis_id").first())
        .filter(pl.col("n") == 1)
        .select("_i", "gsis_id")
    )
    # fallback: match_name alone, unambiguous across the whole crosswalk
    by_name = (
        ids.group_by("match_name")
        .agg(pl.col("gsis_id").n_unique().alias("n"), pl.col("gsis_id").first())
        .filter(pl.col("n") == 1)
        .select("match_name", pl.col("gsis_id").alias("gsis_id_fb"))
    )

    out = adp.join(cand, on="_i", how="left")
    out = out.join(by_name, on="match_name", how="left")
    return (
        out.with_columns(pl.coalesce("gsis_id", "gsis_id_fb").alias("gsis_id"))
        .drop("gsis_id_fb", "match_name", "_i")
        .sort(["season", "adp_rank"])
    )


def main() -> None:
    adp = join_gsis(build_adp_table())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    adp.write_parquet(OUT)

    counts = adp.group_by("season").len().sort("season")
    print("rows per season:")
    for s, n in counts.iter_rows():
        print(f"  {s}: {n}")

    top = adp.filter(pl.col("adp_rank") <= 150)
    match = (
        top.group_by("season")
        .agg(
            pl.len().alias("n"),
            pl.col("gsis_id").is_not_null().sum().alias("matched"),
        )
        .sort("season")
        .with_columns((pl.col("matched") / pl.col("n")).round(4).alias("rate"))
    )
    print("gsis match rate (adp_rank<=150):")
    for s, n, m, r in match.iter_rows():
        print(f"  {s}: {m}/{n} = {r:.1%}")
    tot = top.select(
        pl.col("gsis_id").is_not_null().sum().alias("m"), pl.len().alias("n")
    ).row(0)
    print(f"  overall: {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.1%}")


if __name__ == "__main__":
    main()
