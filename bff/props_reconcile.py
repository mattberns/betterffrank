"""Cross-capture reconciliation of the props corpus (READ-ONLY: Wayback replay
only, no Save Page Now, no writes to archive.org).

For every (season, market) board in data/processed/season_props.parquet, fetch
up to N INDEPENDENT Wayback captures (timestamp-diverse, both hosts), parse each
with the shipped row-scoped parser, and require the parsed quote set to be
identical. A preseason board is supposed to be frozen, so any odds or row-count
disagreement between captures means at least one capture is untrustworthy.

Legit reason two captures CAN differ: the site adds the `** WINNER **` / `N/A`
row after the season ends. That row is unpriced and excluded by the parser, so
it must not change the parsed set -- if it does, that is itself a finding.

Writes reconcile_report.json + a printed summary. Resumable: fetched HTML is
cached under scratchpad/recon_cache/.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import polars as pl

import sys
from bff.props_wayback import (MARKETS, PRESEASON_RE, find_captures, iter_rows,
                               audit_page)

ROOT = Path("/home/bernstml19/source/betterffrank")
SCRATCH = ROOT / "reports"
CACHE = ROOT / "data" / "raw" / "props" / "recon_cache"
UA = {"User-Agent": "betterffrank-research/1.0"}
MAX_PER_KEY = 4          # extra captures compared per board
INV = {v: k for k, v in MARKETS.items()}


def get(url, timeout=90, tries=4):
    last = None
    for a in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if "429" in str(e):
                time.sleep(20 * (a + 1))
            else:
                time.sleep(4 * (a + 1))
    raise last


def fetch_capture(orig, ts):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"\W+", "_", f"{ts}_{orig}")[-120:]
    f = CACHE / f"{key}.html"
    if f.exists():
        return f.read_text("utf-8", "replace")
    raw = get(f"https://web.archive.org/web/{ts}id_/{orig}")
    time.sleep(1.0)
    f.write_bytes(raw)
    return raw.decode("utf-8", "replace")


def quote_set(text):
    """Canonical parsed content: frozenset of (name, odds) + attestation."""
    qs = frozenset((urllib.parse.unquote_plus(n).strip(), int(o))
                   for n, o in iter_rows(text))
    m = PRESEASON_RE.search(text)
    return qs, (m.group(1) if m else None)


def pick(caps):
    """Timestamp-diverse subset: earliest, latest, evenly spaced between."""
    caps = sorted(set(caps), key=lambda ut: ut[1])
    if len(caps) <= MAX_PER_KEY:
        return caps
    idx = [round(i * (len(caps) - 1) / (MAX_PER_KEY - 1)) for i in range(MAX_PER_KEY)]
    return [caps[i] for i in sorted(set(idx))]


def main():
    built = pl.read_parquet(ROOT / "data/processed/season_props.parquet")
    keys = sorted(set(built.select("season", "market").unique().rows()))
    # the board as SHIPPED (from the cached page the pipeline actually used)
    shipped = defaultdict(dict)
    for r in built.iter_rows(named=True):
        shipped[(r["season"], r["market"])][r["player"]] = r["odds_american"]

    caps = find_captures()
    report, counts = [], defaultdict(int)
    for season, market in keys:
        code = INV[market]
        cand = pick(caps.get((season, code), []))
        ship = frozenset(shipped[(season, market)].items())
        variants, errors = {}, []
        for orig, ts in cand:
            try:
                text = fetch_capture(orig, ts)
            except Exception as e:
                errors.append(f"{ts}: {e}")
                continue
            rows, parsed, unpriced = audit_page(text)
            if rows != parsed + unpriced:
                errors.append(f"{ts}: AUDIT FAIL rows={rows} parsed={parsed} unpriced={unpriced}")
                continue
            qs, asof = quote_set(text)
            if not asof or len(qs) < 8:
                continue          # shell / unattested capture: not a comparator
            variants[ts] = (qs, asof, unpriced)

        comparators = len(variants)
        if comparators == 0:
            status = "SINGLE_SOURCED"
        else:
            diffs = []
            for ts, (qs, asof, unp) in sorted(variants.items()):
                if qs != ship:
                    only_ship = sorted(ship - qs)[:4]
                    only_cap = sorted(qs - ship)[:4]
                    diffs.append({"ts": ts, "n_capture": len(qs), "n_shipped": len(ship),
                                  "only_in_shipped": only_ship, "only_in_capture": only_cap,
                                  "asof": asof})
            status = "AGREE" if not diffs else "DISAGREE"
        counts[status] += 1
        entry = {"season": season, "market": market, "status": status,
                 "n_shipped": len(ship), "comparators": comparators,
                 "capture_ts": sorted(variants), "errors": errors}
        if status == "DISAGREE":
            entry["diffs"] = diffs
        report.append(entry)
        print(f"{season} {market:<9} {status:<15} shipped={len(ship):>3} "
              f"comparators={comparators}" + (f"  ERR={len(errors)}" if errors else ""), flush=True)

    (ROOT / "reports" / "props_reconcile.json").write_text(json.dumps(report, indent=1))
    print("\n=== SUMMARY ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:<16} {v}")
    dis = [r for r in report if r["status"] == "DISAGREE"]
    if dis:
        print(f"\n=== {len(dis)} DISAGREEING BOARDS ===")
        for r in dis:
            print(f"\n  {r['season']} {r['market']} (shipped {r['n_shipped']})")
            for d in r["diffs"]:
                print(f"    capture {d['ts']}: n={d['n_capture']} vs shipped {d['n_shipped']}"
                      f"  asof={d['asof']}")
                if d["only_in_shipped"]:
                    print(f"      only shipped: {d['only_in_shipped']}")
                if d["only_in_capture"]:
                    print(f"      only capture: {d['only_in_capture']}")
    single = [f"{r['season']}/{r['market']}" for r in report if r["status"] == "SINGLE_SOURCED"]
    if single:
        print(f"\n=== {len(single)} SINGLE-SOURCED (no independent comparator) ===")
        print("  " + ", ".join(single))


if __name__ == "__main__":
    main()
