"""FantasyPros ADP boards, signed-in, via Playwright.

Replaces FantasyFootballCalculator as the ADP source. FFC's boards are pruned
at the publisher in ways that cannot be repaired from any FFC endpoint: the
2022 half-PPR board is missing Austin Ekeler outright (PPR ADP rank 3), the
2025 board is missing five top-120 players, and half-PPR does not exist at all
before 2018. See the "Half-PPR ADP" and "Three input defects" sections in
CLAUDE.md for the full history.

    https://www.fantasypros.com/nfl/adp/<slug>.php?year=<season>

**The page is JS-rendered AND behind a registration fence** — anonymous
requests get a 5-row teaser ("Create a free account to unlock"), which is why
this is a Playwright module and not a `requests` scrape.

**The sign-in page is also behind a Cloudflare Turnstile challenge that
Playwright's bundled Chromium CANNOT pass** — it renders "Verification failed"
and leaves the Sign In button disabled, so no session is ever created. Three
auth routes, in order of reliability:

  1. **Sign in with plain Chrome, pointed at this module's profile.** No
     automation touches the login, so Turnstile sees an ordinary browser:

         google-chrome --user-data-dir=<repo>/data/raw/fp_adp/chrome-profile

     Sign in, quit Chrome (release the profile lock), then `--check`. Every
     later run reuses that profile. This is the recommended route.
  2. `--login` does the same thing but drives the window through Playwright
     (real Chrome, `channel="chrome"`, persistent profile, headed). More
     convenient, but CDP-driven Chrome can still trip Turnstile.
  3. `--import-cookies FILE` — sign in on any browser, export the
     fantasypros.com cookies (a cookies.txt extension, or a JSON export) and
     import them here. Works when the profile routes do not; note
     `cf_clearance` is IP- and User-Agent-bound and may not transfer.

The profile and the cookie file both live under `data/raw/fp_adp/` and are
git-ignored as credentials. Fetching is headed by default for the same
anti-automation reason; `--headless` exists but expect challenges.

    uv run python -m bff.fp_adp --login                    # once, headed real Chrome
    uv run python -m bff.fp_adp --check                     # is the session live?
    uv run python -m bff.fp_adp --probe --format half       # which years exist
    uv run python -m bff.fp_adp --fetch --format all        # cache raw boards
    uv run python -m bff.fp_adp --build --format all        # -> parquet
    uv run python -m bff.fp_adp --build --format half --realtime   # live season
                                                           # off REAL-TIME ADP

Extensibility, since this will be re-run every preseason:
  - `FORMATS` maps a short key to a page slug. A new format (2QB, dynasty,
    superflex) is one line here and needs no other change.
  - Seasons are a CLI range, and `--fetch` is CACHE-FIRST: a season already in
    `data/raw/fp_adp/<format>_<season>.json` is never re-fetched without
    `--refresh`. So next August is `--fetch --seasons 2027` and nothing else
    moves. Past seasons are frozen artifacts; only the live season should ever
    be refreshed.
  - The per-site columns are discovered from the DOM, not hardcoded. The
    contributing sites CHANGE BY YEAR (2022 renders YAHOO/SLEEPER/RTSPORTS),
    and each cached board records its own `sites` map, so a cross-season
    comparison can check whether it is comparing like with like. This is the
    provenance FFC never exposed.

Identity: rows carry a real FantasyPros player id in the `fp-id-<n>` class, so
the gsis join is id-first (crosswalk `fantasypros_id`) with the same name
ladder `bff.ecr` uses as fallback, and `norm_name` is imported from `bff.ecr`
rather than defined again — this is the same publisher as ECR, so it must
normalize identically. Do not add a fifth `norm_name` variant.

CAUTION, read before wiring this into the model: FantasyPros ADP is a
CONSENSUS ACROSS SITES (Yahoo, Sleeper, RTSports, ...), not a single site's
mock-draft sample the way FFC's board is. It is a different measurement of
"the market". Swapping it in changes the ADP baseline every reported number is
measured against, so it is a results-affecting change requiring the full doc
sync in CLAUDE.md, and the model/ADP deltas and p-values must be re-derived,
not carried over.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from bff.ecr import norm_name

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fp_adp"
PROFILE = RAW / "chrome-profile"          # persistent signed-in Chrome profile
STATE = RAW / "auth_state.json"           # optional cookie-import fallback
CHROME_CHANNEL = "chrome"                 # real Chrome, NOT bundled Chromium
CROSSWALK = ROOT / "data" / "raw" / "db_playerids.csv"
PROC = ROOT / "data" / "processed"

# short key -> page slug. Add a format here and nothing else changes.
FORMATS = {
    "half": "half-point-ppr-overall",
    "ppr": "ppr-overall",
    "std": "overall",
}
URL = "https://www.fantasypros.com/nfl/adp/{slug}.php?year={season}"
SIGNIN = "https://www.fantasypros.com/accounts/signin/"

DEFAULT_SEASONS = tuple(range(2012, 2027))
KEEP_POS = ("QB", "RB", "WR", "TE")

# --- REAL-TIME ADP -----------------------------------------------------------
# The board carries a trailing "REAL-TIME ADP TAKEN FROM REAL RECENT DRAFTS
# ACROSS PLATFORMS" column (cell class `realtime`) alongside AVG. AVG is a mean
# pick number over the whole sampling window; real-time is a dense RANK over
# only the recent drafts, so it reprices news the season-long average is still
# averaging away. 2026 example, measured 2026-09-02: Josh Jacobs sits at AVG
# 48.0 (rank 49) and real-time 96 -- and the half-PPR ECR has him 152, so the
# experts and the recent market agree with each other and disagree with AVG.
#
# `--realtime` is OPT-IN and applies ONLY to the seasons listed here, which is
# the live season and nothing else. Two reasons it must stay pinned:
#
#   1. Only the 2025 and 2026 captures have the column at all (it post-dates
#      the earlier pages), so a blanket flag would silently reprice one or two
#      seasons of a fourteen-season series and leave the rest on AVG.
#   2. 2025 is a COMPLETED season, so its page's "recent drafts" are drafts
#      happening NOW, not in its preseason -- an in-season number written into
#      a preseason board. That is the `bff.fp_ecr.check_preseason` failure mode
#      in a different column, so 2025 stays on AVG permanently.
#
# The value is live: it moves hour to hour, so the cached capture is the pinned
# artifact and `--build` prints each payload's fetch time.
REALTIME_SEASONS = frozenset({2026})

# "96 -47" (rank then move vs AVG), "1" (unchanged), or an em dash when the
# recent-draft sample has never taken the player. Only the leading rank is used.
_REALTIME_RE = re.compile(r"^(\d+)")

# A signed-in board renders the full field. Anything at or below the fence's
# teaser depth means the session is dead or the page changed -- refuse to cache
# it, because a silently truncated board is exactly the FFC failure mode this
# module exists to escape.
FENCE_ROWS = 20
MIN_ROWS = 100

# One evaluate() call: find the ADP table by its headers (the table has no
# stable id), then return headers, the first row's cell classes (which carry
# the site ids), and every row. Kept as one round-trip so a lazy-loading page
# cannot interleave a re-render between reads.
EXTRACT_JS = r"""() => {
  const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
  const tables = [...document.querySelectorAll("table")];
  const t = tables.find((t) => {
    const h = [...t.querySelectorAll("thead th")].map((x) =>
      norm(x.innerText).toUpperCase());
    return h.includes("AVG") && h.some((x) => x.startsWith("RANK"));
  });
  if (!t) return null;
  const headers = [...t.querySelectorAll("thead th")].map((x) => norm(x.innerText));
  const cellClass = (td) => {
    const c = [...td.classList].find((x) => x.startsWith("mcu-table__cell--"));
    return c ? c.replace("mcu-table__cell--", "") : "";
  };
  const trs = [...t.querySelectorAll("tbody tr")];
  const classes = trs.length
    ? [...trs[0].querySelectorAll("td")].map(cellClass) : [];
  const rows = trs.map((tr) => {
    const a = tr.querySelector("a.fp-player-link");
    const m = a ? (a.className || "").match(/fp-id-(\d+)/) : null;
    const team = tr.querySelector(".reports__player-team");
    return {
      fp_id: m ? m[1] : null,
      name: a ? norm(a.getAttribute("fp-player-name") || a.innerText) : null,
      team_bye: team ? norm(team.innerText) : null,
      cells: [...tr.querySelectorAll("td")].map((td) => norm(td.innerText)),
    };
  });
  return {headers, classes, rows};
}"""


def _payload_path(fmt: str, season: int) -> Path:
    return RAW / f"{fmt}_{season}.json"


def _require_state() -> None:
    """Fail BEFORE launching Playwright — exiting inside the sync_playwright
    context manager buries the message under driver-teardown tracebacks."""
    if not (PROFILE / "Default").exists() and not STATE.exists():
        sys.exit(
            f"no signed-in browser profile at {PROFILE.relative_to(ROOT)}.\n"
            f"Recommended (no automation at sign-in, so Cloudflare behaves):\n"
            f"    google-chrome --user-data-dir={PROFILE}\n"
            f"  sign in, quit Chrome, then:\n"
            f"    uv run python -m bff.fp_adp --check\n"
            f"Alternatives: --login (driven Chrome) or --import-cookies FILE"
        )


def _new_context(pw, headless: bool = False, require_state: bool = True):
    """A PERSISTENT real-Chrome context, returned as (context, close_fn).

    Why not `chromium.launch()` + storage_state: the sign-in page is behind a
    **Cloudflare Turnstile** challenge, and Playwright's bundled Chromium fails
    it outright ("Verification failed", Sign In stays disabled), so the session
    can never be established in the first place. Three things fix that, and all
    three are load-bearing:

      1. `channel="chrome"` — the real Google Chrome build (present at
         /usr/bin/google-chrome), not the Playwright Chromium/headless-shell.
      2. A PERSISTENT `user_data_dir`, so Turnstile sees a browser with history
         and a stable fingerprint rather than a fresh throwaway profile, and so
         the clearance cookie it issues survives to later runs.
      3. HEADED by default. Headless is the single strongest automation signal;
         `--headless` exists but expect challenges. Fetching ~15 boards headed
         costs nothing, so leave it headed.

    `--disable-blink-features=AutomationControlled` drops the `navigator.
    webdriver` flag. This is ordinary first-party access to a site the user has
    an account on, at human pace; do not bolt on evasion beyond this. If
    Turnstile still blocks, use the `--import-cookies` route, which never
    automates the login at all.
    """
    if require_state:
        _require_state()
    PROFILE.mkdir(parents=True, exist_ok=True)
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE),
        channel=CHROME_CHANNEL,
        headless=headless,
        viewport={"width": 1600, "height": 1200},
        args=["--disable-blink-features=AutomationControlled"],
    )
    if STATE.exists():
        # cookie-import route: fold the exported cookies into this profile
        st = json.loads(STATE.read_text())
        if st.get("cookies"):
            ctx.add_cookies(st["cookies"])
    return ctx, ctx.close


def _scrape(page, fmt: str, season: int) -> dict:
    """Render one board and return the extracted payload (no caching here)."""
    url = URL.format(slug=FORMATS[fmt], season=season)
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    try:
        page.wait_for_selector("table tbody tr", timeout=45_000)
    except Exception:
        pass
    # The table lazy-renders on scroll; scroll until the row count stops moving.
    prev, stable = -1, 0
    for _ in range(40):
        d = page.evaluate(EXTRACT_JS)
        n = len(d["rows"]) if d else 0
        stable = stable + 1 if n == prev else 0
        if stable >= 2:
            break
        prev = n
        page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        page.wait_for_timeout(600)
    d = page.evaluate(EXTRACT_JS)
    if not d:
        raise RuntimeError(f"{fmt} {season}: no ADP table on {url}")

    headers, classes, rows = d["headers"], d["classes"], d["rows"]
    sites = {
        c: h
        for c, h in zip(classes, headers)
        if c.startswith("src_")
    }
    fenced = bool(
        page.locator("#registration-fence, .registration-fence").count()
    ) and len(rows) <= FENCE_ROWS
    return {
        "source": "fantasypros",
        "format": fmt,
        "season": season,
        "url": url,
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signed_in": not fenced,
        "headers": headers,
        "classes": classes,
        "sites": sites,
        "n_rows": len(rows),
        "players": rows,
    }


def cmd_login() -> None:
    """One-time interactive sign-in in a persistent REAL-Chrome profile.

    The profile itself is the stored session; nothing is exported. Sign in
    inside this window and the cookies persist at PROFILE for every later run.
    """
    from playwright.sync_api import sync_playwright

    PROFILE.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        ctx, close = _new_context(pw, headless=False, require_state=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(SIGNIN, wait_until="domcontentloaded", timeout=120_000)
        print(
            "\nReal Chrome is open on the FantasyPros sign-in page, using the\n"
            f"persistent profile at {PROFILE.relative_to(ROOT)}.\n\n"
            "  1. Sign in. Let the Cloudflare check finish before clicking\n"
            "     Sign In — the button stays greyed out until it passes.\n"
            "  2. Dismiss any cookie/consent banner.\n"
            "  3. Come back here and press Enter.\n\n"
            "If Cloudflare still refuses to verify, quit and use the no-\n"
            "automation route instead:\n"
            "     uv run python -m bff.fp_adp --import-cookies <file>\n"
        )
        input("press Enter once you are signed in... ")
        close()
    print("profile saved; verifying...")
    cmd_check()


def cmd_import_cookies(path: str) -> None:
    """Fallback for when Turnstile refuses an automated browser entirely: sign
    in with your OWN browser (no automation, so the challenge passes normally),
    export the fantasypros.com cookies, and import them here.

    Accepts either a Netscape `cookies.txt` or a JSON array from a cookie-export
    extension. Note `cf_clearance` is bound to the issuing browser's IP AND
    User-Agent, so it only transfers if the export came from this machine; the
    plain session cookies are what actually matter for the fence.
    """
    src = Path(path)
    if not src.exists():
        sys.exit(f"no such file: {path}")
    text = src.read_text()
    cookies: list[dict] = []
    if text.lstrip().startswith(("[", "{")):
        raw = json.loads(text)
        raw = raw if isinstance(raw, list) else raw.get("cookies", [])
        for c in raw:
            cookies.append(
                {
                    "name": c["name"],
                    "value": c["value"],
                    "domain": c.get("domain") or ".fantasypros.com",
                    "path": c.get("path", "/"),
                    "expires": float(c.get("expirationDate", c.get("expires", -1))),
                    "httpOnly": bool(c.get("httpOnly", False)),
                    "secure": bool(c.get("secure", True)),
                    "sameSite": "Lax",
                }
            )
    else:  # Netscape: domain, flag, path, secure, expiry, name, value
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 7:
                continue
            cookies.append(
                {
                    "name": f[5],
                    "value": f[6].strip(),
                    "domain": f[0],
                    "path": f[2],
                    "expires": float(f[4] or -1),
                    "httpOnly": False,
                    "secure": f[3].upper() == "TRUE",
                    "sameSite": "Lax",
                }
            )
    fp = [c for c in cookies if "fantasypros" in c["domain"]]
    if not fp:
        sys.exit("no fantasypros.com cookies found in that export")
    RAW.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"cookies": fp, "origins": []}))
    STATE.chmod(0o600)
    names = sorted(c["name"] for c in fp)
    print(f"imported {len(fp)} cookies -> {STATE.relative_to(ROOT)} (git-ignored)")
    print(f"  {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
    cmd_check()


def cmd_check(fmt: str = "half", season: int = 2022, headless: bool = False) -> None:
    """Load one board with the stored session and report whether it is unlocked."""
    _require_state()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        ctx, close = _new_context(pw, headless=headless)
        try:
            d = _scrape(ctx.new_page(), fmt, season)
        finally:
            close()
    state = "SIGNED IN" if d["signed_in"] and d["n_rows"] > FENCE_ROWS else "GATED"
    print(f"{state}: {fmt} {season} rendered {d['n_rows']} rows")
    print(f"  sites: {d['sites']}")
    if state == "GATED":
        sys.exit("session is not signed in — re-run --login")


def cmd_probe(fmts: list[str], seasons: tuple[int, ...], headless: bool = False) -> None:
    """Report row depth per (format, season) WITHOUT caching. Use this to find
    how far back FantasyPros ADP actually goes before committing to a range."""
    _require_state()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        ctx, close = _new_context(pw, headless=headless)
        page = ctx.new_page()
        for fmt in fmts:
            print(f"\n{fmt} ({FORMATS[fmt]}):")
            for season in seasons:
                try:
                    d = _scrape(page, fmt, season)
                except Exception as e:  # a missing year renders no table
                    print(f"  {season}: — ({type(e).__name__})")
                    continue
                gate = "" if d["signed_in"] else "  GATED"
                print(
                    f"  {season}: {d['n_rows']:4d} rows  "
                    f"sites={list(d['sites'].values())}{gate}"
                )
        close()


def cmd_fetch(fmts: list[str], seasons: tuple[int, ...], refresh: bool,
              headless: bool = False) -> None:
    """Cache-first fetch. A cached (format, season) is never re-fetched unless
    --refresh; past boards are frozen artifacts."""
    _require_state()
    from playwright.sync_api import sync_playwright

    RAW.mkdir(parents=True, exist_ok=True)
    todo = [
        (f, s)
        for f in fmts
        for s in seasons
        if refresh or not _payload_path(f, s).exists()
    ]
    if not todo:
        print("nothing to fetch (all cached; use --refresh to force)")
        return
    print(f"fetching {len(todo)} board(s)")
    with sync_playwright() as pw:
        ctx, close = _new_context(pw, headless=headless)
        page = ctx.new_page()
        for fmt, season in todo:
            try:
                d = _scrape(page, fmt, season)
            except Exception as e:
                print(f"  {fmt} {season}: FAILED {type(e).__name__}: {e}")
                continue
            if not d["signed_in"] or d["n_rows"] < MIN_ROWS:
                print(
                    f"  {fmt} {season}: {d['n_rows']} rows "
                    f"(signed_in={d['signed_in']}) — REFUSED, not cached"
                )
                continue
            _payload_path(fmt, season).write_text(json.dumps(d))
            print(f"  {fmt} {season}: cached {d['n_rows']} rows")
        close()


_POS_RE = re.compile(r"^([A-Z]+)(\d+)?$")
_TEAM_RE = re.compile(r"^([A-Z]{2,3})?\s*(?:\((\d+)\))?$")


def load_cached(fmt: str) -> list[dict]:
    out = []
    for p in sorted(RAW.glob(f"{fmt}_*.json")):
        if re.match(rf"{fmt}_\d{{4}}\.json$", p.name):
            out.append(json.loads(p.read_text()))
    return out


def _realtime_index(payload: dict) -> int | None:
    """Column index of the real-time ADP cell, from the captured cell classes.

    The classes are the reliable locator: the header text is a whole sentence
    ("REAL-TIME ADP TAKEN FROM REAL RECENT DRAFTS ACROSS PLATFORMS") and the
    site is free to reword it, but `mcu-table__cell--realtime` is what the
    scrape already keys the per-site columns on.
    """
    classes = payload.get("classes") or []
    return classes.index("realtime") if "realtime" in classes else None


def build_table(fmt: str, realtime: bool = False) -> pl.DataFrame:
    """Cached payloads -> the adp.parquet column contract.

    With `realtime`, seasons in REALTIME_SEASONS take `adp` from the real-time
    column instead of AVG; every other season is untouched. A player the recent
    -draft sample has not taken keeps his AVG, and `adp_col` records per row
    which column the number came from, so a mixed board can never be mistaken
    for a uniform one.
    """
    payloads = load_cached(fmt)
    if not payloads:
        sys.exit(f"no cached {fmt} boards — run --fetch first")
    rows = []
    for d in payloads:
        idx = {h.upper(): i for i, h in enumerate(d["headers"])}
        i_pos, i_avg = idx.get("POS"), idx.get("AVG")
        n_sites = len(d["sites"])
        use_rt = realtime and d["season"] in REALTIME_SEASONS
        i_rt = _realtime_index(d) if use_rt else None
        if use_rt and i_rt is None:
            sys.exit(
                f"{fmt} {d['season']}: --realtime asked for, but the cached "
                f"capture has no `realtime` cell class (columns: "
                f"{d.get('classes')}). Re-fetch that season with --refresh."
            )
        for pp in d["players"]:
            cells = pp["cells"]
            if i_avg is None or i_avg >= len(cells):
                continue
            try:
                adp = float(cells[i_avg].replace(",", ""))
            except ValueError:
                continue
            adp_col = "avg"
            if i_rt is not None and i_rt < len(cells):
                m_rt = _REALTIME_RE.match(cells[i_rt].strip())
                if m_rt:
                    adp, adp_col = float(m_rt.group(1)), "realtime"
            pos_cell = cells[i_pos] if i_pos is not None and i_pos < len(cells) else ""
            m = _POS_RE.match(pos_cell)
            position = m.group(1) if m else pos_cell
            pos_rank = int(m.group(2)) if m and m.group(2) else None
            tb = _TEAM_RE.match(pp.get("team_bye") or "")
            rows.append(
                {
                    "season": d["season"],
                    "name": pp["name"],
                    "position": position,
                    "team": tb.group(1) if tb else None,
                    "bye": int(tb.group(2)) if tb and tb.group(2) else None,
                    "adp": adp,
                    "adp_col": adp_col,
                    "pos_rank": pos_rank,
                    "fantasypros_id": pp["fp_id"],
                    "sites_n": n_sites,
                }
            )
    adp = pl.DataFrame(rows).filter(
        pl.col("position").is_in(KEEP_POS) & pl.col("name").is_not_null()
    )
    adp = adp.with_columns(
        pl.col("name").map_elements(norm_name, return_dtype=pl.Utf8).alias("norm_name")
    )
    # deterministic order: adp, then name, so the ordinal rank is reproducible
    adp = adp.sort(["season", "adp", "name"]).with_columns(
        pl.int_range(1, pl.len() + 1).over("season").alias("adp_rank")
    )
    return adp


def join_gsis(adp: pl.DataFrame) -> pl.DataFrame:
    """id-first, then the same name ladder bff.ecr uses. The fp id makes this a
    much stronger join than the FFC boards ever had (which lost every "Mike
    Williams" row to name ambiguity)."""
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
        .join(adp.select("norm_name", "position", "season").unique(),
              on=["norm_name", "position"], how="inner")
        .filter(pl.col("draft_year").is_null()
                | (pl.col("draft_year") <= pl.col("season")))
        .group_by("norm_name", "position", "season")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("g_era"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "position", "season", "g_era")
    )
    by_np = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .group_by("norm_name", "position")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("g_np"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "position", "g_np")
    )
    out = (
        adp.join(by_fp, on="fantasypros_id", how="left")
        .join(by_era, on=["norm_name", "position", "season"], how="left")
        .join(by_np, on=["norm_name", "position"], how="left")
    )
    return out.with_columns(
        pl.coalesce("g_fp", "g_era", "g_np").alias("gsis_id")
    ).drop("g_fp", "g_era", "g_np")


def _realtime_report(adp: pl.DataFrame, fmt: str) -> None:
    """What the real-time column actually changed, per season it applied to.

    Printed rather than merely counted because this is a LIVE number: the same
    command an hour later writes a different board, so the run has to say what
    it wrote and when the underlying capture was taken.
    """
    for season in sorted(adp.filter(pl.col("adp_col") == "realtime")
                         ["season"].unique().to_list()):
        s = adp.filter(pl.col("season") == season)
        n_rt = int((s["adp_col"] == "realtime").sum())
        fell_back = s.filter(pl.col("adp_col") == "avg")
        cap = RAW / f"{fmt}_{season}.json"
        fetched = json.loads(cap.read_text()).get("fetched") if cap.exists() else "?"
        print(f"\n  {season} REAL-TIME ADP: {n_rt}/{s.height} rows from the "
              f"real-time column, {fell_back.height} kept AVG "
              f"(no recent-draft sample)")
        print(f"    capture fetched {fetched} — this board ages, re-fetch before "
              f"you draft")
        if fell_back.height:
            names = fell_back.sort("adp")["name"].to_list()
            print(f"    AVG fallbacks (deepest board rows first): "
                  f"{', '.join(names[:6])}{' ...' if len(names) > 6 else ''}")


def cmd_build(fmts: list[str], realtime: bool = False) -> None:
    for fmt in fmts:
        adp = join_gsis(build_table(fmt, realtime=realtime))
        out = PROC / f"fp_adp_{fmt}.parquet"
        adp.write_parquet(out)
        print(f"\n{fmt}: wrote {out.relative_to(ROOT)} ({adp.height} rows)")
        per = (
            adp.group_by("season")
            .agg(
                pl.len().alias("n"),
                pl.col("adp").max().round(1).alias("max_adp"),
                pl.col("sites_n").first().alias("sites"),
                pl.col("gsis_id").is_not_null().mean().round(4).alias("gsis"),
                (pl.col("adp_col") == "realtime").any().alias("rt"),
            )
            .sort("season")
        )
        for r in per.iter_rows(named=True):
            print(
                f"  {r['season']}: n={r['n']:4d} max_adp={r['max_adp']:6.1f} "
                f"sites={r['sites']} gsis={r['gsis']:.1%}"
                f"{'  <- REAL-TIME' if r['rt'] else ''}"
            )
        if realtime:
            _realtime_report(adp, fmt)


def _seasons(arg: str | None) -> tuple[int, ...]:
    if not arg:
        return DEFAULT_SEASONS
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
    ap.add_argument("--login", action="store_true", help="one-time headed sign-in")
    ap.add_argument("--import-cookies", metavar="FILE", default=None,
                    help="fallback: cookies.txt / JSON export from YOUR browser")
    ap.add_argument("--headless", action="store_true",
                    help="run without a window (expect Cloudflare challenges)")
    ap.add_argument("--check", action="store_true", help="is the stored session live?")
    ap.add_argument("--probe", action="store_true", help="report depth, cache nothing")
    ap.add_argument("--fetch", action="store_true", help="cache raw boards")
    ap.add_argument("--build", action="store_true", help="cached JSON -> parquet")
    ap.add_argument("--realtime", action="store_true",
                    help="take adp from the board's REAL-TIME column (recent "
                         f"drafts) instead of AVG, for {sorted(REALTIME_SEASONS)} "
                         "only; other seasons keep AVG")
    ap.add_argument("--format", default="half",
                    help=f"{'|'.join(FORMATS)}|all, or a comma list "
                         f"(e.g. half,ppr). Default half")
    ap.add_argument("--seasons", default=None,
                    help="e.g. 2012-2026 or 2022,2025 (default 2012-2026)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch even if cached (only for the live season)")
    a = ap.parse_args()

    fmts = list(FORMATS) if a.format == "all" else [
        f.strip() for f in a.format.split(",") if f.strip()
    ]
    for f in fmts:
        if f not in FORMATS:
            sys.exit(f"unknown format {f!r}; known: {', '.join(FORMATS)}|all")
    seasons = _seasons(a.seasons)

    if a.login:
        cmd_login()
    if a.import_cookies:
        cmd_import_cookies(a.import_cookies)
    if a.check:
        cmd_check(headless=a.headless)
    if a.probe:
        cmd_probe(fmts, seasons, headless=a.headless)
    if a.fetch:
        cmd_fetch(fmts, seasons, a.refresh, headless=a.headless)
    if a.build:
        cmd_build(fmts, realtime=a.realtime)
    if not any((a.login, a.import_cookies, a.check, a.probe, a.fetch, a.build)):
        ap.print_help()


if __name__ == "__main__":
    main()
