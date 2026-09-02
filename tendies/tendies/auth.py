"""Get ESPN's private-league cookies (`espn_s2`, `SWID`) and keep them.

The browser is used for AUTHENTICATION ONLY. Once the two cookies are in hand
every later run is a plain HTTPS call, so a refresh of eight seasons needs no
GUI. Resolution order:

  1. ESPN_S2 / ESPN_SWID environment variables (CI, headless boxes)
  2. .auth/espn_cookies.json from a previous login, if it still authenticates
  3. a headed Chrome window (Playwright, persistent profile under .auth/)
     where the user signs in to ESPN once

The persistent profile means step 3 usually happens once per machine: ESPN's
login survives in the profile, so a later re-login is a page load, not a form.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import COOKIE_FILE, PROFILE_DIR
from .espn import AuthError, EspnClient

COOKIE_NAMES = ("espn_s2", "SWID")
# Force a visible, foreground window: on WSLg a default-placed Chrome can open
# behind the editor, which reads as "nothing happened".
WINDOW_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--window-position=60,60",
    "--window-size=1500,950",
]


def league_url(league_id: int, season: int) -> str:
    return (
        "https://fantasy.espn.com/football/league/draftrecap"
        f"?seasonId={season}&leagueId={league_id}"
    )


def from_env() -> dict[str, str] | None:
    s2, swid = os.environ.get("ESPN_S2"), os.environ.get("ESPN_SWID")
    if s2 and swid:
        return {"espn_s2": s2, "SWID": swid}
    return None


def load_saved(path: Path = COOKIE_FILE) -> dict[str, str] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if all(k in data for k in COOKIE_NAMES) else None


def save(cookies: dict[str, str], path: Path = COOKIE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cookies, indent=2))
    path.chmod(0o600)


def works(cookies: dict[str, str] | None, league_id: int, season: int) -> bool:
    """Cheapest possible authorization probe against the real league."""
    if not cookies:
        return False
    try:
        EspnClient(league_id, cookies, refresh=True).league(season, views=("mTeam",))
        return True
    except AuthError:
        return False
    except Exception:
        return False


def get_cookies(
    league_id: int,
    season: int,
    interactive: bool = True,
    force_login: bool = False,
    profile_dir: Path = PROFILE_DIR,
    timeout: int = 900,
) -> dict[str, str]:
    if not force_login:
        for source in (from_env(), load_saved()):
            if works(source, league_id, season):
                return source
    if not interactive:
        raise AuthError(
            "No working ESPN cookies and interactive login is off. "
            "Run `uv run python -m tendies login` or set ESPN_S2 / ESPN_SWID."
        )
    cookies = browser_login(league_id, season, profile_dir=profile_dir, timeout=timeout)
    save(cookies)
    return cookies


def browser_login(
    league_id: int,
    season: int,
    profile_dir: Path = PROFILE_DIR,
    timeout: int = 900,
) -> dict[str, str]:
    """Open Chrome, wait for the ESPN session cookies to appear, return them."""
    from playwright.sync_api import sync_playwright

    profile_dir.mkdir(parents=True, exist_ok=True)
    url = league_url(league_id, season)
    print(f"Opening Chrome at {url}", flush=True)
    print("Sign in to ESPN in that window if prompted; this returns on its own.", flush=True)

    with sync_playwright() as p:
        try:
            ctx = p.chromium.launch_persistent_context(
                str(profile_dir), headless=False, channel="chrome",
                args=WINDOW_ARGS, no_viewport=True,
            )
        except Exception:
            # No system Chrome: fall back to Playwright's own build
            # (needs `uv run playwright install chromium` once).
            ctx = p.chromium.launch_persistent_context(
                str(profile_dir), headless=False, args=WINDOW_ARGS, no_viewport=True,
            )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url, wait_until="domcontentloaded")
        page.bring_to_front()
        start = time.time()
        deadline = start + timeout
        found: dict[str, str] = {}
        while time.time() < deadline:
            jar = {c["name"]: c["value"] for c in ctx.cookies()}
            if int(time.time() - start) % 30 == 0:
                print(f"  waiting for ESPN login... {int(deadline - time.time())}s left",
                      flush=True)
            if all(jar.get(n) for n in COOKIE_NAMES):
                found = {n: jar[n] for n in COOKIE_NAMES}
                if works(found, league_id, season):
                    break
            time.sleep(2)
        ctx.close()

    if not found:
        raise AuthError(
            f"Timed out after {timeout}s without ESPN cookies. "
            "Log in inside the Chrome window and rerun `login`."
        )
    print("Got ESPN cookies; saved to .auth/espn_cookies.json")
    return found
