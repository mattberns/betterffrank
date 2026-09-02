"""Read-only ESPN fantasy-football API client.

Two facts shape this module:

1. A private league needs the browser cookies `espn_s2` and `SWID`; without
   them the league endpoint answers 401 AUTH_LEAGUE_NOT_VISIBLE. See auth.py.
2. ESPN serves the CURRENT season and PAST seasons from different paths. Past
   seasons live under `leagueHistory` and come back wrapped in a list. We try
   the season path first and fall back, so a caller just asks for a season.

The season-level PLAYER UNIVERSE (`/seasons/{y}/players`) is public — no
cookies — which is how pick playerIds become names without touching a roster
view.

Everything fetched is cached under data/raw/espn/<league_id>/ as the exact
JSON the API returned, so reruns are offline and the raw payload stays
auditable. Pass refresh=True to re-fetch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from .config import RAW

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)

# defaultPositionId -> fantasy position. Only the ids a fantasy board can hold;
# anything else (IDP ids 9-13, punters) falls through to "OTHER".
POSITIONS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}

PRO_TEAMS = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL",
    7: "DEN", 8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV",
    14: "LAR", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB",
    28: "WSH", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

DRAFT_VIEWS = ("mDraftDetail", "mTeam", "mSettings")


class AuthError(RuntimeError):
    """The league is private and the cookies are missing or stale."""


class SeasonUnavailable(RuntimeError):
    """ESPN has no league record for that season (it predates the league)."""


class TransientError(RuntimeError):
    """ESPN throttled or failed. Nothing is cached; rerun the command."""


class EspnClient:
    def __init__(
        self,
        league_id: int,
        cookies: dict[str, str] | None = None,
        cache_dir: Path | None = None,
        refresh: bool = False,
        pause: float = 0.4,
    ) -> None:
        self.league_id = int(league_id)
        self.refresh = refresh
        self.pause = pause
        self.cache = (cache_dir or RAW / "espn") / str(self.league_id)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.players_cache = (cache_dir or RAW / "espn") / "players"
        self.players_cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        if cookies:
            for k, v in cookies.items():
                self.session.cookies.set(k, v, domain=".espn.com")

    # ---- plumbing ---------------------------------------------------------

    def _get(self, url: str, params: dict | None = None, headers: dict | None = None,
             tries: int = 4):
        """GET with backoff. 401/403 is auth, 404 is a real answer, the rest
        retries — a throttled response that slipped through used to be cached
        as if it were data."""
        last = None
        for attempt in range(tries):
            time.sleep(self.pause)
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=45)
            except requests.RequestException as exc:
                last = exc
                time.sleep(2**attempt)
                continue
            if r.status_code in (401, 403):
                raise AuthError(
                    f"ESPN returned {r.status_code} for {r.url}\n"
                    "The league is private. Run:  uv run python -m tendies login\n"
                    "(or set ESPN_S2 / ESPN_SWID in the environment)."
                )
            if r.status_code == 429 or r.status_code >= 500:
                last = requests.HTTPError(f"{r.status_code} for {r.url}", response=r)
                time.sleep(2**attempt + 1)
                continue
            r.raise_for_status()
            return r.json()
        raise TransientError(f"ESPN kept failing after {tries} tries: {last}")

    def _cached(self, path: Path, fetch, guard=None):
        if path.exists() and not self.refresh:
            payload = json.loads(path.read_text())
            if guard is None or guard(payload):
                return payload
            path.unlink()          # a bad cache entry is worse than none
        payload = fetch()
        if guard is not None and not guard(payload):
            raise TransientError(
                f"ESPN returned an incoherent payload for {path.name} "
                "(claims a completed draft with no picks). Nothing cached; rerun."
            )
        path.write_text(json.dumps(payload))
        return payload

    # ---- league -----------------------------------------------------------

    def league(self, season: int, views: tuple[str, ...] = DRAFT_VIEWS) -> dict:
        """One season of league data. Season path first, leagueHistory second."""
        path = self.cache / f"league_{season}.json"

        def fetch():
            params = [("view", v) for v in views]
            try:
                payload = self._get(
                    f"{BASE}/seasons/{season}/segments/0/leagues/{self.league_id}",
                    params=params,
                )
                if _has_draft(payload):
                    return payload
            except AuthError:
                raise
            except requests.HTTPError:
                payload = None
            try:
                hist = self._get(
                    f"{BASE}/leagueHistory/{self.league_id}",
                    params=params + [("seasonId", season)],
                )
            except requests.HTTPError:
                if payload is None:
                    # Neither path knows the season: the league did not exist.
                    raise SeasonUnavailable(season) from None
                return payload
            if isinstance(hist, list):
                hist = next((h for h in hist if int(h.get("seasonId", season)) == season), hist[0])
            return hist if _has_draft(hist) or payload is None else payload

        payload = self._cached(path, fetch, guard=_coherent)
        return payload

    # ---- player universe (public) ----------------------------------------

    def players(self, season: int) -> dict[int, dict]:
        """playerId -> {name, position, pro_team} for one season."""
        path = self.players_cache / f"players_{season}.json"

        def fetch():
            return self._get(
                f"{BASE}/seasons/{season}/players",
                params={"scoringPeriodId": 0, "view": "players_wl"},
                headers={"x-fantasy-filter": json.dumps({"filterActive": {"value": True}})},
            )

        return {int(p["id"]): _player_row(p) for p in self._cached(path, fetch)}

    def players_by_id(self, season: int, ids: list[int]) -> dict[int, dict]:
        """Gap-filler for ids the season universe omits (retired/inactive)."""
        if not ids:
            return {}
        out: dict[int, dict] = {}
        for i in range(0, len(ids), 300):
            chunk = [int(x) for x in ids[i : i + 300]]
            payload = self._get(
                f"{BASE}/seasons/{season}/players",
                params={"scoringPeriodId": 0, "view": "players_wl"},
                headers={
                    "x-fantasy-filter": json.dumps(
                        {"players": {"filterIds": {"value": chunk}}}
                    )
                },
            )
            for p in payload:
                out[int(p["id"])] = _player_row(p)
        return out


def _player_row(p: dict) -> dict:
    return {
        "name": p.get("fullName")
        or " ".join(x for x in (p.get("firstName"), p.get("lastName")) if x).strip(),
        "position": POSITIONS.get(p.get("defaultPositionId"), "OTHER"),
        "pro_team": PRO_TEAMS.get(p.get("proTeamId"), None),
    }


def _coherent(payload) -> bool:
    """A payload that says the draft happened must carry picks."""
    p = payload[0] if isinstance(payload, list) and payload else payload
    detail = (p or {}).get("draftDetail") or {}
    return not (detail.get("drafted") and not detail.get("picks"))


def _has_draft(payload) -> bool:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return bool((payload or {}).get("draftDetail", {}).get("picks"))
