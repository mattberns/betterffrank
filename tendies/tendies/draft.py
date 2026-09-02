"""Turn raw ESPN league payloads into tidy per-pick and per-team frames.

One row per draft pick, per season. Snake and auction leagues both land here:
`bid_amount` is 0 in a snake league, `overall_pick` is the nomination order in
an auction, and `draft_type` on the season meta says which you are reading.
"""

from __future__ import annotations

import polars as pl

from .espn import PRO_TEAMS, EspnClient, SeasonUnavailable

PICK_SCHEMA = {
    "season": pl.Int64,
    "league_id": pl.Int64,
    "team_id": pl.Int64,
    "overall_pick": pl.Int64,
    "round": pl.Int64,
    "pick_in_round": pl.Int64,
    "bid_amount": pl.Int64,
    "keeper": pl.Boolean,
    "auto_pick": pl.Boolean,
    "player_id": pl.Int64,
    "player_name": pl.Utf8,
    "position": pl.Utf8,
    "pro_team": pl.Utf8,
}


def _unwrap(payload):
    return payload[0] if isinstance(payload, list) else payload


def team_name(t: dict) -> str:
    name = (t.get("name") or "").strip()
    if not name:
        name = f"{(t.get('location') or '').strip()} {(t.get('nickname') or '').strip()}".strip()
    return name or f"Team {t.get('id')}"


def season_meta(payload: dict, season: int, league_id: int) -> dict:
    settings = payload.get("settings") or {}
    draft = settings.get("draftSettings") or {}
    return {
        "season": season,
        "league_id": league_id,
        "league_name": settings.get("name"),
        "teams": settings.get("size") or len(payload.get("teams") or []),
        "draft_type": draft.get("type"),
        "draft_date": draft.get("date"),
        "keeper_count": draft.get("keeperCount"),
    }


def teams_frame(payload: dict, season: int, league_id: int) -> pl.DataFrame:
    members = {m.get("id"): m for m in (payload.get("members") or [])}
    rows = []
    for t in payload.get("teams") or []:
        owners = [o for o in (t.get("owners") or []) if o]
        primary = t.get("primaryOwner") or (owners[0] if owners else None)
        m = members.get(primary, {})
        full = " ".join(x for x in (m.get("firstName"), m.get("lastName")) if x).strip()
        rows.append(
            {
                "season": season,
                "league_id": league_id,
                "team_id": int(t["id"]),
                "team_name": team_name(t),
                "abbrev": t.get("abbrev"),
                "primary_owner": primary,
                "owners": owners,
                "manager": (m.get("displayName") or full or None),
                "manager_name": full or None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "season": pl.Int64, "league_id": pl.Int64, "team_id": pl.Int64,
            "team_name": pl.Utf8, "abbrev": pl.Utf8, "primary_owner": pl.Utf8,
            "owners": pl.List(pl.Utf8), "manager": pl.Utf8, "manager_name": pl.Utf8,
        },
    )


def _dst_fallback(player_id: int) -> dict | None:
    """ESPN codes team defenses as negative ids: -16000 - proTeamId."""
    if player_id >= 0:
        return None
    pro = abs(player_id) - 16000
    if pro in PRO_TEAMS:
        return {"name": f"{PRO_TEAMS[pro]} D/ST", "position": "D/ST", "pro_team": PRO_TEAMS[pro]}
    return None


def picks_frame(client: EspnClient, payload: dict, season: int) -> pl.DataFrame:
    detail = payload.get("draftDetail") or {}
    # A scheduled-but-unheld draft comes back fully formed with playerId -1 on
    # every pick, so "the payload has picks" is not "the draft happened".
    picks = [
        p for p in (detail.get("picks") or [])
        if int(p.get("playerId", -1)) > 0 or _dst_fallback(int(p.get("playerId", -1)))
    ]
    if not picks or detail.get("drafted") is False:
        return pl.DataFrame(schema=PICK_SCHEMA)

    universe = client.players(season)
    missing = sorted({int(p["playerId"]) for p in picks} - set(universe))
    if missing:
        universe |= client.players_by_id(season, [i for i in missing if i > 0])
    for pid in missing:
        if pid not in universe:
            fb = _dst_fallback(pid)
            if fb:
                universe[pid] = fb

    rows = []
    for p in picks:
        pid = int(p["playerId"])
        info = universe.get(pid, {})
        rows.append(
            {
                "season": season,
                "league_id": client.league_id,
                "team_id": int(p["teamId"]),
                "overall_pick": int(p.get("overallPickNumber") or 0),
                "round": int(p.get("roundId") or 0),
                "pick_in_round": int(p.get("roundPickNumber") or 0),
                "bid_amount": int(p.get("bidAmount") or 0),
                "keeper": bool(p.get("keeper")),
                "auto_pick": bool(p.get("autoDraftTypeId")),
                "player_id": pid,
                "player_name": info.get("name"),
                "position": info.get("position"),
                "pro_team": info.get("pro_team"),
            }
        )
    return pl.DataFrame(rows, schema=PICK_SCHEMA).sort("overall_pick")


def collect(client: EspnClient, seasons: list[int]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(picks, teams, meta) across seasons; a season with no draft is skipped."""
    picks, teams, meta = [], [], []
    for season in seasons:
        try:
            payload = _unwrap(client.league(season))
        except SeasonUnavailable:
            print(f"  {season}: no league on ESPN under this id — skipped")
            continue
        p = picks_frame(client, payload, season)
        if p.is_empty():
            print(f"  {season}: draft not held yet — skipped")
            continue
        t = teams_frame(payload, season, client.league_id)
        picks.append(p)
        teams.append(t)
        meta.append(season_meta(payload, season, client.league_id))
        print(f"  {season}: {p.height} picks, {t.height} teams")
    if not picks:
        raise RuntimeError("No draft data for any requested season.")
    return pl.concat(picks), pl.concat(teams), pl.DataFrame(meta)
