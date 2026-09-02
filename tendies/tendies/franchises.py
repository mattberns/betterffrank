"""Link team-seasons into franchises across years.

ESPN's `team_id` is NOT a stable identity: it is a slot in that season's
league, reused when a manager leaves. The stable key is the owner's SWID GUID,
which follows a person across seasons and across renamed teams. So franchises
are the connected components of the graph "these two team-seasons share an
owner GUID" — which also survives co-ownership and a co-owner taking over.

Fallbacks, in order, when a payload carries no owners (rare, old seasons):
same team_id in adjacent seasons, then same team name. Both are announced.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import polars as pl


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def link(teams: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    rows = teams.sort(["season", "team_id"]).to_dicts()
    dsu = _DSU(len(rows))

    by_owner: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        for owner in r["owners"] or ([r["primary_owner"]] if r["primary_owner"] else []):
            by_owner[owner].append(i)
    for members in by_owner.values():
        for j in members[1:]:
            dsu.union(members[0], j)

    ownerless = [i for i, r in enumerate(rows) if not (r["owners"] or r["primary_owner"])]
    if ownerless:
        if verbose:
            seasons = sorted({rows[i]["season"] for i in ownerless})
            print(f"  {len(ownerless)} team-seasons have no owner GUID {seasons}: "
                  "linking those by team_id, then team name")
        by_team: dict[int, list[int]] = defaultdict(list)
        by_name: dict[str, list[int]] = defaultdict(list)
        for i in ownerless:
            by_team[rows[i]["team_id"]].append(i)
            by_name[(rows[i]["team_name"] or "").lower()].append(i)
        for group in list(by_team.values()) + list(by_name.values()):
            for j in group[1:]:
                dsu.union(group[0], j)

    # Stable ids: order components by their earliest season, then team_id.
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        comps[dsu.find(i)].append(i)
    order = sorted(comps, key=lambda root: (rows[min(comps[root])]["season"],
                                            rows[min(comps[root])]["team_id"]))
    fid = {root: f"F{n:02d}" for n, root in enumerate(order, start=1)}

    # Label with the real name when ESPN has one: half these display names are
    # auto-generated ("ESPNFAN0746322618"), which is useless in a report.
    labels = {}
    for root, members in comps.items():
        latest = max(members, key=lambda i: rows[i]["season"])
        seen = [rows[i]["manager_name"] or rows[i]["manager"] for i in members]
        seen = [n for n in seen if n]
        label = (
            rows[latest]["manager_name"]
            or rows[latest]["manager"]
            or (Counter(seen).most_common(1)[0][0] if seen else None)
            or rows[latest]["team_name"]
        )
        labels[root] = label

    out = teams.sort(["season", "team_id"]).with_columns(
        franchise_id=pl.Series([fid[dsu.find(i)] for i in range(len(rows))]),
        franchise_label=pl.Series([labels[dsu.find(i)] for i in range(len(rows))]),
    )
    if verbose:
        dupes = [n for n, c in Counter(labels.values()).items() if c > 1]
        if dupes:
            # Two owner GUIDs, one human: a second ESPN account, not a bug to
            # paper over silently. Left unmerged; merge downstream if you want.
            print(f"  NOTE: separate franchises share a manager name {dupes} — "
                  "one person with two ESPN accounts; left as separate franchises")
        n_seasons = out["season"].n_unique()
        span = (
            out.group_by("franchise_id").agg(pl.col("season").n_unique().alias("n"))
            ["n"].value_counts().sort("n")
        )
        print(f"  {out['franchise_id'].n_unique()} franchises over {n_seasons} seasons; "
              f"seasons-per-franchise: {dict(zip(span['n'], span['count']))}")
    return out


def summary(teams_linked: pl.DataFrame) -> pl.DataFrame:
    return (
        teams_linked.group_by("franchise_id", "franchise_label")
        .agg(
            first_season=pl.col("season").min(),
            last_season=pl.col("season").max(),
            n_seasons=pl.col("season").n_unique(),
            managers=pl.col("manager").drop_nulls().unique().sort(),
            team_names=pl.col("team_name").unique().sort(),
            team_ids=pl.col("team_id").unique().sort(),
        )
        .sort("franchise_id")
    )
