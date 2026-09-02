"""Command line: `uv run python -m tendies <command>` from the tendies/ folder.

  login   open Chrome once and store the ESPN session cookies
  fetch   pull + cache the raw league JSON for a range of seasons
  build   fetch -> tidy picks -> link franchises -> join ADP -> write dataset
  report  read the built dataset and print a few tendency cuts (sanity check)
  streaming  derive each position's replacement level (the record behind REPL_RANKS)

Every command takes --league-id / --first-season / --last-season, so pointing
this at another league is a flag, not an edit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from . import adp_match, draft, franchises
from .auth import browser_login, get_cookies, save
from .config import (
    DEFAULT_ADP_PARQUET, DEFAULT_FALLBACK_BOARDS, DEFAULT_FIRST_SEASON,
    DEFAULT_LAST_SEASON, DEFAULT_LEAGUE_ID, PROCESSED,
)
from .espn import EspnClient

# identity, then the pick, then the player; the ADP columns are appended by
# adp_match.board_columns() since they depend on which boards were attached.
BASE_COLUMNS = [
    "season", "league_id", "franchise_id", "franchise_label", "manager",
    "team_id", "team_name",
    "overall_pick", "round", "pick_in_round", "bid_amount", "keeper", "auto_pick",
    "player_id", "player_name", "position", "pro_team", "gsis_id",
]


def _parse_board(spec: str) -> tuple[str, Path]:
    """`path` or `path=tag`; the tag names that board's columns."""
    path, _, tag = str(spec).partition("=")
    p = Path(path)
    return (tag or p.stem.replace("fp_adp_", "").replace("adp_", "")), p


def _seasons(args) -> list[int]:
    return list(range(args.first_season, args.last_season + 1))


def _client(args) -> EspnClient:
    seasons = _seasons(args)
    cookies = get_cookies(
        args.league_id, seasons[-1], interactive=not args.no_browser,
        force_login=getattr(args, "relogin", False),
    )
    return EspnClient(args.league_id, cookies, refresh=args.refresh)


def cmd_login(args) -> None:
    cookies = browser_login(args.league_id, args.last_season)
    save(cookies)


def cmd_fetch(args) -> None:
    client = _client(args)
    for season in _seasons(args):
        payload = draft._unwrap(client.league(season))
        picks = (payload.get("draftDetail") or {}).get("picks") or []
        print(f"  {season}: {len(picks)} picks cached")


def cmd_build(args) -> None:
    seasons = _seasons(args)
    client = _client(args)
    print(f"League {args.league_id}, seasons {seasons[0]}-{seasons[-1]}")

    picks, teams, meta = draft.collect(client, seasons)
    teams = franchises.link(teams)

    joined = picks.join(
        teams.select("season", "team_id", "franchise_id", "franchise_label",
                     "team_name", "manager"),
        on=["season", "team_id"], how="left",
    )
    print(f"ADP board (primary, '{args.primary_tag}'): {args.adp}")
    board = adp_match.load_board(args.adp, seasons)
    secondaries = []
    if not args.no_fallback:
        for tag, path in (_parse_board(s) for s in args.adp_fallback):
            if not path.exists():
                print(f"  fallback board missing, skipped: {path} — '{tag}' columns "
                      "will be absent and adp_rank_filled has no backstop where the "
                      "primary board drops a player")
                continue
            print(f"ADP board (fallback, '{tag}' columns): {path}")
            secondaries.append((tag, adp_match.load_board(path, seasons)))
    joined = adp_match.attach(joined, board, secondaries, primary_tag=args.primary_tag)
    cols = BASE_COLUMNS + adp_match.board_columns([t for t, _ in secondaries])
    joined = joined.select([c for c in cols if c in joined.columns]).sort(
        "season", "overall_pick"
    )

    _verify(joined)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    stem = f"draft_picks_{args.league_id}"
    joined.write_parquet(PROCESSED / f"{stem}.parquet")
    joined.write_csv(PROCESSED / f"{stem}.csv")
    fr = franchises.summary(teams)
    fr.with_columns(
        pl.col(pl.List(pl.Utf8), pl.List(pl.Int64)).cast(pl.List(pl.Utf8)).list.join("; ")
    ).write_csv(
        PROCESSED / f"franchises_{args.league_id}.csv"
    )
    meta.write_csv(PROCESSED / f"seasons_{args.league_id}.csv")

    print(f"\nwrote {PROCESSED/stem}.parquet ({joined.height} picks) + .csv")
    print(f"wrote franchises_{args.league_id}.csv ({fr.height} franchises), "
          f"seasons_{args.league_id}.csv")
    print("\nADP coverage (skill positions only; primary_* is the primary board, "
          "filled_rate adds the fallback chain):")
    print(adp_match.coverage(joined))
    miss = adp_match.unmatched(joined, max_pick=args.miss_through)
    if miss.height:
        print(f"\nunmatched on EVERY board inside the first {args.miss_through} picks "
              f"({miss.height}) — each is either off-board or an ALIASES entry:")
        print(miss)


def _verify(df: pl.DataFrame) -> None:
    """Invariants worth crashing over — a silently wrong row is the failure
    mode that matters here, not a missing one."""
    dupes = df.group_by("season", "overall_pick").len().filter(pl.col("len") > 1)
    assert dupes.is_empty(), f"duplicate picks:\n{dupes}"
    for col in ("franchise_id", "team_name", "player_name", "position"):
        n = df[col].null_count()
        assert n == 0, f"{n} null {col} values"
    counts = df.group_by("season", "team_id").len().rename({"len": "n"})
    uneven = (
        counts.group_by("season")
        .agg(distinct=pl.col("n").n_unique(), fewest=pl.col("n").min(), most=pl.col("n").max())
        .filter(pl.col("distinct") > 1)
        .sort("season")
    )
    if not uneven.is_empty():
        print(f"  NOTE: uneven picks per team in some seasons (keepers or a "
              f"forfeited pick will do this):\n{uneven}")


def cmd_ecr_check(args) -> None:
    """Audit the half-PPR series against evidence before trusting its labels.

    Exits non-zero on a failure so this can gate a rebuild. The asserts only
    bind on seasons PINNED to half in config; a revised season that is pinned to
    ppr is reported for information, because it is not pricing anything.
    """
    import sys

    from . import dataset
    from .config import ECR_SOURCE_BY_SEASON
    seasons = None
    if args.seasons:
        a, _, b = args.seasons.partition("-")
        seasons = list(range(int(a), int(b or a) + 1))
    fails, warns = dataset.check_ecr(seasons=seasons)
    pinned = sorted(s for s, t in ECR_SOURCE_BY_SEASON.items() if t == "half")
    print(f"\npinned to half: {dataset._runs(pinned) if pinned else 'none'}")
    for w in warns:
        print(f"  warning: {w}")
    for f in fails:
        print(f"  FAIL: {f}")
    print(f"\n{len(fails)} failure(s), {len(warns)} warning(s)")
    if fails:
        sys.exit(1)


def cmd_report(args) -> None:
    """Dumb parquet read, on purpose: a sanity check on what `build` wrote.

    The per-franchise cut here is HUMAN PICKS ONLY. It used to include
    autodraft, which for three managers is half their picks, and REPORT.md
    records that as the error that made the first tendency analysis wrong.
    `tendies traits` is the measured version of this question.
    """
    path = PROCESSED / f"draft_picks_{args.league_id}.parquet"
    df = pl.read_parquet(path)
    print(f"{path.name}: {df.height} picks, seasons "
          f"{df['season'].min()}-{df['season'].max()}\n")
    print("Position mix by round (share of picks):")
    print(
        df.filter(pl.col("round") <= 8)
        .group_by("round", "position").agg(n=pl.len())
        .with_columns(share=(pl.col("n") / pl.col("n").sum().over("round")).round(2))
        .pivot(on="position", index="round", values="share").sort("round").fill_null(0)
    )
    print("\nBy franchise (human picks only): how far from ADP they draft "
          "(+ = later than market / value, - = reach):")
    print(
        df.filter(~pl.col("auto_pick") & pl.col("adp_rank_filled").is_not_null())
        .group_by("franchise_id", "franchise_label")
        .agg(
            seasons=pl.col("season").n_unique(),
            picks=pl.len(),
            mean_adp_delta=pl.col("adp_delta_filled").mean().round(1),
            median_adp_delta=pl.col("adp_delta_filled").median(),
            reach_rate=(pl.col("adp_delta_filled") < -10).mean().round(3),
        )
        .sort("mean_adp_delta")
    )
    print("\n`tendies traits` measures these properly — split-half reliability "
          "across each manager's own seasons, plus the rookie and timing cuts.")


def cmd_streaming(args) -> None:
    """The derivation behind vorp.REPL_RANKS. Prints; writes nothing."""
    from . import streaming
    streaming.main(args.league_id)


def cmd_traits(args) -> None:
    """Per-manager tendencies, with the reliability that says which are real."""
    from . import attrs, dataset, traits
    seasons = dataset.seasons_available()
    ecr, _ = dataset.load_ecr(on_undeclared=getattr(args, "ecr_undeclared", "fail"))
    raw_picks = dataset.load_picks(args.league_id)
    # prev_owner needs the picks, so it is attached here the same way build_all
    # attaches it; the VORP curve is not needed for any trait.
    boards = attrs.attach_prev_owner(
        dataset.build_boards(seasons, ecr=ecr), raw_picks
    )
    picks = dataset.link(boards, raw_picks)
    picks = traits.attach_board(picks, boards)

    human = picks.filter(~pl.col("auto_pick"))
    print(f"\n{picks.height} picks, {human.height} of them human, seasons "
          f"{picks['season'].min()}-{picks['season'].max()}")
    print(f"rookies: {int(human['rookie'].sum())} of {human.height} human picks "
          f"({human['rookie'].mean():.1%})")

    avg = traits.league_average(picks)
    print("\nLeague average:")
    for name, v in avg.items():
        print(f"  {name:16s} {'—' if v is None else f'{v:.3f}'}")

    print("\nPer manager:")
    print(traits.per_manager(picks))

    print(f"\nSplit-half reliability (odd vs even seasons of each manager's own "
          f"career, both halves carrying >= {args.min_picks} human picks):")
    print(traits.split_half(picks, min_picks=args.min_picks))
    print("  r = Pearson across managers · full_r = Spearman-Brown · "
          "p = seeded two-sided permutation test")
    print("  At this sample the SE on r is ~0.45. Read the ORDER, not the "
          "third decimal.")


# --------------------------------------------------------------------------
# the predictive model
# --------------------------------------------------------------------------


def _assemble(args):
    """Boards + linked picks + league rules + replayed examples."""
    from . import dataset, rookies
    print(f"League {args.league_id}: assembling boards and replaying drafts")
    boards, linked, cfgs = dataset.build_all(
        args.league_id, on_undeclared=getattr(args, "ecr_undeclared", "fail")
    )
    cov = rookies.coverage(boards)
    print(f"  rookie status: {cov['rookies'].sum()} rookies on "
          f"{cov['skill'].sum()} skill rows · {cov['unresolved'].sum()} unresolved "
          f"(0 inside the top {rookies.CHECK_DEPTH} of any board)")
    bb = dataset.boards_by_season(boards)
    from . import train as tr
    ex = tr.build_examples(bb, linked, cfgs)
    off = sum(1 for e in ex if e.chosen_pid is None)
    inw = sum(1 for e in ex if e.in_window)
    print(f"  {len(ex)} examples · {inw} in the candidate window · {off} off-board")
    return boards, linked, cfgs, bb, ex


def _flow(linked, cfg):
    """Measured positional flow for this league's draft shape (depletion.py)."""
    from . import depletion
    table, seasons = depletion.flow_table(linked, cfg.teams, cfg.rounds)
    print(f"  positional flow from {len(seasons)} seasons "
          f"({min(seasons)}-{max(seasons)}) at {cfg.teams}x{cfg.rounds}")
    return table


def cmd_model(args) -> None:
    from . import autopick, export
    from . import train as tr
    from .config import MODELS
    boards, linked, cfgs, bb, ex = _assemble(args)
    f = tr.train(ex)
    chain = autopick.fit_chain(linked)
    print(f"\nlambda (inclusive value) = {f.lam:.3f}"
          "   [1 = joint conditional logit, 0 = independent stages]")
    # Slice by the layout, not by a hardcoded count: adding a feature used to
    # silently push a manager coefficient into the structural block's printout.
    pl_, po = f.plr_layout, f.pos_layout
    print("\nplayer stage:")
    for n, v in zip(f.plr_spec.names[: pl_.n_feat], f.gamma[: pl_.n_feat]):
        print(f"  {n:24s} {v:+.3f}")
    print("\nposition stage:")
    for n, v in zip(f.pos_spec.names[: po.lam + 1], f.theta_human[: po.lam + 1]):
        print(f"  {n:24s} {v:+.3f}")

    channels = [
        ("pos (asc[p|m])", f.pos_spec.names, f.theta_human,
         lambda n: n.startswith("asc[") and "|" in n and not n.endswith("|early]")),
        ("timing (asc[p|m|early])", f.pos_spec.names, f.theta_human,
         lambda n: n.endswith("|early]")),
    ] + [
        (f"{name} ({feature})", f.plr_spec.names, f.gamma,
         (lambda pre: (lambda n: n.startswith(pre)))(f"{name}["))
        for name, feature in tr.PLAYER_CHANNELS
    ]
    print("\nlargest manager deviations by channel "
          "(shrunk toward the pooled model; `evaluate --ablate` prices them):")
    for label, names, vec, keep in channels:
        dev = sorted(((n, v) for n, v in zip(names, vec) if keep(n)),
                     key=lambda t: -abs(t[1]))
        if not dev:
            continue
        top = "  ".join(f"{n}={v:+.3f}" for n, v in dev[:3])
        print(f"  {label:24s} max|b|={abs(dev[0][1]):.3f}   {top}")
    print("\nautodraft chain:")
    print(autopick.summary(chain, linked).head(12))
    season = max(cfgs)
    MODELS.mkdir(parents=True, exist_ok=True)
    out = export.write(
        MODELS / f"model_{args.league_id}.json",
        boards.filter(pl.col("season") == season), f, chain, cfgs[season],
        meta={"season": season, "league_id": args.league_id},
        flow=_flow(linked, cfgs[season]),
    )
    print(f"\nwrote {out}")


def cmd_evaluate(args) -> None:
    from . import autopick, evaluate
    from . import train as tr
    boards, linked, cfgs, bb, ex = _assemble(args)
    print("\nwalk-forward (fit on seasons < s, score season s):")
    df = evaluate.walk_forward(
        ex, linked, fit_chain=autopick.fit_chain,
        blocks=tr.NO_BLOCKS if args.no_manager else tr.DEFAULT_BLOCKS,
    )
    print("\nheadline:")
    print(evaluate.baseline_table(df))
    print("\nby round bucket:")
    print(evaluate.by_bucket(df))
    if args.b4:
        print("\nB4 — the same model with every manager deviation zeroed:")
        b4 = evaluate.walk_forward(ex, linked, fit_chain=autopick.fit_chain,
                                   blocks=tr.NO_BLOCKS, verbose=False)
        import numpy as np
        r, r4 = df["rank"].to_numpy(), b4["rank"].to_numpy()
        print(f"  manager on : top1={np.mean(r==0):.4f} top3={np.mean(r<3):.4f}")
        print(f"  manager off: top1={np.mean(r4==0):.4f} top3={np.mean(r4<3):.4f}")
        print(f"  delta       top1={np.mean(r==0)-np.mean(r4==0):+.4f}"
              f" top3={np.mean(r<3)-np.mean(r4<3):+.4f}")
    if args.ablate:
        print("\nPer-channel leave-one-out (each row is a full walk-forward refit;"
              "\nd_nll > 0 means dropping the block HURT, i.e. it was doing work):")
        with pl.Config(tbl_rows=len(tr.MANAGER_BLOCKS) + 4):
            print(evaluate.ablate(ex, linked, fit_chain=autopick.fit_chain))
    from .config import ROOT
    out = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    df.write_csv(out / f"eval_{args.league_id}.csv")
    print(f"\nwrote {out / f'eval_{args.league_id}.csv'}")


def cmd_site(args) -> None:
    from . import autopick, export, site, train as tr
    from .config import DOCS, WEB
    boards, linked, cfgs, bb, ex = _assemble(args)
    season = args.season or max(boards["season"].unique().to_list())
    f = tr.train(ex)
    chain = autopick.fit_chain(linked)
    cfg = cfgs.get(season) or cfgs[max(cfgs)]

    # Seat order: ESPN publishes it before the draft (full pick list, teamId
    # set, playerId -1), so use it when it exists rather than making the user
    # retype it. Falls back to last season's order; the page can change either.
    from . import league as league_mod
    from .config import ESPN_CACHE
    managers = None
    espn_order = league_mod.draft_order(ESPN_CACHE / str(args.league_id), season)
    if espn_order:
        owner_to_franchise = {}
        import json as _json
        for s in sorted(cfgs):
            path = ESPN_CACHE / str(args.league_id) / f"league_{s}.json"
            if not path.exists():
                continue
            pay = league_mod._unwrap(_json.loads(path.read_text()))
            fr = {r["team_id"]: r["franchise_id"] for r in
                  linked.filter(pl.col("season") == s)
                  .unique(subset=["team_id"]).iter_rows(named=True)}
            for t in pay.get("teams") or []:
                o = t.get("primaryOwner") or ((t.get("owners") or [None])[0])
                if o and int(t["id"]) in fr:
                    owner_to_franchise[o] = fr[int(t["id"])]
        mapped = [owner_to_franchise.get(o, "") for o in espn_order]
        if sum(1 for m in mapped if m) >= len(mapped) - 2:
            managers = mapped
            print(f"  draft order for {season} read from ESPN "
                  f"({sum(1 for m in mapped if m)}/{len(mapped)} seats matched)")
    if managers is None:
        prev = max(s for s in cfgs if s < season) if any(s < season for s in cfgs) else max(cfgs)
        last = linked.filter(pl.col("season") == prev).sort("overall_pick")
        seats = sorted(last["team_id"].unique().to_list())
        seat_of = {t: i for i, t in enumerate(seats)}
        managers = [""] * len(seats)
        for r in last.unique(subset=["team_id"]).iter_rows(named=True):
            managers[seat_of[r["team_id"]]] = r["franchise_id"]
        print(f"  {season} draft order not set on ESPN yet — using {prev}'s order; "
              "change seats on the page")
    labels = {r["franchise_id"]: r["franchise_label"]
              for r in linked.unique(subset=["franchise_id"]).iter_rows(named=True)}
    counts = {r["franchise_id"]: int(r["n"]) for r in
              linked.filter(~pl.col("auto_pick")).group_by("franchise_id")
              .agg(n=pl.len()).iter_rows(named=True)}

    board_df = boards.filter(pl.col("season") == season)
    payload = {
        "meta": {
            "season": int(season), "league_id": args.league_id,
            "managers": managers, "labels": labels, "pickCounts": counts,
            "note": (f"Model fitted on {linked['season'].min()}-{linked['season'].max()}, "
                     f"{len(ex)} picks. Survival probabilities are recalibrated and "
                     f"carry a few points of error; treat them as guidance, not fact."),
        },
        "model": export.model_payload(f, chain, cfg, _flow(linked, cfg)),
        "board": export.board_payload(board_df),
    }
    out = site.render(payload, WEB, DOCS / "index.html")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {board_df.height} players)")


def cmd_fixtures(args) -> None:
    from . import autopick, export, train as tr
    from .config import ROOT
    boards, linked, cfgs, bb, ex = _assemble(args)
    f = tr.train(ex)
    chain = autopick.fit_chain(linked)
    season = 2025 if 2025 in cfgs else max(cfgs)
    picks = linked.filter(pl.col("season") == season).sort("overall_pick")
    seq = [None if p is None else int(p) for p in picks["pid"].to_list()]
    seats = sorted(picks["team_id"].unique().to_list())
    seat_of = {t: i for i, t in enumerate(seats)}
    managers = [""] * len(seats)
    for r in picks.unique(subset=["team_id"]).iter_rows(named=True):
        managers[seat_of[r["team_id"]]] = r["franchise_id"]
    # 50/60 straddle the EARLY_ROUNDS=6 boundary in a 10-team draft (pick 60
    # is the last of round 6, pick 61 the first of round 7), so an off-by-one
    # on `<=` in early_qb/early_te shows up as a parity diff.
    states = [seq[:n] for n in (0, 1, 11, 37, 50, 60, 75, 120, 140, 148)]
    path = export.fixtures(bb[season], cfgs[season], f, chain, states, managers,
                           ROOT / "tests" / "fixtures" / "parity.json")
    export.write(ROOT / "tests" / "fixtures" / "payload.json",
                 boards.filter(pl.col("season") == season), f, chain, cfgs[season],
                 meta={"season": season, "managers": managers, "league_id": args.league_id},
                 flow=_flow(linked, cfgs[season]))
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tendies", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    ap.add_argument("--first-season", type=int, default=DEFAULT_FIRST_SEASON)
    ap.add_argument("--last-season", type=int, default=DEFAULT_LAST_SEASON)
    ap.add_argument("--refresh", action="store_true", help="re-fetch cached raw JSON")
    ap.add_argument("--no-browser", action="store_true",
                    help="never open Chrome; fail if stored cookies do not work")
    ap.add_argument("--adp", type=Path, default=DEFAULT_ADP_PARQUET,
                    help="ADP board parquet (season, name, position, adp, adp_rank); "
                         "default is the half-PPR board")
    ap.add_argument("--primary-tag", default="half",
                    help="name the primary board reports under in adp_source")
    ap.add_argument("--adp-fallback", action="append", default=None, metavar="PATH[=TAG]",
                    help="extra board, in its own *_<tag> columns and next in the "
                         "adp_rank_filled chain; repeatable, order is the fill order")
    ap.add_argument("--no-fallback", action="store_true", help="primary board only")
    ap.add_argument("--miss-through", type=int, default=120,
                    help="report unmatched picks inside this many overall picks")
    ap.add_argument("--ecr-undeclared", choices=("fail", "ppr", "use"), default="fail",
                    help="a season missing from config.ECR_SOURCE_BY_SEASON: stop "
                         "(default), fall back to full-PPR with a warning, or take "
                         "the half board anyway and mark the run TAINTED")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("login", help="open Chrome and store ESPN cookies")
    p.add_argument("--relogin", action="store_true")
    p.set_defaults(func=cmd_login)
    sub.add_parser("fetch", help="cache raw league JSON").set_defaults(func=cmd_fetch)
    sub.add_parser("build", help="build the pick-level dataset").set_defaults(func=cmd_build)
    sub.add_parser("report", help="print tendency cuts from the built dataset").set_defaults(func=cmd_report)
    sub.add_parser("streaming", help="derive the replacement level behind REPL_RANKS"
                   ).set_defaults(func=cmd_streaming)
    p = sub.add_parser("traits", help="per-manager tendencies + split-half reliability")
    p.add_argument("--min-picks", type=int, default=None,
                   help="a manager enters the reliability table when BOTH "
                        "halves of his career carry this many human picks "
                        "(default: traits.MIN_PICKS)")
    p.set_defaults(func=cmd_traits)
    p = sub.add_parser("ecr-check", help="audit the half-PPR ECR series before pinning it")
    p.add_argument("--seasons", default=None, metavar="A-B",
                   help="inclusive range, e.g. 2013-2026 (default: every season present)")
    p.set_defaults(func=cmd_ecr_check)
    sub.add_parser("model", help="fit the pick model and write its coefficients").set_defaults(func=cmd_model)
    p = sub.add_parser("evaluate", help="walk-forward evaluation against baselines")
    p.add_argument("--b4", action="store_true",
                   help="also fit with manager deviations off, to price them")
    p.add_argument("--no-manager", action="store_true", help="pooled model only")
    p.add_argument("--ablate", action="store_true",
                   help="leave-one-out over each per-manager channel")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("site", help="render the draft-day page into docs/")
    p.add_argument("--season", type=int, default=None)
    p.set_defaults(func=cmd_site)
    sub.add_parser("fixtures", help="regenerate the Python/JS parity fixtures").set_defaults(func=cmd_fixtures)
    args = ap.parse_args(argv)
    if getattr(args, "min_picks", "absent") is None:
        from .traits import MIN_PICKS
        args.min_picks = MIN_PICKS
    if getattr(args, "adp_fallback", None) is None:
        args.adp_fallback = [f"{p}={t}" for t, p in DEFAULT_FALLBACK_BOARDS]
    args.func(args)
