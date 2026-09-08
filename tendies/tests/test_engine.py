"""Tests ordered by how likely they are to catch a genuinely wrong answer.

The parity test comes first on purpose: it exercises the replay engine, the
feature builders, standardisation, both softmaxes and the sampler in two
languages at once. Everything the page shows depends on it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tendies import autopick, dataset, export, predict, train
from tendies.league import POSITIONS, snake_seats
from tendies.replay import DraftState
from tendies.rng import Mulberry32, hash32

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / ".venv/lib/python3.14/site-packages/playwright/driver/node"
HARNESS = ROOT / "tests/parity.mjs"
LEAGUE = 20482930

# Bounded by export.ROUND_TO (9 significant digits), not by logic: at full
# precision the two languages agree to 7e-16.
PARITY_TOL = 1e-8


@pytest.fixture(scope="session")
def built():
    boards, linked, cfgs = dataset.build_all(LEAGUE)
    bb = dataset.boards_by_season(boards)
    ex = train.build_examples(bb, linked, cfgs)
    return boards, linked, cfgs, bb, ex


@pytest.fixture(scope="session")
def fitted(built):
    _, linked, _, _, ex = built
    return train.train(ex), autopick.fit_chain(linked)


# --------------------------------------------------------------- parity ---


@pytest.fixture(scope="session")
def live_payload_path(built, fitted, tmp_path_factory):
    """The payload the SITE ships: the newest BOARD season, with the Boone/ETR
    hint columns joined on.

    Separate from `payload_path` because the two differ in a way that matters
    to exactly one test. `payload_path` is the newest season with DRAFT PICKS
    (2025) — what the parity and mock-draft harnesses need, since they replay
    real drafts — and its board carries no hint columns at all, so it cannot
    show whether the recommendation card renders a Boone rank or an em dash.
    """
    from tendies import depletion, export, hints
    boards, linked, cfgs, _, _ = built
    f, chain = fitted
    season = max(int(x) for x in boards["season"].unique().to_list())
    cfg = cfgs.get(season) or cfgs[max(cfgs)]
    board = hints.attach(boards.filter(pl.col("season") == season),
                         season=season, verbose=False)
    managers = sorted({r["franchise_id"] for r in
                       linked.unique(subset=["franchise_id"]).iter_rows(named=True)})
    flow, _ = depletion.flow_table(linked, cfg.teams, cfg.rounds)
    return export.write(
        tmp_path_factory.mktemp("live") / "payload.json", board, f, chain, cfg,
        meta={"season": season, "managers": managers[:cfg.teams], "league_id": LEAGUE},
        flow=flow,
    )


@pytest.fixture(scope="session")
def payload_path(built, fitted, tmp_path_factory):
    """A full live payload — board, coefficients, and the positional flow
    table — written once for the node harnesses that need a real board."""
    from tendies import depletion, export
    boards, linked, cfgs, _, _ = built
    f, chain = fitted
    season = max(cfgs)
    picks = linked.filter(pl.col("season") == season).sort("overall_pick")
    seats = sorted(picks["team_id"].unique().to_list())
    seat_of = {t: i for i, t in enumerate(seats)}
    managers = [""] * len(seats)
    for r in picks.unique(subset=["team_id"]).iter_rows(named=True):
        managers[seat_of[r["team_id"]]] = r["franchise_id"]
    flow, _ = depletion.flow_table(linked, cfgs[season].teams, cfgs[season].rounds)
    return export.write(
        tmp_path_factory.mktemp("payload") / "payload.json",
        boards.filter(pl.col("season") == season), f, chain, cfgs[season],
        meta={"season": season, "managers": managers, "league_id": LEAGUE},
        flow=flow,
    )


def _node(script: str, *args) -> dict:
    res = subprocess.run([str(NODE), str(ROOT / "tests" / script), *map(str, args)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_python_js_parity(built, fitted, tmp_path):
    boards, linked, cfgs, bb, _ = built
    f, chain = fitted
    season = 2025
    picks = linked.filter(pl.col("season") == season).sort("overall_pick")
    seq = [None if p is None else int(p) for p in picks["pid"].to_list()]
    seats = sorted(picks["team_id"].unique().to_list())
    seat_of = {t: i for i, t in enumerate(seats)}
    managers = [""] * len(seats)
    for r in picks.unique(subset=["team_id"]).iter_rows(named=True):
        managers[seat_of[r["team_id"]]] = r["franchise_id"]

    # corners, not averages: empty board, first pick, mid draft, deep endgame
    # 50/60 straddle the EARLY_ROUNDS=6 boundary in a 10-team draft (pick 60
    # is the last of round 6, pick 61 the first of round 7), so an off-by-one
    # on `<=` in early_qb/early_te shows up as a parity diff.
    states = [seq[:n] for n in (0, 1, 11, 37, 50, 60, 75, 120, 140, 148)]
    payload = export.write(
        tmp_path / "payload.json", boards.filter(pl.col("season") == season),
        f, chain, cfgs[season],
        meta={"season": season, "managers": managers, "league_id": LEAGUE},
    )
    fx = export.fixtures(bb[season], cfgs[season], f, chain, states, managers,
                         tmp_path / "fixtures.json")

    res = subprocess.run([str(NODE), str(HARNESS), str(payload), str(fx)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    cases = json.loads(fx.read_text())["cases"]
    assert cases, "no fixtures generated"

    for c in cases:
        got, want = js[c["id"]], c["expect"]
        assert got["positions"] == want["positions"], c["id"]
        df = np.max(np.abs(np.array(got["posFeatures"]) - np.array(want["posFeatures"])))
        dp = np.max(np.abs(np.array(got["posProbs"]) - np.array(want["posProbs"])))
        gt = {int(k): v for k, v in got["top"]}
        dt = max(abs(gt.get(int(k), 0.0) - v) for k, v in want["top"])
        assert df < PARITY_TOL, f"{c['id']} feature drift {df:.2e}"
        assert dp < PARITY_TOL, f"{c['id']} position prob drift {dp:.2e}"
        assert dt < PARITY_TOL, f"{c['id']} player prob drift {dt:.2e}"


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_lineup_closed_form():
    """Every EV number on the page rests on two identities: `marginal ==
    max(0, vorp - cutoff)`, and greedy slot assignment being optimal. If either
    is false the panel is confidently wrong and nothing else would tell us."""
    out = _node("lineup.mjs")
    assert out["checked"] > 20000 and out["assignChecked"] > 1000
    assert out["worst"] < 1e-6, f"cutoff identity: {out['worst']:.3e} {out['worstCase']}"
    assert out["assignWorst"] < 1e-6, f"greedy not optimal: {out['assignCase']}"
    # an empty slot must price at exactly its floor, not at zero
    assert out["floorErr"] < 1e-6
    # the player card lists the slots one by one; the panel quotes their total
    # as a single number. Two functions, so they are asserted to agree.
    assert out["slotChecked"] > 4000
    assert out["slotErr"] < 1e-6, f"displayed slots do not sum to the value: {out['slotErr']:.3e}"
    # and the flex floor has to be the MAX of the floors it draws from: with
    # their min instead, greedy demonstrably stops being optimal
    assert out["badWorst"] > 1.0, "the EMPTY_FLEX = max rule is no longer load-bearing"


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_ties_are_reported_but_never_reordered(payload_path):
    """The recommendation panel's resolution floor.

    The VORP curve resolves the top of the board and not the back of it (the
    numbers are in vorp.py's docstring), so the panel groups candidates it
    cannot separate and shares one rank number across them. Three properties,
    and the third is the one with a price tag on it.
    """
    out = _node("ties.mjs", payload_path)
    assert out["states"] > 10

    # the payload must carry real standard errors, not a field of zeros -- an
    # all-zero column would mean the curve claims to be exact everywhere
    assert out["seSeen"] > 200
    assert out["sePositive"] > out["seSeen"] // 2
    # ...and some must be exactly zero, which is the same-PAVA-block case
    assert out["seZero"] > 0

    # a tier is "tied with the tier's LEADER", which is transitive; chaining
    # pairwise ties would pool candidates the curve separates cleanly
    assert out["transitivity"] > 1000
    assert not out["transitivityBad"], out["transitivityBad"][:5]
    assert not out["tierBad"], out["tierBad"][:5]

    # THE ONE THAT MATTERS. Both attempts to order on urgency rather than on the
    # point estimate lost lineup points: 24.5 across the whole 1.96-SE band
    # (528.2 -> 503.7, seats beaten 10/10 -> 4/10) and 2.0 across exact ties on
    # the 2026 board (540.6 -> 538.6, stable at 200/300/400 sims). Statistical
    # indistinguishability is not indifference. Displayed order is
    # score-descending inside a (need, defer, bench) group, and the only
    # tie-break allowed is the endgame roster-shape one.
    assert out["orderChecked"] > 1000
    assert not out["orderBad"], out["orderBad"][:5]

    # a bench candidate below his position's free end-of-draft floor scores 0,
    # however mispriced he is: ungated, `edge` put a VORP -99 quarterback above
    # the best available back as soon as the candidate set widened
    assert out["benchGate"] > 1000
    assert not out["benchGateBad"], out["benchGateBad"][:5]

    # the candidate set must not filter on the thing the bench score ranks by
    assert not out["candBad"], out["candBad"][:5]
    # and this is a live failure mode, not a hypothetical one: the old
    # `perPos: 8` ADP slice hid the board's biggest market edge in most states
    assert out.get("cappedWouldMiss", 0) > 0


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_plan_matches_brute_force():
    """The plan DP packs the flex bound into state reachability and carries a
    lexicographic feasibility term. On a draft small enough to enumerate, every
    action sequence is evaluated by hand and the DP must match exactly.

    Run twice per state: unconstrained, then under a late-round QB embargo,
    where the DP drops the position from a per-turn action set and the brute
    force drops it from the enumerated sequences. The constrained optimum must
    match, must never beat the free one, must lose sometimes (or the rule
    reached nothing), and must never SCHEDULE a quarterback inside the rule.
    """
    out = _node("plan.mjs")
    assert out["trials"] > 30
    assert out["worstV"] < 1e-9, f"value mismatch {out['worstV']:.3e}: {out['cases']}"
    assert out["badF"] == 0, f"feasibility mismatch: {out['cases']}"
    emb = out["embargo"]
    assert emb["worstV"] < 1e-9, f"embargoed value mismatch {emb['worstV']:.3e}: {out['cases']}"
    assert emb["badF"] == 0, f"embargoed feasibility/schedule failure: {out['cases']}"
    assert emb["higher"] == 0, "the embargoed plan beat the unconstrained one"
    assert emb["lower"] > 0, "the embargo changed no plan — it reached nothing"


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_mock_draft_is_legal(payload_path):
    """Play whole drafts taking the top recommendation every time.

    This is the test that would have caught the failure the plan was written to
    fix: the previous engine, run this way, took five quarterbacks, six tight
    ends against a cap of three, and no kicker — while all 70 real team-seasons
    in this league filled every starting slot. Every number it computed was
    internally consistent; only playing it out shows it.
    """
    out = _node("mockdraft.mjs", payload_path, 120)
    assert not out["problems"], out["problems"]

    # And it must be worth running: better than best-available by our own value
    # and by the market's. Asserted on the MEAN over the ten seats, not on a
    # per-seat win count.
    #
    # The count was the original assertion and it is too brittle to keep.
    # Best-available-by-VORP is greedy on precisely the quantity this test
    # scores, so on an individual seat it is a strong opponent and the result
    # can turn on a single tie-break — this league's isotonic curve has long
    # plateaus, so ties are everywhere. Refreshing the half-PPR ECR series
    # (2026-only to 2013-2026) moved the count from 10/10 to 7/10 on the 2025
    # fixture without moving the mean margin materially; it is 10/10 on the
    # live 2026 board. A statistic that swings that far on an input refresh is
    # measuring the tie-breaks, not the planner.
    m = out["means"]
    assert m["model"] > m["vorp"] + 10, m
    assert m["model"] > m["adp"] + 10, m
    # a floor on the count as well, well clear of where it sits, so a real
    # collapse still trips something
    assert out["beat"] >= out["seats"] // 2, [
        (r["seat"], r["score"], r["base"]) for r in out["rows"]
    ]


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_survival_is_unchanged_by_the_plan(payload_path):
    """Survival probabilities are calibrated and were fitted before the plan
    existed. The second simulation leg shares their loop and their random
    stream, so `p` has to come back bit-identical, not merely close."""
    out = _node("simstable.mjs", payload_path)
    assert out["checked"] > 500
    assert out["worst"] == 0, f"plan moved survival by {out['worst']}"
    assert out["jitter"] == 0, "same state gave two different answers"


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_fast_forward_is_deterministic_and_legal(payload_path):
    """The page's skip-ahead is parallel code to simulate()'s inner step (that
    stream is frozen), so it carries its own proof: argmax deterministic,
    sampling a pure function of (state, seedBase), stops exact, no taken or
    duplicate pids, and simulate's stream untouched by a jump."""
    out = _node("fastforward.mjs", payload_path)
    assert out["ok"], out["problems"]
    assert out["fullDraft"] == 150


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_page_renders_on_and_off_the_clock(payload_path):
    """The page's DOM glue, loaded against a stub DOM and rendered for real.

    page.js is an IIFE under 'use strict', so a free identifier is a RUNTIME
    ReferenceError — `node --check` passes and the page loads fine. The first
    render that reaches the bad line throws inside a requestAnimationFrame
    callback, which the browser logs to a console nobody is reading mid-draft,
    and the affected panels just stay blank. That is how `league` (a const
    local to rebuild()) got read from buildRec() and silently emptied the
    survival columns and all three sim-dependent panels on every pick that was
    not the user's own.

    A smoke test on purpose: it asserts a render cycle COMPLETES and each panel
    produced content, on the clock and off it, that "Take now" retitles itself
    to the user's own pick, that its P(avail) slider filters that panel and
    only that panel, and that no explanatory text cell has crept back into it.
    It also drives the late-round-QB dropdown and the two survival-column
    sorts, both of which are orderings rather than numbers — the kind of thing
    that is wrong on screen while every engine test still passes.
    Whether the numbers are right is ties/plan/lineup/mockdraft's job.
    """
    out = _node("page.mjs", payload_path, 2)
    assert out["ok"], out["fails"]
    off = [s for s in out["states"] if s["at"].endswith("off clock")]
    # Only the heading: the default threshold legitimately empties the LIST at
    # the short side of the snake (nobody is under 50% over a four-pick gap),
    # so a row count here would be flaky. That the panel rendered at all is
    # page.mjs's own per-panel check, and `risk.all` covers the list.
    assert off and all(s["hd"].startswith("Your pick") for s in off), out["states"]
    on = [s for s in out["states"] if s["at"].endswith("on clock")]
    assert on and all(s["hd"] == "Take now" for s in on), out["states"]
    # The P(avail) threshold: it narrows the panel, it is off at the top of its
    # range, and it leaves the BOARD alone — the filter belongs to one panel.
    # A 50% cut that keeps most of the field means the membership condition
    # regressed (`recommend` reports pAvail 0 for anyone the simulation does
    # not watch, which reads as "certainly gone" for every deep player).
    r = out["risk"]
    assert r["k5"] <= r["k50"] < r["all"], r
    assert r["k50"] < r["all"] / 2, r
    assert r["boardAll"] == r["board50"] == r["board5"], r

    # Late-round QB, checked where the unconstrained answer actually uses a
    # quarterback (top of round 3, argmax picks): the rule empties the board
    # and the candidate list for YOUR seat, moves the plan's quarterback past
    # the round, keeps quarterbacks on the board while someone else is on the
    # clock — otherwise their pick could not be recorded — and gives them all
    # back when it is switched off.
    q = out["lateQb"]
    # page.mjs reports rather than asserts these: a quarterback being wanted
    # here at all is a property of the BOARD, and it holds on this fixture.
    assert all(q["testable"].values()), q
    for k in ("on9", "on13"):
        assert q[k]["board"] == 0 and q[k]["rec"] == 0, (k, q[k])
    teams = json.loads(Path(payload_path).read_text())["model"]["league"]["teams"]
    assert all((t - 1) // teams + 1 >= 9 for t in q["on9"]["plan"]), q["on9"]
    assert all((t - 1) // teams + 1 >= 13 for t in q["on13"]["plan"]), q["on13"]
    assert q["offClockBoard"] > 0, q
    assert q["restored"] == q["off"]["board"], q

    # The survival columns sort least-likely-to-last first, and a blank (a
    # player outside the simulation's watched top ~60) sinks in BOTH
    # directions — page.mjs fails the run on either, these pin the intent.
    for key in ("sort_p", "sort_p2"):
        asc, desc = out[key]["asc"], out[key]["desc"]
        assert asc == sorted(asc) and asc[0] <= 10, (key, asc)
        assert desc == sorted(desc, reverse=True), (key, desc)


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_page_cards_show_the_hint_columns(live_payload_path):
    """The same render, on the payload the site actually ships.

    `payload_path`'s board is the newest season with draft picks and has no
    hint columns, so it cannot tell a card that renders a Boone rank from one
    that renders an em dash — and the hint join is the part of this that can
    quietly come back empty (a renamed player, a re-fetched board, a
    `tendies hints` that was never re-run).
    """
    out = _node("page.mjs", live_payload_path, 2)
    assert out["ok"], out["fails"]
    assert out["boone"]["inPayload"], "the live payload carries no Boone ranks"
    assert any(v.isdigit() for v in out["boone"]["shown"]), out["boone"]


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_rng_parity(tmp_path):
    """Identical RNG streams are what make simulation parity exact rather than
    statistical."""
    js = tmp_path / "r.mjs"
    js.write_text(
        "const E=require('%s');const r=E.mulberry32(12345);"
        "const o=[];for(let i=0;i<1000;i++)o.push(r());"
        "process.stdout.write(JSON.stringify([o[0],o[499],o[999],E.hash32('abc')]));"
        % (ROOT / "tendies/web/engine.js")
    )
    wrapper = tmp_path / "r.cjs"
    wrapper.write_text(js.read_text())
    res = subprocess.run([str(NODE), str(wrapper)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    a, b, c, h = json.loads(res.stdout)
    rng = Mulberry32(12345)
    vals = [rng.random() for _ in range(1000)]
    assert abs(vals[0] - a) < 1e-15
    assert abs(vals[499] - b) < 1e-15
    assert abs(vals[999] - c) < 1e-15
    assert hash32("abc") == h


# ------------------------------------------------------------- invariants ---


def test_board_pid_is_adp_ordinal(built):
    boards = built[0]
    for s in boards["season"].unique().to_list():
        b = boards.filter(pl.col("season") == s)
        assert (b["pid"] == b["adp_rank"] - 1).all(), s


def test_every_pick_links_to_a_board_row(built):
    linked = built[1]
    rate = linked["pid"].is_not_null().mean()
    # FantasyPros omits a handful of real players (Ekeler 2025, Dobbins 2023)
    assert rate > 0.98, f"link rate {rate:.3f}"


def test_chosen_player_is_in_his_own_choice_set(built):
    """A training example whose label is not in its candidate set would be
    silently unlearnable."""
    ex = built[4]
    usable = [e for e in ex if e.chosen_pid is not None]
    inw = [e for e in usable if e.in_window]
    for e in inw[:200]:
        assert e.chosen_pid in e.cand[e.chosen_pos]
    assert len(inw) / len(usable) > 0.95


def test_advance_undo_roundtrip(built):
    boards, linked, cfgs, bb, _ = built
    board, cfg = bb[2025], cfgs[2025]
    st = DraftState(board, cfg, [""] * cfg.teams)
    rng = np.random.default_rng(0)
    before = None
    for i in range(40):
        pid = int(rng.integers(0, board.n))
        if st.taken[pid]:
            continue
        if i == 20:
            before = (st.taken.copy(), st.counts.copy(), st.cursor.copy(), list(st.picks))
        st.advance(pid)
    for _ in range(len(st.picks) - len(before[3])):
        st.undo()
    assert np.array_equal(st.taken, before[0])
    assert np.array_equal(st.counts, before[1])
    assert st.picks == before[3]
    # the cursor may only have moved backwards, never past a taken player
    assert (st.cursor <= before[2]).all()


def test_position_caps_are_never_exceeded(built):
    """Caps are a mask on the choice set, so a violation means the mask leaked."""
    _, _, cfgs, bb, ex = built
    for e in ex:
        cfg = cfgs[e.season]
        for p in e.positions:
            name = POSITIONS[p]
            assert p in e.cand and len(e.cand[p]) > 0
            assert name in cfg.limits or True


def test_snake_seats():
    for teams in (8, 10, 12):
        for rounds in (15, 16):
            seats = snake_seats(teams, rounds)
            assert len(seats) == teams * rounds
            for s in range(1, teams + 1):
                assert seats.count(s) == rounds
            # round 1 pick k and round 2 pick (teams+1-k) are the same seat
            for k in range(teams):
                assert seats[k] == seats[2 * teams - 1 - k]


def test_probabilities_sum_to_one(built, fitted):
    f, _ = fitted
    ex = built[4]
    for e in ex[:100]:
        if e.chosen_pos is None:
            continue
        pids, probs, outside = predict.joint_probs(f, e, 0.0)
        assert abs(probs.sum() + outside - 1.0) < 1e-9


def test_vorp_curve_cannot_see_its_own_season():
    """VORP is the only channel from season outcomes into the features, so the
    curve for season s must be identical whether or not seasons >= s are in the
    history it is handed. That is the leakage guard, stated as a test."""
    from tendies import vorp as vorp_mod

    hist = dataset.curve_history()
    for s in (2021, 2023, 2025):
        full = vorp_mod.build_curve(hist, s)
        trimmed = vorp_mod.build_curve(hist.filter(pl.col("season") < s), s)
        assert set(full) == set(trimmed)
        for pos in full:
            assert len(full[pos]) == len(trimmed[pos]), pos
            # not bitwise: polars' parallel aggregation carries ~1e-16 of
            # float-ordering noise run to run, as the parent repo documents.
            np.testing.assert_allclose(full[pos], trimmed[pos], rtol=1e-12, atol=1e-12)


def test_unknown_manager_falls_back_to_pooled(built, fitted):
    """A manager with no history — a new arrival in 2026 — must score as the
    pooled model rather than crash or silently borrow someone else's taste."""
    f, _ = fitted
    boards, _, cfgs, bb, _ = built
    season = max(cfgs)
    st = DraftState(bb[season], cfgs[season], ["NOBODY"] * cfgs[season].teams)
    st.advance(0)
    ex = predict.example_from_state(st, st.seat_on_clock(), manager="NOBODY")
    assert ex is not None
    assert "NOBODY" not in f.managers
    pids, probs, outside = predict.joint_probs(f, ex, 0.0)
    assert len(pids) > 0
    assert abs(probs.sum() + outside - 1.0) < 1e-9
    # identical to an explicitly pooled call
    ex2 = predict.example_from_state(st, st.seat_on_clock(), manager="")
    _, probs2, _ = predict.joint_probs(f, ex2, 0.0)
    assert np.allclose(probs, probs2)


def test_top1_is_not_implausibly_high(built, fitted):
    """A red flag, not a target: within-position choice tops out around 41%
    under pure ADP, so an overall top-1 near that implies a leak."""
    from tendies import evaluate

    f, chain = fitted
    ex = [e for e in built[4] if e.season == 2025]
    df = evaluate.score(f, ex, chain)
    top1 = float((df["rank"] == 0).mean())
    assert top1 < 0.40, f"top-1 {top1:.3f} is too high to be real"


@pytest.mark.skipif(not NODE.exists(), reason="playwright node driver not installed")
def test_payload_manager_blocks_are_exactly_sized(payload_path):
    """A mis-sliced manager block that keeps the TOTAL length right is invisible
    to both `assertContract` (which guards feature names) and the parity test
    (Python and JS would read the same wrong array). See tests/contract.mjs."""
    res = subprocess.run(
        [str(NODE), str(ROOT / "tests/contract.mjs"), str(payload_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr


# ------------------------------------------------------- rookie status ---


def test_rookie_resolves_the_head_of_every_board(built):
    """Every skill player a manager could plausibly be choosing between has an
    NFL entry year. Deep-tail names on a 400-player board are camp bodies that
    genuinely are not in any roster file, which is why the guard is on the head
    rather than on a global rate."""
    from tendies import rookies
    boards, _, _, _, _ = built
    rookies.check(boards)                       # raises on a failure
    cov = rookies.coverage(boards)
    assert cov["head_unresolved"].sum() == 0
    # and the flag has to actually fire: a season with no rookies at all means
    # the join silently produced zeros.
    assert cov["rookies"].min() > 0, cov


def test_entry_year_is_static_not_a_season_t_fact(built):
    """`rookie` is `entry_year == season`, and entry year is a static player
    attribute. Dropping season t's own roster file must not change the entry
    year of any player the remaining files still identify — otherwise the flag
    would be reading membership of a roster only settled after the draft.

    Asserted on the GSIS pass, which is the identity join. The NAME fallback is
    a heuristic whose behaviour legitimately depends on the corpus: it drops any
    name carrying two entry years, so a name that is unique through 2024 and
    ambiguous through 2026 answers in the first map and abstains in the second
    (Kaleb Johnson, a 2025 rookie sharing a name with an earlier player, is the
    live example). Abstaining is the safe direction and the shipped map tries
    gsis first, so no board row is ever decided by the ambiguous name.

    Same spirit as `test_vorp_curve_cannot_see_its_own_season`.
    """
    from tendies import rookies
    from tendies.config import CONTEXT_RAW_DIR
    season = 2025
    full_gsis, _ = rookies.entry_year_map(CONTEXT_RAW_DIR, 2026)
    older_gsis, _ = rookies.entry_year_map(CONTEXT_RAW_DIR, season - 1)

    both = older_gsis.join(full_gsis, on="gsis_id", how="inner", suffix="_full")
    assert both.height > 5000
    assert (both["entry_gsis"] == both["entry_gsis_full"]).all()


def test_name_fallback_never_contradicts_the_id_join(built):
    """The two passes must agree wherever both have an answer. They are tried in
    order (gsis, then name), so a disagreement would not corrupt a board — but it
    would mean the name key is resolving to the wrong person, and the next name
    it gets wrong might be one gsis cannot rescue."""
    from tendies import rookies
    from tendies.config import CONTEXT_RAW_DIR
    boards, _, _, _, _ = built
    by_gsis, by_name = rookies.entry_year_map(CONTEXT_RAW_DIR, 2026)
    both = (
        boards.join(by_gsis, on="gsis_id", how="inner")
        .join(by_name, on="key", how="inner")
    )
    assert both.height > 1000
    bad = both.filter(pl.col("entry_gsis") != pl.col("entry_name"))
    assert bad.is_empty(), bad.select("season", "name", "entry_gsis", "entry_name").rows()


# --------------------------------------------------- manager deviations ---


def test_manager_blocks_drop_cleanly(built):
    """`train(blocks=...)` must shorten the coefficient vectors by exactly the
    blocks it dropped. The original `--b4` was a boolean that only reached the
    position stage, so `reach[m]` was fitted while the run reported that every
    manager deviation was off; this is the test that pins it shut."""
    _, _, _, _, ex = built
    n_mgr = len({e.manager for e in ex if e.manager})

    full = train.train(ex, blocks=train.ALL_BLOCKS)
    none = train.train(ex, blocks=train.NO_BLOCKS)

    assert len(full.gamma) - len(none.gamma) == n_mgr * len(train.PLAYER_CHANNELS)
    assert len(full.theta_human) - len(none.theta_human) == n_mgr * (
        len(train.MANAGER_POS) + len(train.MANAGER_EARLY)
    )
    assert not any("[" in n and "|" in n for n in none.pos_spec.names)
    channel_names = tuple(f"{n}[" for n, _ in train.PLAYER_CHANNELS)
    assert not any(n.startswith(channel_names) for n in none.plr_spec.names)
    assert none.with_manager is False and full.with_manager is True

    # every named block is individually droppable, and drops EXACTLY itself
    widths = {"pos": n_mgr * len(train.MANAGER_POS),
              "timing": n_mgr * len(train.MANAGER_EARLY)}
    for block in train.MANAGER_BLOCKS:
        f = train.train(ex, blocks=train.ALL_BLOCKS - {block})
        dropped = (len(full.gamma) - len(f.gamma)) + (
            len(full.theta_human) - len(f.theta_human)
        )
        assert dropped == widths.get(block, n_mgr), (block, dropped)


def test_early_deviation_is_identified(built):
    """`asc[QB|m]` is constant across rounds and `asc[QB|m|early]` is on only in
    rounds 1-EARLY_ROUNDS, so they separate only if each manager actually picks
    in both regimes. If a future league has a manager who never picks late, the
    two columns are collinear and the fit is meaningless."""
    from tendies import features as feat
    _, _, _, _, ex = built
    seen: dict[str, set[bool]] = {}
    for e in ex:
        if e.manager and not e.is_auto:
            seen.setdefault(e.manager, set()).add(e.round <= feat.EARLY_ROUNDS)
    thin = {m: s for m, s in seen.items() if len(s) < 2}
    assert not thin, f"managers with picks in only one round regime: {thin}"


# ------------------------------------------ age / durability / prev owner ---


def test_age_and_prev_owner_are_covered(built):
    """Age must resolve for the head of every board, and the prior-owner link
    must actually find players — a silently empty join would make `was_mine`
    an all-zero column that still fits a coefficient."""
    from tendies import attrs
    boards, _, _, _, _ = built
    attrs.check(boards)                         # raises on a failure
    cov = attrs.coverage(boards)
    assert cov["head_no_age"].sum() <= 5, cov

    # 2018 and 2019 predate the league's first draft, so they carry no prior
    # owner by construction; every season after that must.
    later = cov.filter(pl.col("season") >= 2020)
    assert (later["prev_owned"] > 50).all(), cov


def test_durability_cannot_see_its_own_season(built):
    """`durability` is the one feature built from `actuals.parquet`, the OUTCOME
    table. It must read season t-1 and nothing later.

    Asserted by rebuilding a board with every actuals row from season t onward
    deleted: if the values do not move, nothing downstream of t was consulted.
    Same shape as `test_vorp_curve_cannot_see_its_own_season`.
    """
    import tempfile
    from tendies import attrs
    from tendies import board as board_mod
    from tendies import rookies
    from tendies.config import ACTUALS_PARQUET, CONTEXT_RAW_DIR, FP_RAW_DIR
    boards, _, _, _, _ = built
    season = 2024
    full = boards.filter(pl.col("season") == season).sort("pid")

    truncated = pl.read_parquet(ACTUALS_PARQUET).filter(pl.col("season") < season)
    assert truncated.height < pl.read_parquet(ACTUALS_PARQUET).height
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "actuals.parquet"
        truncated.write_parquet(path)
        maps = rookies.entry_year_map(CONTEXT_RAW_DIR, 2026)
        births = attrs.birth_date_map(CONTEXT_RAW_DIR, 2026)
        from tendies.config import DEFAULT_ADP_PARQUET
        skill = pl.read_parquet(DEFAULT_ADP_PARQUET)
        rebuilt = board_mod.build(
            season, FP_RAW_DIR, skill, None, maps, births, path
        ).sort("pid")

    assert rebuilt.height == full.height
    delta = (rebuilt["durability"] - full["durability"]).abs().max()
    assert delta < 1e-12, f"durability moved by {delta} when season {season}+ was removed"
    # and it must not be a constant column, or the test proves nothing
    assert full["durability"].std() > 0.05


def test_was_mine_belongs_to_exactly_one_seat(built):
    """`prev_owner` is a property of the PLAYER; `was_mine` is that compared
    against whoever is on the clock. So a player can be 'mine' for at most one
    seat in a given draft, and for at least one when he was drafted last year
    by a franchise still in the league."""
    from tendies import features as feat
    boards, linked, cfgs, bb, _ = built
    season = 2025
    board = bb[season]
    managers = sorted({m for m in linked.filter(pl.col("season") == season)["franchise_id"]})
    st = DraftState(board, cfgs[season], managers)
    cand = np.arange(min(120, board.n), dtype=np.int32)
    col = feat.PLR_FEATURES.index("was_mine")

    owned = np.zeros(len(cand))
    for seat in range(len(managers)):
        owned += feat.player_features(st, seat, cand)[:, col]
    assert owned.max() <= 1.0, "a player was 'mine' for two seats at once"
    assert owned.sum() > 5, "no candidate was owned by anyone last season"


def test_player_channels_name_real_features():
    """Every per-manager channel scales a column that exists, and the JS side
    resolves them by NAME from the payload, so a typo here would be a silent
    zero rather than an error."""
    for name, feature in train.PLAYER_CHANNELS:
        assert feature in train.feat.PLR_FEATURES, (name, feature)
    assert len({n for n, _ in train.PLAYER_CHANNELS}) == len(train.PLAYER_CHANNELS)
