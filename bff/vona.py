"""VONA (Value Over Next Available) — a draft-strategy overlay on the 2026 board.

This is NOT part of the scored model. The board (`preds_model_2026.parquet`)
ranks players by season-long predicted VORP; that value list, correctly for
full PPR, leans WR at the very top. VONA answers a different, draft-day
question: at each pick, how much value do you LOSE at each position by waiting
until your next turn? That is where positional scarcity lives — startable RB
evaporates faster than WR at specific picks, so "take the RB before the run"
is right at some picks and wrong at others. VONA makes that timing explicit
without touching the value ranking, the VORP conversion, the metric, or the
test set.

Pure function of two byte-stable artifacts:
- `data/processed/preds_model_2026.parquet`  (gsis_id, score = predicted VORP)
- `data/processed/adp.parquet` season 2026    (gsis_id, name, position, adp_rank)

Availability model: players leave the board in ADP order. At overall pick p,
everyone with adp_rank < p is gone; the best available at a position is the
max predicted VORP among the remaining players there (OUR value order within
position, which can differ from ADP). Because availability at p+h is a subset
of availability at p, VONA is non-negative by construction.

League: 12-team snake, 1QB/2RB/2WR/1TE/1FLEX, PPR. Opponents are assumed to
draft by ADP; this is a positional-timing guide, not a full draft simulator.

    uv run python -m bff.vona          # -> reports/vona_2026.csv, prints turn matrix
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from bff.backtest import POSITIONS

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT_VONA = ROOT / "reports" / "vona_2026.csv"

SEASON = 2026
TEAMS = 12
N_PICKS = 150          # overall picks the per-pick table covers (pool is 185)
MATRIX_ROUNDS = 5      # snake-turn matrix depth
LOOKAHEADS = (12, 24)  # one round / two rounds of a 12-team draft


def load_board() -> pl.DataFrame:
    """(gsis_id, name, position, adp_rank, vorp) for the 2026 pool, inner-joined
    on the same 185 players `to_vorp`'s 2026 branch scores."""
    preds = pl.read_parquet(PROC / "preds_model_2026.parquet").select(
        "gsis_id", pl.col("score").alias("vorp")
    )
    adp = (
        pl.read_parquet(PROC / "adp.parquet")
        .filter((pl.col("season") == SEASON) & pl.col("gsis_id").is_not_null()
                & pl.col("position").is_in(POSITIONS))
        .select("gsis_id", "name", "position", "adp_rank")
    )
    board = adp.join(preds, on="gsis_id", how="inner")
    return board.sort("adp_rank")


def best_available(board: pl.DataFrame) -> dict[str, list[tuple[int, float, str]]]:
    """Per position, the running best-available (max VORP among players with
    adp_rank >= p) as p increases. Returns pos -> list indexed by pick-1 of
    (adp_rank_of_best, vorp_of_best, name_of_best); value 0 / '' when the
    position is exhausted."""
    out: dict[str, list[tuple[int, float, str]]] = {}
    for pos in POSITIONS:
        sub = board.filter(pl.col("position") == pos)
        ranks = sub["adp_rank"].to_numpy()
        vorp = sub["vorp"].to_numpy()
        names = sub["name"].to_list()
        # for each overall pick p (1..N_PICKS), best player with adp_rank >= p
        # walk from the deep end so we can carry the running max cheaply
        order = np.argsort(ranks)          # by adp_rank asc
        ranks, vorp = ranks[order], vorp[order]
        names = [names[i] for i in order]
        best: list[tuple[int, float, str]] = []
        for p in range(1, N_PICKS + 1):
            mask = ranks >= p
            if not mask.any():
                best.append((0, 0.0, ""))
                continue
            cand_r = ranks[mask]; cand_v = vorp[mask]
            cand_n = [names[i] for i in range(len(ranks)) if mask[i]]
            j = int(np.argmax(cand_v))
            best.append((int(cand_r[j]), float(cand_v[j]), cand_n[j]))
        out[pos] = best
    return out


def _at(best_list: list[tuple[int, float, str]], pick: int) -> tuple[float, str]:
    """(vorp, name) of the best available at overall pick `pick` (1-indexed);
    picks past the table clamp to the last entry (value already 0 there)."""
    i = min(max(pick, 1), len(best_list)) - 1
    _, v, n = best_list[i]
    return v, n


def per_pick_table(board: pl.DataFrame) -> pl.DataFrame:
    """One row per overall pick p = 1..N_PICKS: best available player + VORP per
    position, and the wait-cost (VONA) at each generic lookahead. This is a
    diagnostic; the slot-correct recommendation is `turn_matrix`, which uses
    each slot's exact next pick instead of a fixed offset."""
    ba = best_available(board)
    rows = []
    for p in range(1, N_PICKS + 1):
        row: dict[str, object] = {"pick": p}
        for pos in POSITIONS:
            v_now, name_now = _at(ba[pos], p)
            row[f"best_{pos}"] = name_now
            row[f"v_{pos}"] = round(v_now, 2)
            for h in LOOKAHEADS:
                v_later, _ = _at(ba[pos], p + h)
                row[f"vona{h}_{pos}"] = round(max(0.0, v_now - v_later), 2)
        rows.append(row)
    return pl.DataFrame(rows)


def snake_picks(slot: int, rounds: int) -> list[int]:
    """Overall pick numbers for a given draft slot (1..TEAMS) over `rounds`
    rounds of a TEAMS-team snake."""
    picks = []
    for r in range(1, rounds + 1):
        picks.append((r - 1) * TEAMS + slot if r % 2 == 1 else r * TEAMS - slot + 1)
    return picks


def turn_matrix(board: pl.DataFrame) -> pl.DataFrame:
    """For each draft slot, the greedy VONA policy pick per round: at each of
    your picks, take the position with the largest [best-now - best-at-your-
    next-pick]; remove that player (deplete his position by one), and let
    everyone else deplete by ADP. Value at your final pick uses lookahead 12.
    Returns (slot, round1..roundN) of 'POS: Player'."""
    ba = best_available(board)
    # mutable per-position available lists sorted by VORP desc: (adp_rank, vorp, name)
    avail = {pos: sorted(
        ((int(r["adp_rank"]), float(r["vorp"]), r["name"])
         for r in board.filter(pl.col("position") == pos).iter_rows(named=True)),
        key=lambda t: -t[1]) for pos in POSITIONS}

    def best_at(pos, pick, taken):
        for r, v, n in avail[pos]:
            if n in taken:
                continue
            if r >= pick:
                return v, n
        return 0.0, ""

    rows = []
    for slot in range(1, TEAMS + 1):
        picks = snake_picks(slot, MATRIX_ROUNDS)
        taken: set[str] = set()
        rec = {"slot": slot}
        for ri, pk in enumerate(picks):
            nxt = picks[ri + 1] if ri + 1 < len(picks) else pk + 12
            scored = []
            for pos in POSITIONS:
                v_now, name_now = best_at(pos, pk, taken)
                v_next, _ = best_at(pos, nxt, taken)
                scored.append((v_now - v_next, v_now, pos, name_now))
            scored.sort(key=lambda t: (-t[0], -t[1]))
            _, _, pos, name = scored[0]
            taken.add(name)
            rec[f"round{ri + 1}"] = f"{pos}: {name}" if name else "--"
        rows.append(rec)
    return pl.DataFrame(rows)


def main() -> None:
    board = load_board()
    table = per_pick_table(board)
    table.write_csv(OUT_VONA)
    print(f"wrote {OUT_VONA} ({table.height} rows)")

    # integrity: VONA non-negative everywhere
    vona_cols = [c for c in table.columns if c.startswith("vona")]
    mn = min(float(table[c].min()) for c in vona_cols)
    assert mn >= 0.0, f"negative VONA found ({mn}) — availability model broken"

    mtx = turn_matrix(board)
    print("\n=== snake-turn guide (opponents draft by ADP; positional timing only) ===")
    hdr = f"{'slot':>4s} | " + " | ".join(f"round {r}" for r in range(1, MATRIX_ROUNDS + 1))
    print(hdr)
    for r in mtx.iter_rows(named=True):
        cells = " | ".join(f"{r[f'round{i}']:<16s}" for i in range(1, MATRIX_ROUNDS + 1))
        print(f"{r['slot']:>4d} | {cells}")


if __name__ == "__main__":
    main()
