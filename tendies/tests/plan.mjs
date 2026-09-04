/* The plan DP against brute force, run under node by test_engine.py.
 *
 * The DP prices every candidate off one backward pass over 544 packed states,
 * with the flex bound folded into reachability and a lexicographic feasibility
 * term riding alongside the value. That is exactly the kind of code that gives
 * plausible answers while being wrong — an off-by-one in the dense key, a flex
 * bound applied to the wrong side of a transition, a depth index that forgets
 * whose picks it is counting. So on a draft small enough to enumerate, every
 * action sequence is evaluated by hand and the DP has to match.
 *
 * Small league on purpose: 4 teams x 5 rounds gives the seat 5 turns and 9
 * lineup slots, so feasibility genuinely binds and the K/D-ST constraint is
 * exercised rather than being trivially satisfiable.
 */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const E = require('../tendies/web/engine.js');

const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];
const league = {
  teams: 4, rounds: 5,
  starters: { QB: 1, RB: 2, WR: 2, TE: 1, K: 1, DST: 1 },
  flex: 1, bench: 0,
  limits: { QB: 4, RB: 8, WR: 8, TE: 3, K: 3, DST: 3 },
};
const empty = new Float64Array([-29.1, -28.3, -19.6, -6.3, 0, 0]);
const blind = new Uint8Array([0, 0, 0, 0, 1, 1]);   // K and D-ST carry no VORP curve
// a hold bonus on the two streamable slots (the shipped shape, vorp.HOLD_GAIN),
// so the DP's dedicated-slot gain `max(0, x - empty) + hold` is exercised
// rather than being trivially the no-hold case
const hold = new Float64Array([15, 0, 0, 6, 0, 0]);
const spec = E.lineupSpec(league, positions, empty, blind, hold);
const K = E.PLAN_K, nPos = positions.length;

const rnd = E.mulberry32(99);
const rows = [];
for (let i = 0; i < 120; i++) {
  rows.push({
    pid: i, name: 'p' + i, pos: i % 6, team: '', bye: 0,
    adp: i + 1, adpRank: i + 1, posRank: Math.floor(i / 6) + 1,
    ecr: null, vorp: Math.round((120 - i) * 1.4 - 45),
  });
}
const board = E.makeBoard(rows, positions);
// flow absent on purpose: the tail then reads straight off the board, so every
// number the DP sees is one this file can reproduce exactly.
const model = { positions, flow: null, noCurve: ['K', 'DST'] };

function bruteForce(st, seat, sim, turns) {
  const T = turns.length;
  const tail = new Float64Array(nPos * K);
  E.topKAvail(st, null, K, tail, 0);
  const availAt = (t, q, k) => {
    if (k >= K) return -Infinity;
    if (t <= E.MODEL_TURNS) return sim.avail[(t - 1) * nPos * K + q * K + k];
    return tail[q * K + k];
  };
  const cnt0 = E.planStateOf(spec, st.counts[seat]);
  const flexUsed = (c) => {
    let u = 0;
    for (let p = 0; p < nPos; p++) if (spec.flexOk[p]) u += Math.max(0, c[p] - spec.starters[p]);
    return u;
  };
  let bestV = -Infinity, bestF = Infinity;
  const c = Int32Array.from(cnt0);
  const go = (t, value) => {
    if (t >= T) {
      let f = 0;
      for (let p = 0; p < nPos; p++) {
        if (spec.mandatory[p]) f += Math.max(0, spec.starters[p] - c[p]);
      }
      if (f < bestF || (f === bestF && value > bestV)) { bestF = f; bestV = value; }
      return;
    }
    go(t + 1, value);                                    // bench
    for (let q = 0; q < nPos; q++) {
      if (c[q] >= spec.cap[q]) continue;
      c[q] += 1;
      if (flexUsed(c) <= spec.flex) {
        const k = Math.max(0, c[q] - 1 - cnt0[q]);
        const x = availAt(t, q, k);
        if (x > -Infinity) {
          const g = (c[q] - 1) < spec.starters[q]
            ? Math.max(0, x - spec.empty[q]) + spec.hold[q]
            : (spec.flexOk[q] ? Math.max(0, x - spec.emptyFlex) : 0);
          go(t + 1, value + g);
        }
      }
      c[q] -= 1;
    }
  };
  go(1, 0);
  return { v: bestV, f: bestF };
}

let trials = 0, worstV = 0, badF = 0, cases = [];
for (let trial = 0; trial < 60; trial++) {
  const st = E.newState(board, league, ['', '', '', '']);
  // give the seat a partial roster by walking a few picks off the top
  const pre = Math.floor(rnd() * 12);
  for (let i = 0; i < pre; i++) {
    const open = E.openPositions(st, E.seatOnClock(st));
    const p = open[Math.floor(rnd() * open.length)];
    const av = E.available(st, p, 3);
    E.advance(st, av.length ? av[Math.floor(rnd() * av.length)] : null);
  }
  const seat = E.seatOnClock(st);
  if (seat < 0) continue;
  const turns = E.myTurns(st, seat);
  if (turns.length < 2) continue;

  // random availability at the two model-driven turns, sorted descending as the
  // simulation would produce it, with occasional exhausted positions
  const avail = new Float64Array(E.MODEL_TURNS * nPos * K);
  for (let t = 0; t < E.MODEL_TURNS; t++) {
    for (let q = 0; q < nPos; q++) {
      const gone = rnd() < 0.12;
      const vs = [];
      for (let k = 0; k < K; k++) vs.push(Math.round(rnd() * 160 - 60));
      vs.sort((a, b) => b - a);
      for (let k = 0; k < K; k++) {
        avail[(t * nPos + q) * K + k] = gone ? -Infinity : vs[k];
      }
    }
  }
  const sim = { avail, availPaths: 1, availK: K };
  const pl = E.plan(model, st, seat, sim, spec);
  const want = bruteForce(st, seat, sim, turns);
  const got = { v: pl.v[pl.i0], f: pl.f[pl.i0] };
  const dv = Math.abs(got.v - want.v);
  if (dv > worstV) worstV = dv;
  if (got.f !== want.f) { badF++; cases.push({ trial, got, want, turns }); }
  if (dv > 1e-9 && cases.length < 5) cases.push({ trial, got, want, turns });
  trials++;
}

const ok = trials > 30 && worstV < 1e-9 && badF === 0;
process.stdout.write(JSON.stringify({ trials, worstV, badF, ok, cases: cases.slice(0, 5) }));
