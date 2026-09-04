/* Property tests for the lineup engine, run under node by test_engine.py.
 *
 * Two identities carry every number on the recommendation panel, and if either
 * is false nothing else in the codebase would tell us.
 *
 * 1. CUTOFF. Adding a player at position P to roster R changes the starting
 *    lineup by exactly `max(0, vorp - cutoff_P(R))`. Six numbers therefore
 *    describe a roster completely for this purpose, which is what makes the
 *    plan cheap enough to recompute on every click. (Before the empty-slot
 *    floors this needed a second case for empty slots and a second probe to
 *    detect them; with the floors an empty slot is just a slot whose cutoff is
 *    EMPTY, and the identity is uniform.) With a hold bonus on a streamable
 *    slot (vorp.HOLD_GAIN) it is `max(0, vorp - cutoff) + fillBonus`, and
 *    every check below runs on a spec with and a spec without one.
 *
 * 2. ASSIGNMENT. Greedy slot-filling — best players into their dedicated
 *    slots, then FLEX from the leftovers — is OPTIMAL. It is, but only while
 *    `EMPTY_FLEX >= max(EMPTY_RB, EMPTY_WR, EMPTY_TE)`. Set the flex floor to
 *    their min instead and greedy loses up to 42 points, so the precondition
 *    is asserted in lineupSpec and this test brute-forces the claim against
 *    every legal assignment.
 */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const E = require('../tendies/web/engine.js');

const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];
const league = {
  teams: 10, rounds: 15,
  starters: { QB: 1, RB: 2, WR: 2, TE: 1, K: 1, DST: 1 },
  flex: 2, bench: 5,
  limits: { QB: 4, RB: 8, WR: 8, TE: 3, K: 3, DST: 3 },
};
// Floors in the shape the real board produces: deep negatives for the skill
// positions, exactly zero for the two with no VORP curve.
const empty = new Float64Array([-29.1, -28.3, -19.6, -6.3, 0, 0]);
const blind = new Uint8Array([0, 0, 0, 0, 1, 1]);
const spec = E.lineupSpec(league, positions, empty, blind);
// the same league with the shipped-shape hold bonus on the two streamable slots
// (vorp.HOLD_GAIN is QB 15 / TE 6); every identity below is checked on both
const specHold = E.lineupSpec(league, positions, empty, blind,
                              new Float64Array([15, 0, 0, 6, 0, 0]));

const rnd = E.mulberry32(20482930);
const rows = [];
for (let i = 0; i < 300; i++) {
  rows.push({
    pid: i, name: 'p' + i, pos: i % 6, team: '', bye: 0,
    adp: i + 1, adpRank: i + 1, posRank: Math.floor(i / 6) + 1,
    ecr: null, vorp: Math.round((300 - i) * 0.7 - 40),
  });
}
const board = E.makeBoard(rows, positions);

/* Exact best legal lineup, by search over slot assignments with memoisation on
 * (slot index, players used). Slow and obviously correct — the point is that it
 * makes no assumption about which player belongs in which slot. */
function optimalLineup(roster, S = spec) {
  const slots = [];
  for (let p = 0; p < S.nPos; p++) {
    for (let i = 0; i < S.starters[p]; i++) slots.push({ pos: p, floor: S.empty[p], hold: S.hold[p] });
  }
  for (let i = 0; i < S.flex; i++) slots.push({ pos: -1, floor: S.emptyFlex, hold: 0 });
  const n = roster.length;
  const memo = new Map();
  const go = (si, used) => {
    if (si === slots.length) return 0;
    const key = si * (1 << n) + used;
    const hit = memo.get(key);
    if (hit !== undefined) return hit;
    let best = slots[si].floor + go(si + 1, used);          // leave it empty
    for (let i = 0; i < n; i++) {
      if (used & (1 << i)) continue;
      const p = board.pos[roster[i]];
      const ok = slots[si].pos === -1 ? S.flexOk[p] : slots[si].pos === p;
      if (!ok) continue;
      const v = Math.max(board.vorp[roster[i]], slots[si].floor) + slots[si].hold
              + go(si + 1, used | (1 << i));
      if (v > best) best = v;
    }
    memo.set(key, best);
    return best;
  };
  return go(0, 0);
}

let checked = 0, worst = 0, worstCase = null;
let assignChecked = 0, assignWorst = 0, assignCase = null;
let slotChecked = 0, slotErr = 0;
for (let trial = 0; trial < 4000; trial++) {
  const S = trial % 2 ? specHold : spec;
  const size = Math.floor(rnd() * 12);
  const roster = [];
  const used = new Set();
  for (let i = 0; i < size; i++) {
    const pid = Math.floor(rnd() * board.n);
    if (used.has(pid)) continue;
    const name = positions[board.pos[pid]];
    const have = roster.filter((q) => board.pos[q] === board.pos[pid]).length;
    if (have >= league.limits[name]) continue;      // as the state machine does
    used.add(pid); roster.push(pid);
  }
  const cut = E.cutoffs(board, S, roster);
  const fill = E.fillBonus(board, S, roster);
  for (let k = 0; k < 6; k++) {
    const pid = Math.floor(rnd() * board.n);
    if (used.has(pid)) continue;
    const p = board.pos[pid];
    const err = Math.abs(E.marginal(board, S, roster, pid)
                       - E.marginalAt(cut, board.vorp[pid], p, fill));
    if (err > worst) { worst = err; worstCase = { roster: roster.slice(), pid, hold: S === specHold }; }
    checked++;
    // The player card displays the slots; the panel quotes the total. They are
    // computed by two different functions, so they are asserted to agree —
    // otherwise the card can show a lineup that does not add up to its own EV.
    const sum = E.lineupSlots(board, S, roster, pid).reduce((a, s) => a + s.value, 0);
    const want = E.lineupValueWith(board, S, roster,
                                   { vorp: board.vorp[pid], pos: p });
    slotErr = Math.max(slotErr, Math.abs(sum - want));
    slotChecked++;
  }
  {
    const sum = E.lineupSlots(board, S, roster).reduce((a, s) => a + s.value, 0);
    slotErr = Math.max(slotErr, Math.abs(sum - E.lineupValue(board, S, roster)));
    slotChecked++;
  }
  if (roster.length <= 11) {
    const err = Math.abs(E.lineupValue(board, S, roster) - optimalLineup(roster, S));
    if (err > assignWorst) { assignWorst = err; assignCase = roster.slice(); }
    assignChecked++;
  }
}

// An empty dedicated slot must price at exactly its floor, which is the whole
// point of the rewrite: filling it with a below-replacement player is worth
// something, and leaving it open costs what streaming costs.
let floorErr = 0;
for (const S of [spec, specHold]) {
  // the threshold alone: the hold a body would earn on top rides in fillBonus
  const bare = E.cutoffs(board, S, []);
  for (let p = 0; p < S.nPos; p++) floorErr = Math.max(floorErr, Math.abs(bare[p] - S.empty[p]));
}

/* The precondition, proved rather than asserted: rebuild the same spec with the
 * flex floor set to the MIN of the three instead of the max — the plausible
 * "simplification" — and greedy stops being optimal. If this ever comes back
 * zero, the max is no longer load-bearing and the comment in lineupSpec is
 * wrong; far more likely, the test broke. */
const badFlex = Object.assign({}, spec, {
  emptyFlex: Math.min(spec.empty[1], spec.empty[2], spec.empty[3]),
});
let badWorst = 0;
{
  const r2 = E.mulberry32(7);
  const optBad = (roster) => {
    const saved = spec.emptyFlex;
    spec.emptyFlex = badFlex.emptyFlex;
    const v = optimalLineup(roster);
    spec.emptyFlex = saved;
    return v;
  };
  for (let trial = 0; trial < 500; trial++) {
    const roster = [];
    for (let i = 0; i < 6; i++) {
      const pid = Math.floor(r2() * board.n);
      if (roster.indexOf(pid) < 0) roster.push(pid);
    }
    badWorst = Math.max(badWorst, optBad(roster) - E.lineupValue(board, badFlex, roster));
  }
}

const TOL = 1e-6;
const ok = worst < TOL && assignWorst < TOL && floorErr < TOL && slotErr < TOL
  && badWorst > 1.0;
process.stdout.write(JSON.stringify({
  checked, worst, assignChecked, assignWorst, floorErr, badWorst, ok,
  slotChecked, slotErr,
  worstCase: worst < TOL ? null : worstCase,
  assignCase: assignWorst < TOL ? null : assignCase,
}));
