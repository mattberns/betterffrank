/* Draft engine: the JavaScript twin of replay.py / features.py / predict.py /
 * simulate.py.
 *
 * This file must produce bit-comparable numbers to the Python. tests/parity.mjs
 * replays a set of golden draft states through both and asserts they agree to
 * 1e-9, which is the test that stands between "the model is right" and "the
 * page is right".
 *
 * Rules that keep the two sides in step, each of which has a matching comment
 * in the Python:
 *   - natural log everywhere; counts are floats before standardisation
 *   - a missing bye scores 0, never NaN
 *   - softmax is always max-subtracted (the two languages diverge in the tails
 *     without it)
 *   - feature ORDER is the contract; it ships in the payload and is asserted
 *     against these constants at load
 */
'use strict';

/* NOTE: log_h, rounds_left and starters_filled are deliberately absent — they
 * are occasion-level, identical across every position in a choice set, and so
 * cancel in the conditional-logit softmax. See features.py. */
const POS_FEATURES = [
  'first_at_pos', 'starter_need', 'flex_need', 'log_count', 'best_vorp',
  'vona', 'n_gone', 'adp_lead', 'run5', 'run10', 'endgame_k', 'endgame_dst',
  'early_qb', 'early_te',
];
const PLR_FEATURES = [
  'u_adp', 'posrank_gap', 'ecr_lean', 'ecr_missing', 'vorp_gap',
  'bye_clash', 'same_team', 'is_rookie', 'age', 'was_mine', 'durability',
];
const FLEX_POSITIONS = ['RB', 'WR', 'TE'];

/* Board columns this engine reads. assertContract guards the FEATURE lists;
 * without this a payload that forgot a board column would read `undefined` as 0
 * for every player and show up only as a numeric parity diff with no cause. */
const BOARD_FIELDS = [
  'pid', 'name', 'pos', 'team', 'bye', 'adp', 'adpRank', 'posRank', 'ecr',
  'vorp', 'vorpAdp', 'vorpSe', 'edgeSe', 'rookie', 'age', 'durability',
  'prevOwner',
];

/* Fallback when a payload predates the standard errors. It must be a LOUD
 * number rather than 0: a zero SE reads as "this curve is exact", which is the
 * one conclusion the measurement in vorp.py rules out, and it would silently
 * restore the confident ranking the ties exist to replace. Infinity ties
 * everything to everything, which is visibly wrong instead of quietly wrong. */
const NO_SE = Infinity;

function assertBoard(rows) {
  if (!rows || !rows.length) throw new Error('empty board');
  const have = new Set(Object.keys(rows[0]));
  const missing = BOARD_FIELDS.filter((f) => !have.has(f));
  if (missing.length) {
    throw new Error('board field contract mismatch, missing: ' + missing);
  }
}

function assertContract(model) {
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  if (!same(model.position.features, POS_FEATURES)) {
    throw new Error('position feature contract mismatch: ' +
      model.position.features + ' vs ' + POS_FEATURES);
  }
  if (!same(model.player.features, PLR_FEATURES)) {
    throw new Error('player feature contract mismatch: ' +
      model.player.features + ' vs ' + PLR_FEATURES);
  }
}

/* ---------------------------------------------------------------- board --- */

function makeBoard(rows, positions) {
  const n = rows.length;
  const b = {
    n, positions,
    name: new Array(n), team: new Array(n),
    pos: new Int8Array(n), adpRank: new Int32Array(n), posRank: new Int32Array(n),
    ecr: new Float64Array(n), hasEcr: new Uint8Array(n),
    vorp: new Float64Array(n), vorpAdp: new Float64Array(n),
    vorpSe: new Float64Array(n), edgeSe: new Float64Array(n),
    bye: new Int16Array(n),
    rookie: new Uint8Array(n),
    age: new Float64Array(n), durability: new Float64Array(n),
    prevOwner: new Array(n),
    byPos: [], slotOf: new Int32Array(n),
  };
  rows.forEach((r, i) => {
    b.name[i] = r.name; b.team[i] = r.team || '';
    b.pos[i] = r.pos; b.adpRank[i] = r.adpRank; b.posRank[i] = r.posRank;
    b.ecr[i] = r.ecr == null ? 0 : r.ecr; b.hasEcr[i] = r.ecr == null ? 0 : 1;
    b.vorp[i] = r.vorp; b.bye[i] = r.bye || 0;
    b.rookie[i] = r.rookie ? 1 : 0;
    b.age[i] = r.age || 0; b.durability[i] = r.durability || 0;
    b.prevOwner[i] = r.prevOwner || '';
    // priced at his MARKET slot instead of his expert slot; the gap is the
    // expert-vs-market edge in points (see vorp.attach_vorp)
    b.vorpAdp[i] = r.vorpAdp == null ? r.vorp : r.vorpAdp;
    b.vorpSe[i] = r.vorpSe == null ? NO_SE : r.vorpSe;
    b.edgeSe[i] = r.edgeSe == null ? NO_SE : r.edgeSe;
  });
  for (let p = 0; p < positions.length; p++) {
    const pids = [];
    for (let i = 0; i < n; i++) if (b.pos[i] === p) pids.push(i);
    b.byPos.push(Int32Array.from(pids));
    pids.forEach((pid, k) => { b.slotOf[pid] = k; });
  }
  return b;
}

/* ---------------------------------------------------------------- state --- */

function snakeSeats(teams, rounds) {
  const out = [];
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < teams; i++) out.push(r % 2 === 0 ? i + 1 : teams - i);
  }
  return out;
}

function newState(board, league, managers, seats) {
  const nPos = board.positions.length;
  return {
    board, league, managers,
    seats: seats || snakeSeats(league.teams, league.rounds),
    picks: [],
    taken: new Uint8Array(board.n),
    counts: Array.from({ length: league.teams }, () => new Int16Array(nPos)),
    rosters: Array.from({ length: league.teams }, () => []),
    cursor: new Int32Array(nPos),
    recent: [],
  };
}

function pickNo(st) { return st.picks.length + 1; }

function seatOnClock(st) {
  const i = st.picks.length;
  return i < st.seats.length ? st.seats[i] - 1 : -1;
}

function nextPickFor(st, seat, after) {
  const start = after === undefined ? st.picks.length : after;
  for (let i = start; i < st.seats.length; i++) if (st.seats[i] - 1 === seat) return i + 1;
  return null;
}

function horizon(st, seat) {
  const i = st.picks.length;
  const onClock = i < st.seats.length && st.seats[i] - 1 === seat;
  const nxt = nextPickFor(st, seat, onClock ? i + 1 : i);
  if (nxt === null) return 0;
  return nxt - pickNo(st) - (onClock ? 1 : 0);
}

function advanceCursor(st, p) {
  const arr = st.board.byPos[p];
  let c = st.cursor[p];
  while (c < arr.length && st.taken[arr[c]]) c++;
  st.cursor[p] = c;
}

function available(st, p, k) {
  advanceCursor(st, p);
  const arr = st.board.byPos[p];
  const out = [];
  for (let i = st.cursor[p]; i < arr.length && out.length < k; i++) {
    if (!st.taken[arr[i]]) out.push(arr[i]);
  }
  return out;
}

function openPositions(st, seat) {
  const out = [];
  for (let p = 0; p < st.board.positions.length; p++) {
    const name = st.board.positions[p];
    const cap = st.league.limits[name] === undefined ? 99 : st.league.limits[name];
    if (st.counts[seat][p] >= cap) continue;
    advanceCursor(st, p);
    if (st.cursor[p] < st.board.byPos[p].length) out.push(p);
  }
  return out;
}

function choiceSet(st, seat, model, kappa) {
  kappa = kappa === undefined ? 1.0 : kappa;
  const cs = {};
  for (const p of openPositions(st, seat)) {
    const name = st.board.positions[p];
    const k = Math.max(1, Math.round(model.window[name] * kappa));
    cs[p] = available(st, p, k);
  }
  return cs;
}

function advance(st, pid) {
  const seat = seatOnClock(st);
  st.picks.push(pid);
  if (pid === null || pid === undefined) { st.recent.push(-1); return; }
  st.taken[pid] = 1;
  const p = st.board.pos[pid];
  if (seat >= 0) { st.counts[seat][p] += 1; st.rosters[seat].push(pid); }
  st.recent.push(p);
}

function undo(st) {
  if (!st.picks.length) return;
  const pid = st.picks.pop();
  st.recent.pop();
  if (pid === null || pid === undefined) return;
  const seat = seatOnClock(st);
  st.taken[pid] = 0;
  const p = st.board.pos[pid];
  if (seat >= 0) {
    st.counts[seat][p] -= 1;
    const r = st.rosters[seat];
    const at = r.indexOf(pid);
    if (at >= 0) r.splice(at, 1);
  }
  st.cursor[p] = Math.min(st.cursor[p], st.board.slotOf[pid]);
}

function forkState(st) {
  const c = {
    board: st.board, league: st.league, managers: st.managers, seats: st.seats,
    picks: st.picks.slice(),
    taken: st.taken.slice(),
    counts: st.counts.map((x) => x.slice()),
    rosters: st.rosters.map((r) => r.slice()),
    cursor: st.cursor.slice(),
    recent: st.recent.slice(),
  };
  return c;
}

function runs(st, window) {
  const out = new Float64Array(st.board.positions.length);
  const from = Math.max(0, st.recent.length - window);
  for (let i = from; i < st.recent.length; i++) if (st.recent[i] >= 0) out[st.recent[i]] += 1;
  return out;
}

/* ------------------------------------------------------------- features --- */

function positionFeatures(st, seat, cs, model) {
  const B = st.board, L = st.league, P = B.positions;
  const pick = pickNo(st), h = horizon(st, seat), counts = st.counts[seat];
  const run5 = runs(st, 5), run10 = runs(st, 10);
  const roundsLeft = L.rounds - Math.floor((pick - 1) / L.teams);
  const rnd = Math.floor((pick - 1) / L.teams) + 1;
  const horizonRank = pick + h;
  const positions = Object.keys(cs).map(Number).sort((a, b) => a - b);
  const X = [];
  for (const p of positions) {
    const cand = cs[p], name = P[p], best = cand[0];
    const bestVorp = B.vorp[best];
    let vorpLater = bestVorp, nGone = 0;
    let found = false;
    for (const q of cand) {
      if (B.adpRank[q] >= horizonRank) { if (!found) { vorpLater = B.vorp[q]; found = true; } }
      else nGone++;
    }
    const dedicated = L.starters[name] || 0;
    let flexNeed = 0;
    if (FLEX_POSITIONS.indexOf(name) >= 0 && counts[p] >= dedicated) {
      let used = 0;
      for (const q of FLEX_POSITIONS) {
        const qi = P.indexOf(q);
        if (qi >= 0) used += Math.max(0, counts[qi] - (L.starters[q] || 0));
      }
      flexNeed = Math.max(0, L.flex - used);
    }
    X.push([
      counts[p] === 0 ? 1.0 : 0.0,
      Math.max(0, dedicated - counts[p]),
      flexNeed,
      Math.log1p(counts[p]),
      bestVorp / 100.0,
      (bestVorp - vorpLater) / 100.0,
      nGone,
      (B.adpRank[best] - pick) / 10.0,
      run5[p], run10[p],
      (name === 'K' && roundsLeft <= model.endgameRounds) ? 1.0 : 0.0,
      (name === 'DST' && roundsLeft <= model.endgameRounds) ? 1.0 : 0.0,
      (name === 'QB' && rnd <= model.earlyRounds) ? 1.0 : 0.0,
      (name === 'TE' && rnd <= model.earlyRounds) ? 1.0 : 0.0,
    ]);
  }
  return { positions, X };
}

function playerFeatures(st, seat, cand) {
  const B = st.board, roster = st.rosters[seat];
  const me = st.managers[seat] || '';
  const myByes = [], myTeams = new Set();
  for (const q of roster) { if (B.bye[q] > 0) myByes.push(B.bye[q]); if (B.team[q]) myTeams.add(B.team[q]); }
  const bestAdp = B.adpRank[cand[0]], bestPosRank = B.posRank[cand[0]], bestVorp = B.vorp[cand[0]];
  const X = [];
  for (const j of cand) {
    const adp = B.adpRank[j];
    let lean = 0.0;
    if (B.hasEcr[j]) lean = Math.log((B.ecr[j] + 10.0) / (adp + 10.0));
    let clash = 0.0;
    if (myByes.length && B.bye[j] > 0) {
      let c = 0;
      for (const b of myByes) if (b === B.bye[j]) c++;
      clash = Math.min(3.0, c);
    }
    X.push([
      -Math.log((adp + 10.0) / (bestAdp + 10.0)),
      (B.posRank[j] - bestPosRank) / 10.0,
      lean,
      B.hasEcr[j] ? 0.0 : 1.0,
      (B.vorp[j] - bestVorp) / 100.0,
      clash,
      (B.team[j] && myTeams.has(B.team[j])) ? 1.0 : 0.0,
      B.rookie[j] ? 1.0 : 0.0,
      B.age[j],
      (me && B.prevOwner[j] === me) ? 1.0 : 0.0,
      B.durability[j],
    ]);
  }
  return X;
}

function standardize(X, mean, sd) {
  return X.map((row) => row.map((v, i) => (v - mean[i]) / sd[i]));
}

/* -------------------------------------------------------------- scoring --- */

function logSumExp(u) {
  let m = -Infinity;
  for (const x of u) if (x > m) m = x;
  let s = 0;
  for (const x of u) s += Math.exp(x - m);
  return m + Math.log(s);
}

function softmax(u) {
  let m = -Infinity;
  for (const x of u) if (x > m) m = x;
  const e = u.map((x) => Math.exp(x - m));
  let s = 0;
  for (const x of e) s += x;
  return e.map((x) => x / s);
}

const U_ADP_COL = PLR_FEATURES.indexOf('u_adp');

function playerUtilities(model, st, seat, p, cand, managerIdx) {
  const M = model.player;
  const raw = playerFeatures(st, seat, cand);
  // Per-manager channels, in the payload's order so the floating-point sum
  // matches Python's. Each is a coefficient on a standardised player feature;
  // `reach` (on u_adp) is one of them, not a special case.
  const chans = [];
  if (managerIdx != null && M.channels) {
    for (const ch of M.channels) {
      const b = ch.beta[managerIdx];
      if (b === undefined) continue;
      const col = PLR_FEATURES.indexOf(ch.feature);
      if (col < 0) throw new Error('unknown channel feature: ' + ch.feature);
      chans.push([col, b]);
    }
  }
  const u = new Array(raw.length + 1);
  for (let j = 0; j < raw.length; j++) {
    const row = raw[j];
    let acc = 0;
    for (let i = 0; i < row.length; i++) acc += ((row[i] - M.mean[i]) / M.sd[i]) * M.beta[i];
    const z = (row[U_ADP_COL] - M.mean[U_ADP_COL]) / M.sd[U_ADP_COL];
    acc += z * M.posAdp[p];
    for (const [col, b] of chans) acc += b * ((row[col] - M.mean[col]) / M.sd[col]);
    u[j] = acc;
  }
  u[raw.length] = M.outside[p];
  return u;
}

/* One pass over the state: player utilities per position are computed ONCE and
 * reused for both the inclusive value and the final player probabilities.
 * Computing them twice (the obvious structure) doubled the simulation cost,
 * which is the difference between a usable page and a 4-second stall. */
function scoreState(model, st, seat, manager, autoWeight, kappa) {
  const cs = choiceSet(st, seat, model, kappa);
  const positions = Object.keys(cs).map(Number).sort((a, b) => a - b);
  if (!positions.length) return { pids: [], probs: [], outside: 1.0, cs, positions: [] };
  const mi = (manager != null && model.managers[manager] !== undefined)
    ? model.managers[manager] : null;

  const utils = [], iv = [];
  for (const p of positions) {
    const u = playerUtilities(model, st, seat, p, cs[p], mi);
    utils.push(u);
    iv.push(logSumExp(u));
  }

  const M = model.position;
  const { X } = positionFeatures(st, seat, cs, model);
  const rnd = Math.floor((pickNo(st) - 1) / model.league.teams) + 1;
  const posU = (side, useManager) => {
    const u = new Array(positions.length);
    for (let i = 0; i < positions.length; i++) {
      const name = st.board.positions[positions[i]];
      let acc = 0;
      for (let k = 0; k < X[i].length; k++) {
        acc += ((X[i][k] - M.mean[k]) / M.sd[k]) * side.beta[k];
      }
      const ai = M.ascPositions.indexOf(name);
      if (ai >= 0) acc += side.asc[ai];
      acc += side.lambda * iv[i];
      if (useManager && mi != null && side.managerAsc && side.managerAsc.length) {
        const mp = side.managerPos.indexOf(name);
        if (mp >= 0) {
          const idx = mi * side.managerPos.length + mp;
          if (idx < side.managerAsc.length) acc += side.managerAsc[idx];
        }
      }
      if (useManager && mi != null && side.managerEarly && side.managerEarly.length
          && rnd <= model.earlyRounds) {
        const me = side.managerEarlyPos.indexOf(name);
        if (me >= 0) {
          const idx = mi * side.managerEarlyPos.length + me;
          if (idx < side.managerEarly.length) acc += side.managerEarly[idx];
        }
      }
      u[i] = acc;
    }
    return u;
  };

  let pp = softmax(posU(M.human, true));
  if (autoWeight > 0) {
    const pa = softmax(posU(M.auto, false));
    pp = pp.map((v, i) => (1 - autoWeight) * v + autoWeight * pa[i]);
  }

  const pids = [], probs = [];
  let outside = 0;
  for (let i = 0; i < positions.length; i++) {
    const q = softmax(utils[i]);
    const cand = cs[positions[i]];
    for (let j = 0; j < cand.length; j++) { pids.push(cand[j]); probs.push(pp[i] * q[j]); }
    outside += pp[i] * q[q.length - 1];
  }
  return { pids, probs, outside, cs, positions, posProbs: pp };
}

function positionProbs(model, st, seat, cs, managerIdx, isAuto) {
  const manager = managerIdx == null ? null
    : Object.keys(model.managers).find((k) => model.managers[k] === managerIdx);
  const r = scoreState(model, st, seat, manager, isAuto ? 1 : 0);
  return { positions: r.positions, probs: r.posProbs };
}

function jointProbs(model, st, seat, manager, autoWeight, kappa) {
  return scoreState(model, st, seat, manager, autoWeight, kappa);
}

/* ---------------------------------------------------------------- lineup --- */

/* An unfilled starting slot is NOT worth zero.
 *
 * VORP is measured OVER REPLACEMENT, so 0 means "a replacement-level player",
 * not "nobody". Scoring an empty slot at 0 made a below-replacement back
 * (VORP -24) look worse than a sixth wide receiver you would never start (0),
 * and the recommendation dutifully refused to fill the slot: taking its top
 * pick fifteen times built five quarterbacks, one running back and no kicker.
 * Real managers in this league filled every starting slot in 70 of 70
 * team-seasons.
 *
 * The floor is what you would actually field, measured rather than assumed:
 * the best player at that position who goes UNDRAFTED, read off the positional
 * flow table (see depletion.py), and a roster therefore never scores below
 * streaming, because in reality you would bench the player and stream.
 *
 * At a STREAMABLE position (`model.streamable` — QB and TE) that floor is
 * `max(0, ...)`. vorp.REPL_RANKS already prices those two against a streaming
 * total, so an empty QB slot costs exactly 0 by construction; charging it the
 * best undrafted quarterback instead would discount the same streaming twice
 * and pay the drafter for it. The max still lets the live board raise the
 * floor when the room genuinely punts a position.
 *
 *     lineupValue = SUM over slots of max(occupant, EMPTY_slot)
 *
 * Two consequences. `cutoffs` collapses to ONE number per position — the old
 * "is this slot empty" flag is gone, since an empty slot is just a slot whose
 * cutoff is EMPTY. And K/D-ST come out at exactly 0 (they carry no VORP curve
 * at all), so the plan handles them as a legality constraint, not a price.
 */

const PROBE = 1e6;

/** Mean cumulative picks by position at overall `pick` of a `total`-pick draft. */
function goneBy(flow, pick, total) {
  const steps = flow[0].length - 1;
  const x = Math.min(Math.max(pick / Math.max(1, total), 0), 1) * steps;
  const lo = Math.floor(x), hi = Math.min(lo + 1, steps), w = x - lo;
  const out = new Float64Array(flow.length);
  for (let p = 0; p < flow.length; p++) out[p] = flow[p][lo] * (1 - w) + flow[p][hi] * w;
  return out;
}

/* VORP of the best player at each position who is still undrafted when the
 * draft ends — the price of leaving a starting slot open. Counted against the
 * board as it stands now, so a draft that has already run RB-heavy gets a
 * worse RB floor than the historical average would give it. */
function emptyValues(model, st) {
  const B = st.board, nPos = B.positions.length;
  const out = new Float64Array(nPos);
  if (!model.flow) return out;
  const streamable = model.streamable || [];
  const total = st.league.teams * st.league.rounds;
  const now = goneBy(model.flow, st.picks.length, total);
  const end = goneBy(model.flow, total, total);
  for (let p = 0; p < nPos; p++) {
    const d = Math.round(Math.max(0, end[p] - now[p]));
    const arr = B.byPos[p];
    let seen = 0, best = 0, found = false;
    for (let i = 0; i < arr.length; i++) {
      if (st.taken[arr[i]]) continue;
      if (seen++ < d) continue;
      const v = B.vorp[arr[i]];
      if (!found || v > best) { best = v; found = true; }
    }
    const floor = found ? best : 0;
    out[p] = streamable.indexOf(B.positions[p]) >= 0 ? Math.max(0, floor) : floor;
  }
  return out;
}

/* Positions the value objective is BLIND to: those with no VORP curve, where
 * every player scores the same 0 (K and D-ST — see vorp.CURVE_POSITIONS). They
 * need separate handling twice over, and for different reasons: the plan cannot
 * see that you must field one, and it cannot see that WHEN you take one is not
 * arbitrary. Distinct from `mandatory` below, which is every dedicated slot. */
function blindMask(model, positions) {
  const m = new Uint8Array(positions.length);
  const names = model.noCurve || [];
  for (let p = 0; p < positions.length; p++) m[p] = names.indexOf(positions[p]) >= 0 ? 1 : 0;
  return m;
}

/* Everything about the lineup that does not depend on the roster: slot counts,
 * the empty-slot floors, and which positions the value objective is blind to. */
function lineupSpec(league, positions, empty, blind) {
  const nPos = positions.length;
  const starters = new Int32Array(nPos), flexOk = new Uint8Array(nPos);
  const cap = new Int32Array(nPos), mandatory = new Uint8Array(nPos);
  // The flex floor is the MAX of the floors it draws from, and that is not a
  // stylistic choice: an unfilled flex is streamed with the best of RB/WR/TE,
  // and greedy slot assignment is only optimal while the flex floor DOMINATES
  // the dedicated floors. Use their min instead and greedy loses up to 42
  // points on real rosters — tests/lineup.mjs brute-forces both directions.
  let emptyFlex = -Infinity;
  for (let p = 0; p < nPos; p++) {
    starters[p] = league.starters[positions[p]] || 0;
    flexOk[p] = FLEX_POSITIONS.indexOf(positions[p]) >= 0 ? 1 : 0;
    cap[p] = starters[p] + (flexOk[p] ? league.flex : 0);
    // Every DEDICATED slot is mandatory. You field a lineup or you forfeit the
    // slot, and this league's managers filled all of them in 70 of 70
    // team-seasons — so "fill the lineup, then maximise it" is the objective,
    // and the plan solves it lexicographically in that order.
    //
    // Leaving this to the value term alone is not enough, and the failure is
    // quiet: where the empty-slot floor is shallow (QB is only -5.8 on the 2026
    // board, because streaming a quarterback genuinely works) the incentive to
    // fill the slot is a couple of points, and a tie broken elsewhere can end
    // the draft with no quarterback at all. FLEX is deliberately NOT mandatory
    // — it is a value decision, and an unfilled flex is a real (if unusual)
    // choice.
    mandatory[p] = starters[p] > 0 ? 1 : 0;
    if (flexOk[p]) emptyFlex = Math.max(emptyFlex, empty[p]);
  }
  if (!isFinite(emptyFlex)) emptyFlex = 0;
  return {
    nPos, positions, starters, flexOk, cap, empty, emptyFlex,
    flex: league.flex, mandatory, blind: blind || new Uint8Array(nPos),
  };
}

function lineupValue(board, spec, roster) { return lineupValueWith(board, spec, roster, null); }

function lineupValueWith(board, spec, roster, probe) {
  const nPos = spec.nPos;
  const have = [];
  for (let p = 0; p < nPos; p++) have.push([]);
  for (const pid of roster) have[board.pos[pid]].push(board.vorp[pid]);
  if (probe) have[probe.pos].push(probe.vorp);
  let total = 0;
  const leftovers = [];
  for (let p = 0; p < nPos; p++) {
    const v = have[p];
    v.sort((a, b) => b - a);
    const need = spec.starters[p];
    for (let i = 0; i < need; i++) {
      total += i < v.length ? Math.max(v[i], spec.empty[p]) : spec.empty[p];
    }
    if (spec.flexOk[p]) for (let i = need; i < v.length; i++) leftovers.push(v[i]);
  }
  leftovers.sort((a, b) => b - a);
  for (let i = 0; i < spec.flex; i++) {
    total += i < leftovers.length ? Math.max(leftovers[i], spec.emptyFlex) : spec.emptyFlex;
  }
  return total;
}

/* The same greedy assignment `lineupValue` scores, but returning WHO fills each
 * slot instead of just the total. Nothing here decides anything — it exists so
 * the page can show the roster it is talking about, and so a test can assert
 * that the slots it displays sum to the value it quotes. `addPid` inserts a
 * hypothetical pick, which is the whole point on a draft page. */
function lineupSlots(board, spec, roster, addPid) {
  const nPos = spec.nPos;
  const have = [];
  for (let p = 0; p < nPos; p++) have.push([]);
  for (const pid of roster) have[board.pos[pid]].push({ pid, v: board.vorp[pid], probe: false });
  if (addPid !== undefined && addPid !== null && addPid >= 0) {
    have[board.pos[addPid]].push({ pid: addPid, v: board.vorp[addPid], probe: true });
  }
  const out = [], leftovers = [];
  const slot = (pos, label, occ, floor) => ({
    pos, label, pid: occ ? occ.pid : -1, probe: occ ? occ.probe : false,
    value: occ ? Math.max(occ.v, floor) : floor, floor,
  });
  for (let p = 0; p < nPos; p++) {
    const v = have[p];
    v.sort((a, b) => b.v - a.v);
    const need = spec.starters[p];
    for (let i = 0; i < need; i++) {
      out.push(slot(p, spec.positions[p] + (need > 1 ? i + 1 : ''),
                    i < v.length ? v[i] : null, spec.empty[p]));
    }
    if (spec.flexOk[p]) for (let i = need; i < v.length; i++) leftovers.push(v[i]);
  }
  leftovers.sort((a, b) => b.v - a.v);
  for (let i = 0; i < spec.flex; i++) {
    out.push(slot(-1, 'FLEX' + (spec.flex > 1 ? i + 1 : ''),
                  i < leftovers.length ? leftovers[i] : null, spec.emptyFlex));
  }
  return out;
}

/** Lineup value added by one more player — what a pick is actually worth. */
function marginal(board, spec, roster, pid) {
  return lineupValueWith(board, spec, roster, { vorp: board.vorp[pid], pos: board.pos[pid] })
       - lineupValue(board, spec, roster);
}

/* The VORP a new player at each position must beat to improve the lineup. One
 * probe per position is now enough: with the floors in place the identity is
 * uniformly `max(0, v - c)`, and an empty slot simply has `c = EMPTY[p]`. The
 * cutoff is routinely NEGATIVE (anyone below replacement has negative VORP and
 * late rosters are full of them), which is why this is a probe and not a
 * search over a positive range. */
function cutoffs(board, spec, roster) {
  const base = lineupValue(board, spec, roster);
  const out = new Float64Array(spec.nPos);
  for (let p = 0; p < spec.nPos; p++) {
    out[p] = PROBE - (lineupValueWith(board, spec, roster, { vorp: PROBE, pos: p }) - base);
  }
  return out;
}

/** Lineup value added by a player of value `v` at position `p`. */
function marginalAt(cut, v, p) { return Math.max(0, v - cut[p]); }

/** Starting slots (dedicated + flex) still unfilled. */
function openSlots(board, spec, roster) {
  const counts = new Int32Array(spec.nPos);
  for (const pid of roster) counts[board.pos[pid]] += 1;
  let open = 0, spare = 0;
  for (let p = 0; p < spec.nPos; p++) {
    open += Math.max(0, spec.starters[p] - counts[p]);
    if (spec.flexOk[p]) spare += Math.max(0, counts[p] - spec.starters[p]);
  }
  return open + Math.max(0, spec.flex - spare);
}

/* ----------------------------------------------------------------- plan --- */

/* Greedy is not enough even with the floors, because a pick spends a TURN as
 * well as filling a slot: taking a receiver now is only free if the plan was
 * not going to need that turn for the back it still owes. So value the lineup
 * you will FINISH the draft with, over all remaining turns at once.
 *
 * State is the vector of slots filled, capped at what the lineup can use
 * (QB 1, RB 4, WR 4, TE 3, K 1, D-ST 1 here) and bounded by the flex count —
 * 544 reachable states, solved backwards over the remaining turns with seven
 * actions each. That is ~57k O(1) transitions, well under a millisecond, and
 * because the state is a count vector ONE backward pass prices every candidate
 * as a table lookup: EV(x) = what x adds now + V(state after x, next turn).
 *
 * Two things the DP has to get right that a simpler version would not:
 *
 * **Depth, not just the best.** Availability is carried K-deep per position.
 * With only the best, a plan that takes RBs at two consecutive turns values
 * both at the SAME back, because the later turn's board never had the earlier
 * pick removed, and the plan systematically doubles up. Indexing by how many
 * of that position the plan has already committed fixes it exactly.
 *
 * **Kickers are a legality constraint, not a price.** K and D-ST have zero
 * DIFFERENTIAL value (K1 scores about what K10 does, which is what their flat
 * curve honestly encodes) but you must field one, and VORP cannot see that.
 * So the DP compares (unfillable mandatory slots, value) lexicographically.
 * Timing then falls out with no tuning at all: their gain is 0 at every turn
 * while every other position decays, so the optimum spends its CHEAPEST turns
 * on them, and the cheapest turns are the last.
 */

const PLAN_K = 5;          // availability depth carried per position
const PLAN_PATHS = 48;     // simulated paths the DP is solved over
const MODEL_TURNS = 2;     // turns 1..2 come from the simulation; 3+ from the flow table

/** Overall pick numbers of this seat's remaining turns, starting with now. */
function myTurns(st, seat) {
  const out = [pickNo(st)];
  let after = st.picks.length + 1;
  for (;;) {
    const n = nextPickFor(st, seat, after);
    if (n === null) break;
    out.push(n);
    after = n;
  }
  return out;
}

/** Top-K available VORP at each position, skipping the first `d[p]`. */
function topKAvail(st, d, K, out, off) {
  const B = st.board, nPos = B.positions.length;
  for (let p = 0; p < nPos; p++) {
    const base = off + p * K;
    for (let i = 0; i < K; i++) out[base + i] = -Infinity;
    const arr = B.byPos[p], skip = d ? d[p] : 0;
    let seen = 0;
    for (let i = 0; i < arr.length; i++) {
      const pid = arr[i];
      if (st.taken[pid]) continue;
      if (seen++ < skip) continue;
      const x = B.vorp[pid];
      if (x <= out[base + K - 1]) continue;
      let j = K - 1;
      while (j > 0 && out[base + j - 1] < x) { out[base + j] = out[base + j - 1]; j--; }
      out[base + j] = x;
    }
  }
}

/* Far-horizon availability, from the measured flow table rather than the model.
 * The increment is scaled by (teams-1)/teams because the plan's own future
 * picks are already accounted for by the depth index — the table counts every
 * team's picks, these are the other nine. */
function tailAvail(model, st, turns, from, K) {
  const nPos = st.board.positions.length;
  const n = Math.max(0, turns.length - from);
  const out = new Float64Array(n * nPos * K);
  if (!n || !model.flow) {
    out.fill(-Infinity);
    if (n && !model.flow) for (let i = 0; i < n; i++) topKAvail(st, null, K, out, i * nPos * K);
    return out;
  }
  const total = st.league.teams * st.league.rounds;
  const share = Math.max(0, st.league.teams - 1) / Math.max(1, st.league.teams);
  const now = goneBy(model.flow, st.picks.length, total);
  const d = new Float64Array(nPos);
  for (let i = 0; i < n; i++) {
    const g = goneBy(model.flow, turns[from + i], total);
    for (let p = 0; p < nPos; p++) d[p] = Math.round(Math.max(0, g[p] - now[p]) * share);
    topKAvail(st, d, K, out, i * nPos * K);
  }
  return out;
}

/* The reachable state space and its transitions. Depends only on the league,
 * so it is the same table every render. */
function planSpace(spec) {
  const nPos = spec.nPos, cap = spec.cap;
  const stride = new Int32Array(nPos);
  let size = 1;
  for (let p = 0; p < nPos; p++) { stride[p] = size; size *= cap[p] + 1; }
  const slotOf = new Int32Array(size).fill(-1);
  const keys = [], cnt = [];
  const c = new Int32Array(nPos);
  for (let key = 0; key < size; key++) {
    let used = 0;
    for (let p = 0; p < nPos; p++) {
      c[p] = Math.floor(key / stride[p]) % (cap[p] + 1);
      if (spec.flexOk[p]) used += Math.max(0, c[p] - spec.starters[p]);
    }
    if (used > spec.flex) continue;
    slotOf[key] = keys.length;
    keys.push(key);
    for (let p = 0; p < nPos; p++) cnt.push(c[p]);
  }
  const n = keys.length;
  const cntA = Int32Array.from(cnt);
  const stepTo = new Int32Array(n * nPos).fill(-1);
  const termF = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < nPos; p++) {
      if (cntA[i * nPos + p] < cap[p]) stepTo[i * nPos + p] = slotOf[keys[i] + stride[p]];
      if (spec.mandatory[p]) termF[i] += Math.max(0, spec.starters[p] - cntA[i * nPos + p]);
    }
  }
  return { n, nPos, cnt: cntA, stepTo, termF, slotOf, stride, keys };
}

/** The plan state a real roster is in: dedicated slots filled, then flex. */
function planStateOf(spec, counts) {
  const c = new Int32Array(spec.nPos);
  let left = spec.flex;
  for (let p = 0; p < spec.nPos; p++) c[p] = Math.min(counts[p], spec.starters[p]);
  for (let p = 0; p < spec.nPos && left > 0; p++) {
    if (!spec.flexOk[p]) continue;
    const extra = Math.min(counts[p] - c[p], left);
    if (extra > 0) { c[p] += extra; left -= extra; }
  }
  return c;
}

/* One backward layer. `availAt(p, k)` is the VORP of the (k+1)-th best player
 * at p still expected at this turn, or -Infinity if the position is exhausted
 * (NOT 0 — under this objective 0 means replacement-level, which is exactly
 * the wrong signal for "there is nobody left"). */
function planLayer(space, spec, base, cnt0, availAt) {
  const n = space.n, nPos = space.nPos;
  const v = new Float64Array(n), f = new Int32Array(n), a = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    let bestF = base.f[i], bestV = base.v[i], bestA = -1;   // BENCH
    for (let p = 0; p < nPos; p++) {
      const j = space.stepTo[i * nPos + p];
      if (j < 0) continue;
      const k = Math.max(0, space.cnt[i * nPos + p] - cnt0[p]);
      const x = availAt(p, k);
      if (!(x > -Infinity)) continue;
      const g = space.cnt[i * nPos + p] < spec.starters[p]
        ? Math.max(0, x - spec.empty[p])
        : (spec.flexOk[p] ? Math.max(0, x - spec.emptyFlex) : 0);
      const fj = base.f[j], vj = g + base.v[j];
      if (fj < bestF || (fj === bestF && vj > bestV)) { bestF = fj; bestV = vj; bestA = p; }
    }
    v[i] = bestV; f[i] = bestF; a[i] = bestA;
  }
  return { v, f, a };
}

/* Solve the plan. Layers from the far tail are path-independent and solved
 * once; only the two model-driven turns are re-solved per simulated path.
 *
 * Per path rather than on the means, because max-of-mean <= mean-of-max: a DP
 * fed average availability quietly throws away the option value of a board
 * that might break your way. */
function plan(model, st, seat, sim, spec) {
  const nPos = spec.nPos;
  const turns = myTurns(st, seat);
  const T = turns.length;
  const space = planSpace(spec);
  const counts = st.counts[seat];
  const cnt0 = planStateOf(spec, counts);
  let key = 0;
  for (let p = 0; p < nPos; p++) key += cnt0[p] * space.stride[p];
  const i0 = space.slotOf[key];

  const K = (sim && sim.availK) || PLAN_K;
  const from = MODEL_TURNS + 1;
  const tail = tailAvail(model, st, turns, from, K);
  const at = (buf, off) => (p, k) => (k < K ? buf[off + p * K + k] : -Infinity);

  // shared tail, solved once; `acts[t]` is the plan's intended action at turn t
  const acts = {};
  let base = { v: new Float64Array(space.n), f: space.termF, a: new Int8Array(space.n) };
  for (let t = T - 1; t >= from; t--) {
    base = planLayer(space, spec, base, cnt0, at(tail, (t - from) * nPos * K));
    acts[t] = base.a;
  }

  const paths = (sim && sim.avail && sim.availPaths) ? sim.availPaths : 1;
  const v = new Float64Array(paths * space.n);
  const f = new Int32Array(paths * space.n);
  for (let s = 0; s < paths; s++) {
    let cur = base;
    for (let t = Math.min(T - 1, MODEL_TURNS); t >= 1; t--) {
      const off = sim && sim.avail ? (s * MODEL_TURNS + (t - 1)) * nPos * K : -1;
      cur = planLayer(space, spec, cur, cnt0, off >= 0 ? at(sim.avail, off) : at(tail, 0));
      if (s === 0) acts[t] = cur.a;
    }
    v.set(cur.v, s * space.n);
    f.set(cur.f, s * space.n);
  }
  return { space, spec, turns, T, paths, v, f, i0, cnt0, acts };
}

/** Mean over simulated paths of (value, unfillable mandatory slots) at one plan
 *  state — the value of every turn AFTER this one, given the state you reach. */
function planValueAt(pl, j) {
  let v = 0, f = 0;
  for (let s = 0; s < pl.paths; s++) { v += pl.v[s * pl.space.n + j]; f += pl.f[s * pl.space.n + j]; }
  return { v: v / pl.paths, f: f / pl.paths };
}

/* ------------------------------------------------------------- recommend --- */

/* Value of taking each candidate now = what he adds to the lineup, plus the
 * value of the best plan for every turn after this one given he is on the
 * roster. The turn's opportunity cost is handled automatically: a bench pick
 * and a starter pick both leave the same turns behind, so the two are compared
 * on the same footing.
 *
 * The FIRST term uses the exact cutoff machinery rather than the DP's slot
 * model, because only it knows about displacement — a back who bumps a
 * receiver out of your flex is worth the difference, which a count vector
 * cannot express. Future acquisitions are treated as slot fills; that is the
 * plan's one stated approximation.
 *
 * Ignored, and said out loud in the UI: taking p also removes him from the
 * opponents' choice sets, which slightly changes their picks. bff/vona.py
 * makes the same assumption. */
function recommend(model, st, seat, sim, cands, spec) {
  const B = st.board, nPos = B.positions.length, roster = st.rosters[seat];
  spec = spec || lineupSpec(st.league, B.positions, emptyValues(model, st),
                            blindMask(model, B.positions));
  const cut = cutoffs(B, spec, roster);
  const pl = plan(model, st, seat, sim, spec);
  const at = (j) => planValueAt(pl, j);
  // Value of spending this turn on nobody. A candidate who cannot beat it adds
  // nothing to the lineup you will finish with, whatever the roster looks like,
  // and has to be judged as a BENCH pick instead.
  //
  // Per candidate, not as a mode for the whole turn. "Is every starting slot
  // full" is the wrong test twice over: it misses the long middle stretch where
  // the slots left open are ones the plan means to fill LATER (there, ranking by
  // a flat lineup value returned the candidate list in ADP order and called it a
  // recommendation — five rounds of it), and a turn-wide switch flickers on ties
  // at the turn of the snake, where waiting is often an exact wash.
  const idle = at(pl.i0);
  const ins = insuranceCutoffs(B, spec, roster);
  const w = insuranceWeights(spec, model);

  const z = model.resolutionZ === undefined ? 1.96 : model.resolutionZ;

  const out = [];
  for (const pid of cands) {
    const p = B.pos[pid];
    const now = marginalAt(cut, B.vorp[pid], p);
    const step = pl.space.stepTo[pl.i0 * nPos + p];
    const j = step >= 0 ? step : pl.i0;          // capped position: a bench pick
    const a = at(j);
    const pAvail = sim && sim.index && sim.index.has(pid) ? sim.p[sim.index.get(pid)] : 0;
    // A kicker taken now and a kicker taken at the last turn are worth exactly
    // the same to the finishing lineup, so the DP is INDIFFERENT between them
    // and will spend the first spare turn on one — round 8, in testing. Real
    // drafts are not indifferent: an early kicker burns a bench slot for seven
    // rounds, which is option value the lineup objective cannot see. So among
    // equals, defer. The exception is the whole point of the feasibility term:
    // when taking one NOW is what keeps the slot fillable at all, `need` drops
    // and it sorts first regardless. That alone puts K in round 14 and D/ST in
    // round 15 with no tuning, which is where this league actually takes them.
    const defer = (spec.blind[p] && a.f >= idle.f - 1e-9) ? 1 : 0;
    const ev = now + a.v;
    const row = {
      pid, now, later: a.v, ev, need: a.f, defer, pAvail,
      // Expected lineup points lost by passing: what he adds now, discounted by
      // the chance he is still there at your next turn. Computed but never read
      // until 2026-09-02.
      //
      // It stays OUT of `score`, where it would double-count: `later` already
      // prices availability through the plan's simulated paths. It orders
      // NOTHING -- both attempts were measured and both lost lineup points
      // (-24.5 across a statistical tie, -2.0 across exact ties; see
      // cmpTiered). It is shown on every row instead, which is where it earns
      // its keep: the panel says when a tie cannot be resolved, and this is the
      // number a manager can resolve it with themselves.
      cost: now * (1 - pAvail),
      bench: ev <= idle.v + 1e-6 ? 1 : 0,
      insurance: w[p] * Math.max(0, B.vorp[pid] - ins[p]),
      edge: B.vorp[pid] - B.vorpAdp[pid],
      // Is he worth a roster spot AT ALL: better than the position's free
      // end-of-draft option. See below — this gates `edge`, it is not cosmetic.
      useful: B.vorp[pid] > spec.empty[p] + 1e-9 ? 1 : 0,
      // Is this a position a lost starter would COST you a pick to cover. Only
      // ever a tie-break, and it exists for the endgame — see cmpTiered.
      insurable: w[p] > 0 ? 1 : 0,
    };
    // A bargain on a worthless asset is worthless, and `edge` alone does not
    // know that. It is a difference of two curve slots, so it grows without
    // bound down the steep tail of the QB curve: with the candidate set capped
    // at 8 per position this never showed, but on the full board a QB nobody
    // would roster (VORP -99) scored +56 of "market edge" purely because his
    // expert and market slots sit 8 apart in freefall, and outranked the best
    // back available. So `edge` counts only while the player beats what the
    // position gives away for free at the end of the draft. Below that line
    // `insurance` is 0 by construction (the cutoff is never under the empty
    // floor), so his score is 0 and he sorts where he belongs.
    row.score = row.bench
      ? row.insurance + (row.useful ? row.edge : 0)
      : ev;
    // How much of `score` the curve can resolve.
    //
    // BENCH mode is where this is measurable, and it is measurable exactly:
    // `edge` is a difference of two slots of one position's curve (`edge_se` is
    // its SE, and it is 0 when both slots share a PAVA block), and
    // `insurance`'s uncertain term is the player's own slot estimate. Same kind
    // of object as the score, so the units line up. A clamped term contributes
    // NO uncertainty: `max(0, ...)` at its floor is the constant 0, not an
    // unknown, and the same goes for a candidate whose `edge` is gated off.
    //
    // STARTER mode gets 0, so only EXACTLY equal scores tie, and that is a
    // retreat from a version that was wrong on the page. `ev = now + later` is
    // not a curve difference: `later` is a DP value that MOVES AGAINST `now`
    // (spend the slot early and the rest of the plan is worth less), so the two
    // share curve error that partly cancels and an ev gap is far smaller than
    // the gap between the raw slots behind it. Banding it with the players' own
    // slot SEs mismatches the two and over-ties: on the opening pick it called
    // Ja'Marr Chase (ev 555.1) and Brock Bowers (516.0) indistinguishable at
    // +/-47, in the one region where the curve demonstrably DOES resolve
    // (WR30-39 vs WR50-54 is t 3.92). Propagating curve error through the plan
    // DP is the honest way to band `ev`; until someone does it, the page claims
    // no tie it cannot support. Exact ties are still caught, which is most of
    // what matters here -- this curve's plateaus make them routine.
    const insSe = (w[p] > 0 && B.vorp[pid] > ins[p]) ? w[p] * B.vorpSe[pid] : 0;
    const edgeSe = row.useful ? B.edgeSe[pid] : 0;
    row.scoreSe = row.bench ? Math.hypot(insSe, edgeSe) : 0;
    out.push(row);
  }

  // Sort by value, which also makes each tier's leader its own best member;
  // tier the result for DISPLAY; then let urgency settle exact ties only.
  out.sort(cmp);
  assignTiers(out, z);
  out.sort(cmpTiered);

  // The panel's framing follows the top of the list: if the best available
  // player still improves the lineup, this is a starter pick.
  out.bench = !out.length || out[0].bench === 1;
  out.idle = idle.v;
  out.plan = planPath(pl);
  // How wide the top tie is, for the panel to say so out loud. The band is the
  // WIDEST inside the tie, so the sentence it feeds ("nothing within +/- X of
  // the best is separable") is true of every row shown and not just of one.
  const top = out.length ? out.filter((r) => r.tier === out[0].tier) : [];
  out.tied = top.length;
  out.tieBand = top.reduce((m, r) => Math.max(m, r.tieBand), 0);
  out.z = z;
  return out;
}

/** Two candidates are TIED when the curve cannot separate their scores. */
function tiedScores(a, b, z) {
  return Math.abs(a.score - b.score) <= z * Math.hypot(a.scoreSe, b.scoreSe) + 1e-9;
}

/* Group a value-sorted list into tiers of picks the curve cannot order.
 *
 * Why this exists at all: the VORP curve resolves the TOP of the board and not
 * the back of it. On the 2012-2025 history the gap between the best and worst
 * available back at a round-8 pick measures +19.1 +/- 10.6 points, t 1.81 — so
 * a panel that ranked them 1 through 8 was ranking noise, and did it with a
 * decisive-looking `+15.5` attached. vorp.py's docstring has the numbers.
 *
 * A tier is "not distinguishable from the tier's LEADER", not "not
 * distinguishable from my predecessor". Pairwise ties are not transitive, so
 * chaining them would let a long list of near-neighbours pool two candidates
 * the curve separates cleanly. Comparing to the leader is transitive by
 * construction and is the standard "significantly worse than the best"
 * grouping.
 *
 * Tiers never span a change of `need`, `defer` or `bench`: those rows are
 * answers to different questions and a tie between them would be meaningless.
 *
 * This is a DISPLAY grouping, and only that. It sets how much the panel may
 * claim; it does not reorder anything (see cmpTiered for what happened when it
 * did). A tie means "we cannot prove the first is better", not "take either".
 */
function assignTiers(rows, z) {
  let tier = 0, lead = null;
  for (const r of rows) {
    if (lead === null) {
      lead = r;
    } else if (r.need !== lead.need || r.defer !== lead.defer
               || r.bench !== lead.bench || !tiedScores(lead, r, z)) {
      tier += 1;
      lead = r;
    }
    r.tier = tier;
    // How far from its tier's leader this row is allowed to sit. For the leader
    // itself that is just its own precision: hypot(se, se) would report a band
    // 1.41x too wide for the number the panel prints as "the tie is +/- X".
    r.tieBand = z * (lead === r ? r.scoreSe : Math.hypot(lead.scoreSe, r.scoreSe));
  }
}

/* Rank order, and every term is doing work:
 *   need    a pick that keeps a mandatory slot fillable outranks everything
 *   defer   among equals, kickers and defenses go LAST (see above)
 *   bench   a pick that improves the lineup outranks one that cannot
 *   score   lineup EV for the first group, insurance + market edge for the
 *           second — two different questions, never mixed in one number
 */
/* Scores are fantasy points, so a difference below this is float dust and must
 * not order anything. It is not a stylistic tolerance: in the endgame every
 * candidate's score is `vorp - ins[p]` with `vorp == ins[p]`, which lands at
 * ~1e-10 rather than at 0, and a bare `b.score - a.score` handed the whole
 * ordering of the last three rounds to that residue -- ranking a 148th-ADP back
 * over a 127th-ADP receiver on 7e-11 of a point. Quantising here is what lets
 * cmpTiered's roster-shape term see those rows as the tie they are. */
const SCORE_EPS = 1e-9;

function cmp(a, b) {
  const ds = b.score - a.score;
  return (a.need - b.need) || (a.defer - b.defer) || (a.bench - b.bench)
      || (Math.abs(ds) > SCORE_EPS ? ds : 0);
}

/* The order the panel actually shows: `cmp`, plus one roster-shape tie-break
 * for the endgame. Two things were tried here and MEASURED, and the record
 * belongs next to the code because both are tempting.
 *
 * **Ordering a statistical TIE by urgency: rejected, -24.5 points.** `cost` is
 * the one number on the row that is not a difference of two isotonic block
 * means, so the first design ordered the whole 1.96-SE band by what you lose by
 * waiting. On tests/mockdraft.mjs (2025 board, ten seats, deterministic
 * opponents) that cost 24.5 mean lineup points, 528.2 -> 503.7, and dropped
 * seats beating best-available from 10/10 to 4/10. The lesson generalises:
 * STATISTICAL INDISTINGUISHABILITY IS NOT INDIFFERENCE. Inside the band the
 * point estimate is still the best estimate available, and swapping in another
 * criterion throws away real expected value. A tie bounds what the panel may
 * CLAIM; it is not a licence to reorder.
 *
 * **Ordering EXACT ties by urgency: not adopted, because it measures as a coin
 * flip.** Restricted to scores equal to the last digit this is the case the
 * plateaus make routine — at a round-3 pick four backs can score identically —
 * so it looked like a free improvement. Across four board snapshots its delta
 * against shipping order is +0.0 (2025 fixture, twice), -2.0 (2026 board as of
 * 2026-09-02 11:00) and +0.3 (2026 board as of 12:27, after the market
 * refreshed under us mid-session). Sign-unstable and board-dependent: no
 * effect.
 *
 * That is what should be expected, and the reason is worth stating rather than
 * averaging away: equal `ev` does not mean equal OUTCOME. `ev` is the plan's
 * estimate, and the plan values future acquisitions as slot fills, so two picks
 * it cannot separate still send the rest of the draft somewhere different --
 * and which of the two is better is not something this estimator knows. So
 * `cost` is DISPLAYED on every row and ordered on by nobody; a manager with a
 * bye chart and an injury hunch settles it better than a coin flip dressed as a
 * ranking. (Caveat, since it argues the other way: mockdraft's opponents are
 * deterministic argmax while `pAvail` comes from the stochastic simulation, so
 * this harness is a weak test of urgency specifically. It is the harness there
 * is, and it does not support the change.)
 *
 * **The endgame term: kept, free, and it closes a hole the report had open.**
 * By round 12 every remaining player is below his position's free floor, so
 * every candidate scores exactly 0; the comparator fell through to array order,
 * which is ADP order, and handed back a THIRD quarterback — 2.9 per seat on the
 * 2026 board against a real-manager mean of 1.4. That is the same failure the
 * report describes at round 7 in an earlier engine ("the sort ran over an
 * all-zero array, so V8's stable sort handed back the input order") surviving
 * in the one stretch where all-zero is the honest answer rather than a bug. The
 * board it showed worst on took 2.0 quarterbacks AND 2.7 tight ends per seat,
 * against real-manager means of 1.4 and 1.4.
 *
 * There is no upside model to break it with — deliberately, this data has an
 * isotonic mean curve and no per-player distribution — so it breaks on roster
 * shape, which needs no new number. Among picks all worth nothing, prefer a
 * position where losing a starter would COST you a pick to cover; at a
 * streamable position it would not, which is what streamable means, so a spare
 * back or receiver beats a third quarterback as a lottery ticket. It cannot
 * move lineup value by construction (everything here is below its floor, which
 * is why the scores are 0) and it does not. Measured 2026-09-02 on the live
 * board and on the 2025 fixture, ten seats each: this term together with the
 * `useful` gate and the streamable insurance weight leaves lineup value
 * UNCHANGED (544.9 and 528.2, 10/10 seats both) while moving the position mix
 * from QB 2.0 / RB 3.6 / WR 4.7 / TE 2.7 to QB 1.0 / RB 5.8 / WR 5.2 / TE 1.0,
 * against real-manager means of 1.4 / 4.9 / 5.6 / 1.4. */
function cmpTiered(a, b) {
  return cmp(a, b) || (b.insurable - a.insurable);
}

/* For a candidate who cannot improve the lineup he would finish with, the
 * lineup objective is flat and ranking on it would be noise, so change the
 * question: what is this player worth if a starter is lost, plus how far the
 * experts have him above his market price.
 *
 * Note what the condition is NOT. `bench` does not mean "every starting slot is
 * filled" — a slot the plan intends to fill at a LATER turn is open right now
 * and still contributes nothing to taking this player today, which is the
 * common case in the middle rounds. The UI said "starters covered" for years
 * and was simply wrong whenever a plan-filled slot was open; see page.js.
 *
 * Deliberately NOT an upside model. There is no per-player distribution in
 * this data, only an isotonic mean curve, so anything more elaborate would be
 * fake precision dressed up as analysis. */
function insuranceCutoffs(board, spec, roster) {
  const out = new Float64Array(spec.nPos);
  for (let p = 0; p < spec.nPos; p++) {
    let worst = -1, wv = Infinity;
    for (const pid of roster) {
      if (board.pos[pid] === p && board.vorp[pid] < wv) { wv = board.vorp[pid]; worst = pid; }
    }
    const thin = worst < 0 ? roster : roster.filter((q) => q !== worst);
    out[p] = cutoffs(board, spec, thin)[p];
  }
  return out;
}

/* Which positions you can usefully buy insurance AT. Two get nothing.
 *
 * **No curve (K, D/ST).** A backup kicker is worth what the starter is worth,
 * which is zero.
 *
 * **Streamable (QB, TE).** This is the substantive one, and it needs no new
 * number: `STREAMABLE` in vorp.py means the free in-season option at that
 * position IS replacement level — you fill a punted slot off waivers each week,
 * and `REPL_RANKS` is set to that streaming total rather than to a roster
 * depth. So the fallback when your quarterback is lost is available in-season
 * for nothing, and a draft pick spent on a backup buys you what waivers would
 * have given you. Insurance at a streamable position is ~0 by the same
 * definition that sets its replacement level.
 *
 * Leaving it at 1 is what the first version of this did, and it was measurably
 * wrong in exactly the way the report already complained about: at 1 the term
 * prices a second quarterback as `vorp - ins[QB]`, and with ONE quarterback
 * rostered `insuranceCutoffs` drops him, leaves the slot empty, and hands the
 * backup his FULL VORP -- i.e. it prices the starter's loss as certain, at a
 * position you would stream anyway. On the live 2026 board that walked ten
 * seats into 2.6 quarterbacks each against a real-manager mean of 1.4.
 *
 * What this REPLACED was worse, and worth recording so nobody restores it. The
 * weight was `|empty[p]| / max |empty|`, described as "how much a lost starter
 * costs, normalised — which is the EMPTY floor again, so no new number is
 * invented here". That number was not invented, it was CIRCULAR: `empty[p]` is
 * the VORP of the best undrafted player at p, and VORP is defined relative to
 * replacement, which vorp.py sets to that same best undrafted player. So
 * `empty[p]` is ~0 by construction, its magnitude is residual curve-shape
 * noise, and dividing noise by noise collapsed the whole term. Observed on a
 * real round-8 board in this 10-team league: insurance read 0.0 for every
 * candidate at every position, so the bench ranking was market edge alone
 * while the UI claimed it was both. */
function insuranceWeights(spec, model) {
  const stream = (model && model.streamable) || [];
  const w = new Float64Array(spec.nPos);
  for (let p = 0; p < spec.nPos; p++) {
    w[p] = (spec.blind[p] || stream.indexOf(spec.positions[p]) >= 0) ? 0 : 1;
  }
  return w;
}

/** The plan the DP intends, turn by turn — the shape of the rest of your
 *  draft, which is the part of the answer a single recommendation hides.
 *  Turns 1-2 come from the first simulated path; the rest from the flow table,
 *  and the panel greys them accordingly. Pass `from` to read the plan that
 *  follows a hypothetical pick, which is what the player card shows. */
function planPath(pl, from) {
  const out = [];
  let i = from === undefined || from < 0 ? pl.i0 : from;
  for (let t = 1; t < pl.T; t++) {
    const a = pl.acts[t];
    if (!a) break;
    const q = a[i];
    out.push({ turn: pl.turns[t], pos: q, modelled: t <= MODEL_TURNS });
    if (q < 0) continue;                       // bench: the state does not move
    const j = pl.space.stepTo[i * pl.space.nPos + q];
    if (j < 0) break;
    i = j;
  }
  return out;
}

/** Cap-aware candidate set: every available player at every position the seat
 *  may still legally draft. Replaces slicing the ADP watch list, which ignored
 *  the position caps entirely and would happily recommend a sixth tight end.
 *
 *  `perPos` defaults to the WHOLE position, and the default is the fix for a
 *  bug that survived a long time because both halves looked reasonable alone.
 *  `available` walks `byPos`, which is in ADP order, so a `perPos` cap takes the
 *  N shallowest-ADP players at each position — while `recommend` ranks bench
 *  picks by MARKET EDGE, i.e. by how far the experts have a player ABOVE his
 *  ADP. A big edge therefore implies a deep ADP, so the filter removed exactly
 *  the players the score was looking for. tests/ties.mjs measures this rather
 *  than asserting it: walking a whole draft on either the live 2026 board or
 *  the 2025 fixture, the old `perPos: 8` slice hid the single biggest market
 *  edge on the board in 8 of the 12 states where the seat was on the clock.
 *  (Do not re-cite the player who prompted this -- the 2026 board is a rolling
 *  window and the market closed that particular gap inside a day.)
 *
 *  Scoring the full board costs nothing: `recommend` is O(1) per candidate once
 *  the plan and cutoffs are built, and the whole legal board is a few hundred
 *  rows. Pass `perPos` only where a fixed-width set is the point (a strategy
 *  baseline in tests/mockdraft.mjs). */
function candidates(st, seat, perPos) {
  const k = perPos === undefined || perPos === null ? Infinity : perPos;
  const out = [];
  for (const p of openPositions(st, seat)) {
    for (const pid of available(st, p, k)) out.push(pid);
  }
  out.sort((a, b) => st.board.adpRank[a] - st.board.adpRank[b]);
  return out;
}

const M32 = 0xFFFFFFFF;
function imul(a, b) { return Math.imul(a, b) >>> 0; }

function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = imul(t ^ (t >>> 15), 1 | t);
    t = ((t + imul(t ^ (t >>> 7), 61 | t)) >>> 0) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash32(text) {
  let h = 0x811C9DC5;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i) & 0xFF; h = imul(h, 0x01000193); }
  return h >>> 0;
}

function stateSeed(st, base) {
  return ((base >>> 0) ^ hash32(`${st.board.season}|${st.league.teams}|${st.picks.join(',')}`)) >>> 0;
}

function samplePick(rnd, probs) {
  const u = rnd();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u < acc) return i; }
  return probs.length - 1;
}

/* ------------------------------------------------------------ simulation --- */

/** P(each available player survives to `seat`'s next turn) — and to the turn
 *  after that — plus the top of the board at each of those turns, per
 *  simulated path.
 *
 *  `p2` costs nothing extra: leg two already runs for the plan's paths, so the
 *  second horizon is a counter, not a second simulation. It is coarser and the
 *  page says so — it is averaged over `p2Paths` (tens to low hundreds) rather
 *  than every sim, it recalibrates with the curve fitted at ONE turn, which
 *  over two turns under-corrects, and it conditions on the seat taking nobody
 *  in between (leg two advances the seat's own pick as `null`). All three
 *  errors push the same way, so read `p2` as an optimistic bound.
 *
 *  The second leg is run for the first PLAN_PATHS paths only, CONTINUING the
 *  same random stream rather than re-seeding: leg one's draws stay bit-identical
 *  so the survival probabilities — which are calibrated, and the number the page
 *  exists to show — do not move at all when the plan is switched on. Recording
 *  the top K rather than the single best is what stops the plan from valuing two
 *  consecutive picks at the same player.
 *
 *  Between the two legs the seat's own pick is advanced as `null`: the player is
 *  not removed, so leg two's board is a couple of players over-available. That
 *  is exactly what the plan's depth index absorbs, and it keeps the seat's own
 *  roster out of the opponents' features, where it does not belong. */
function simulate(model, st, seat, opts) {
  opts = opts || {};
  const nSims = opts.nSims || 2000;
  const chain = model.autoChain;
  const override = opts.autoOverride || {};
  const h = horizon(st, seat);
  const watchTop = opts.watchTop || 60;
  const K = opts.availK || PLAN_K;
  const planOn = opts.plan !== false;
  const turns = myTurns(st, seat);
  const nPaths = planOn ? Math.min(nSims, opts.planPaths || PLAN_PATHS) : 0;
  // The players a drafter could plausibly be waiting on: the top of the
  // available board. Mirrors simulate.py — watching everyone makes the
  // calibration metric meaningless and the panel unreadable.
  let watch = [];
  for (let p = 0; p < st.board.positions.length; p++) {
    for (const pid of available(st, p, watchTop)) watch.push(pid);
  }
  watch.sort((a, b) => st.board.adpRank[a] - st.board.adpRank[b]);
  watch = watch.slice(0, watchTop);
  const nPos = st.board.positions.length;
  if (!watch.length) {
    const avail = new Float64Array(Math.max(1, nPaths) * MODEL_TURNS * nPos * K);
    avail.fill(-Infinity);
    return {
      pids: [], p: [], p2: null, p2Paths: 0, horizon: h, nSims, turns,
      avail, availPaths: Math.max(1, nPaths), availK: K, index: new Map(),
    };
  }
  const index = new Map(watch.map((pid, i) => [pid, i]));
  const survived = new Int32Array(watch.length);
  const survived2 = new Int32Array(watch.length);
  let n2 = 0;
  // Back-to-back at the turn of the snake: nothing happens before the next
  // turn, so survival there is exactly 1 and the first leg has no work to do.
  // The paths still run, because the turn AFTER that is ~18 picks away and is
  // precisely where "can I wait on him" is hardest to answer by eye.
  const nRun = h > 0 ? nSims : nPaths;
  // Top-K surviving VORP per position at each of the next two turns, per path.
  const avail = new Float64Array(Math.max(1, nPaths) * MODEL_TURNS * nPos * K);
  avail.fill(-Infinity);
  const base = opts.seed !== undefined ? opts.seed : stateSeed(st, 20482930);

  const belief = [];
  for (let s = 0; s < st.league.teams; s++) {
    if (override[s] !== undefined) belief.push(override[s]);
    else if (chain) belief.push(chain.p_start[st.managers[s]] !== undefined
      ? chain.p_start[st.managers[s]] : chain.league_start);
    else belief.push(0);
  }

  const oneStep = (sim_st, autoState, rnd) => {
    const s = seatOnClock(sim_st);
    if (s < 0) return false;
    const mgr = sim_st.managers[s];
    if (chain && override[s] === undefined) {
      const pAuto = autoState[s] ? chain.p_a_to_a
        : (chain.p_h_to_a[mgr] !== undefined ? chain.p_h_to_a[mgr] : chain.league_h_to_a);
      autoState[s] = rnd() < pAuto;
    }
    const r = jointProbs(model, sim_st, s, mgr, autoState[s] ? 1 : 0, opts.kappa);
    const total = r.probs.reduce((a, b) => a + b, 0) + r.outside;
    if (total <= 0) return false;
    const all = r.probs.map((x) => x / total).concat([r.outside / total]);
    const j = samplePick(rnd, all);
    advance(sim_st, j >= r.pids.length ? null : r.pids[j]);
    return true;
  };

  const leg2 = turns.length > 2 ? turns[2] - turns[1] - 1 : 0;
  for (let sim = 0; sim < nRun; sim++) {
    const rnd = mulberry32((base + sim * 2654435761) >>> 0);
    const sim_st = forkState(st);
    const autoState = belief.map((b) => rnd() < b);
    for (let step = 0; step < h; step++) if (!oneStep(sim_st, autoState, rnd)) break;
    // survival is counted BEFORE leg two touches the state, and leg two only
    // draws from this path's own stream, so `p` is unchanged by the plan
    for (let i = 0; i < watch.length; i++) if (!sim_st.taken[watch[i]]) survived[i]++;
    if (sim < nPaths) {
      topKAvail(sim_st, null, K, avail, sim * MODEL_TURNS * nPos * K);
      if (turns.length > 2) {
        advance(sim_st, null);
        for (let step = 0; step < leg2; step++) if (!oneStep(sim_st, autoState, rnd)) break;
        for (let i = 0; i < watch.length; i++) if (!sim_st.taken[watch[i]]) survived2[i]++;
        n2++;
      }
      topKAvail(sim_st, null, K, avail, (sim * MODEL_TURNS + 1) * nPos * K);
    }
  }
  const raw = h > 0 ? Array.from(survived, (c) => c / nSims) : watch.map(() => 1);
  // Same recalibration as simulate.py: the raw simulation over-predicts
  // survival because independent draws under-produce positional runs.
  const rc = model.recal || [0, 1];
  const cal = (v) => {
    const q = Math.min(Math.max(v, 1e-4), 1 - 1e-4);
    const z = rc[0] + rc[1] * Math.log(q / (1 - q));
    return 1 / (1 + Math.exp(-z));
  };
  const on = opts.calibrated !== false;
  const p = (on && h > 0) ? raw.map(cal) : raw;
  const raw2 = n2 ? Array.from(survived2, (c) => c / n2) : null;
  return {
    pids: watch, p, raw, horizon: h, nSims, index,
    p2: raw2 && on ? raw2.map(cal) : raw2, raw2, p2Paths: n2,
    turns, avail, availPaths: nPaths, availK: K,
  };
}

/* --------------------------------------------------------- fast-forward ---
 * Skip-ahead for the page: the model makes real picks until a target seat is
 * on the clock or a target pick number is reached. Deliberately PARALLEL to
 * simulate()'s inner step rather than a refactor of it: tests/simstable.mjs
 * asserts simulate's RNG stream bit-identical, so its internals are frozen,
 * and these functions never draw from its stream. Two other differences are
 * intentional: the caller supplies a continuous autoWeight per seat (the same
 * mixture renderUpcoming shows) instead of sampling the autodraft Markov
 * chain, and an "outside the window" draw is renormalised away rather than
 * advanced as null — a null here would land in ui.picks, indistinguishable
 * from a genuine off-board pick, and leave the pool over-available for every
 * later simmed pick. */

const FF_SEED = 0x74656E64;   // deterministic default; the page passes its own

/** One model pick for the seat on the clock. Returns a pid, or null when no
 *  open position has a candidate. Never advances the state.
 *  opts: { autoWeight, sample, rnd, kappa } */
function botPick(model, st, opts) {
  opts = opts || {};
  const s = seatOnClock(st);
  if (s < 0) return null;
  const r = jointProbs(model, st, s, st.managers[s], opts.autoWeight || 0, opts.kappa);
  if (!r.pids.length) return null;
  if (!opts.sample) {          // argmax — same rule mockdraft.mjs plays drafts by
    let best = 0;
    for (let i = 1; i < r.probs.length; i++) if (r.probs[i] > r.probs[best]) best = i;
    return r.pids[best];
  }
  let total = 0;
  for (const q of r.probs) total += q;
  if (total <= 0) return r.pids[0];
  const j = samplePick(opts.rnd, r.probs.map((q) => q / total));
  return r.pids[j];
}

/** Advance `st` with model picks until `stopSeat` is on the clock, pickNo
 *  reaches `stopPick` (that pick is left ON the clock, unmade), or the draft
 *  ends. Mutates st. Returns the pids appended, in order, so the caller can
 *  record their provenance. Sampled runs reseed per pick from the state
 *  (mulberry32(stateSeed(st, seedBase))), so the result is a pure function of
 *  (state, seedBase).
 *  opts: { stopSeat, stopPick, sample, seedBase,
 *          autoWeight: (seat) => w in [0,1] (default () => 0), kappa } */
function fastForward(model, st, opts) {
  opts = opts || {};
  const awOf = opts.autoWeight || (() => 0);
  const seedBase = opts.seedBase === undefined ? FF_SEED : (opts.seedBase >>> 0);
  const out = [];
  const maxSteps = st.seats.length - st.picks.length;
  for (let i = 0; i < maxSteps; i++) {
    const s = seatOnClock(st);
    if (s < 0) break;
    if (opts.stopSeat !== undefined && opts.stopSeat !== null && s === opts.stopSeat) break;
    if (opts.stopPick !== undefined && opts.stopPick !== null && pickNo(st) >= opts.stopPick) break;
    const rnd = opts.sample ? mulberry32(stateSeed(st, seedBase)) : null;
    let pid = botPick(model, st, { autoWeight: awOf(s), sample: !!opts.sample, rnd, kappa: opts.kappa });
    if (pid !== null && st.taken[pid]) pid = null;   // belt-and-braces; never corrupt state
    advance(st, pid);
    out.push(pid);
  }
  return out;
}

const API = {
  POS_FEATURES, PLR_FEATURES, BOARD_FIELDS, assertContract, assertBoard,
  makeBoard, newState, snakeSeats,
  pickNo, seatOnClock, nextPickFor, horizon, available, openPositions, choiceSet,
  advance, undo, forkState, positionFeatures, playerFeatures, positionProbs,
  jointProbs, scoreState, playerUtilities, simulate, mulberry32, hash32, stateSeed,
  botPick, fastForward,
  logSumExp, softmax, lineupValue, lineupValueWith, lineupSlots, marginal,
  marginalAt, cutoffs,
  lineupSpec, emptyValues, blindMask, goneBy, openSlots, myTurns, topKAvail,
  planSpace, planStateOf, plan, planPath, planValueAt, recommend, candidates,
  insuranceCutoffs, insuranceWeights, assignTiers, tiedScores,
  PLAN_K, PLAN_PATHS, MODEL_TURNS, NO_SE,
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof window !== 'undefined') window.TendiesEngine = API;
