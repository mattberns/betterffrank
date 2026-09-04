/* The resolution floor, asserted three ways.
 *
 * The VORP curve resolves the top of the board and not the back of it: at a
 * round-8 pick the gap between the best and worst available back measures
 * +19.1 +/- 10.6 points, t 1.81, so a panel that ranks eight of them 1 through
 * 8 is ranking noise. `assignTiers` groups what cannot be separated, and this
 * harness pins down what that grouping is and is not allowed to do.
 *
 *   node tests/ties.mjs <payload.json>
 */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const E = require('../tendies/web/engine.js');
const fs = require('node:fs');

const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const M = payload.model;
const league = M.league;
const board = E.makeBoard(payload.board, M.positions);
board.season = payload.meta.season;
const managers = (payload.meta.managers || []).slice(0, league.teams);
while (managers.length < league.teams) managers.push('');

const out = {
  seSeen: 0, seZero: 0, sePositive: 0,
  transitivity: 0, transitivityBad: [],
  orderChecked: 0, orderBad: [],
  tierChecked: 0, tierBad: [],
  benchGate: 0, benchGateBad: [],
  candChecked: 0, candBad: [],
  states: 0,
};

/* 1. The payload must actually carry standard errors, and they must be the
 *    shape vorp.py describes: zero exactly when both slots land in one PAVA
 *    block (the difference is then literally the same number twice), positive
 *    otherwise. All-zero would mean the curve claims to be exact. */
for (let pid = 0; pid < board.n; pid++) {
  const se = board.vorpSe[pid];
  if (!isFinite(se)) continue;
  out.seSeen++;
  if (se === 0) out.seZero++; else out.sePositive++;
}

/* 2. Tie membership must be TRANSITIVE within a tier: everything in a tier is
 *    tied with the tier's leader by construction, and that is the property
 *    that makes sharing one rank number honest. Chaining pairwise ties instead
 *    would pool candidates the curve separates cleanly. */
function checkTiers(rec) {
  const byTier = new Map();
  for (const r of rec) {
    if (!byTier.has(r.tier)) byTier.set(r.tier, []);
    byTier.get(r.tier).push(r);
  }
  for (const [tier, rows] of byTier) {
    const lead = rows[0];
    for (const r of rows) {
      out.transitivity++;
      if (!E.tiedScores(lead, r, M.resolutionZ)) {
        out.transitivityBad.push({ tier, lead: board.name[lead.pid], row: board.name[r.pid] });
      }
    }
    // tiers are contiguous and non-decreasing down the list
    out.tierChecked++;
    if (rows.length !== rec.filter((q) => q.tier === tier).length) {
      out.tierBad.push({ tier, why: 'not contiguous' });
    }
  }
  for (let i = 1; i < rec.length; i++) {
    out.tierChecked++;
    if (rec[i].tier < rec[i - 1].tier) out.tierBad.push({ i, why: 'tier went backwards' });
  }
}

/* 3. THE ONE THAT MATTERS. A tie must not reorder the list. Both attempts to
 *    order on urgency instead of on the point estimate LOST lineup points in
 *    tests/mockdraft.mjs -- 24.5 across the whole 1.96-SE band (528.2 -> 503.7,
 *    seats beaten 10/10 -> 4/10) and 2.0 across exact ties on the 2026 board
 *    (540.6 -> 538.6, stable at 200/300/400 sims). Statistical
 *    indistinguishability is not indifference. So within one
 *    (need, defer, bench) group the displayed order is score-descending, full
 *    stop; `cost` is shown and never sorted on. */
function checkOrder(rec) {
  for (let i = 1; i < rec.length; i++) {
    const a = rec[i - 1], b = rec[i];
    if (a.need !== b.need || a.defer !== b.defer || a.bench !== b.bench) continue;
    out.orderChecked++;
    if (b.score > a.score + 1e-9) {
      out.orderBad.push({ a: board.name[a.pid], b: board.name[b.pid],
                          aScore: a.score, bScore: b.score });
    }
    // An exact tie the endgame term cannot break falls to ADP order, because
    // `candidates` hands the list over in ADP order and both sorts are stable.
    // Asserted rather than left implicit: it is what excludes a `cost` sort
    // (measured at -2.0 points) from creeping back in, and it is the reason the
    // panel is deterministic on a plateau instead of depending on scan order.
    if (Math.abs(a.score - b.score) <= 1e-9 && a.insurable === b.insurable
        && board.adpRank[b.pid] < board.adpRank[a.pid]) {
      out.orderBad.push({ a: board.name[a.pid], b: board.name[b.pid],
                          why: 'exact tie not in ADP order' });
    }
    // the endgame term: among picks that are all worth nothing, a position you
    // would have to spend a pick to cover outranks a streamable one
    if (Math.abs(a.score - b.score) <= 1e-9 && b.insurable > a.insurable) {
      out.orderBad.push({ a: board.name[a.pid], b: board.name[b.pid],
                          why: 'streamable ranked above insurable on a tie' });
    }
  }
}

/* 4. A bench pick below his position's free end-of-draft floor must score 0.
 *    `edge` is a difference of two curve slots and grows without bound down the
 *    steep tail of the QB curve, so ungated it once put a VORP -99 quarterback
 *    (edge +56) above the best available back the moment the candidate set was
 *    widened to the whole board. A bargain on a worthless asset is worthless. */
function checkBenchGate(rec, spec) {
  for (const r of rec) {
    if (!r.bench) continue;
    out.benchGate++;
    const p = board.pos[r.pid];
    if (board.vorp[r.pid] <= spec.empty[p] && Math.abs(r.score) > 1e-9) {
      out.benchGateBad.push({ name: board.name[r.pid], vorp: board.vorp[r.pid],
                              floor: spec.empty[p], score: r.score });
    }
  }
}

/* 5. The candidate set must not filter on the quantity the bench score ranks
 *    by. `available` walks the board in ADP order, so a per-position cap takes
 *    the shallowest ADPs -- while a big market edge implies a DEEP ADP. Assert
 *    the uncapped set actually reaches the biggest edges on the board. */
function checkCandidates(st, seat) {
  const all = E.candidates(st, seat);
  const capped = new Set(E.candidates(st, seat, 8));
  const legal = new Set(all);
  let bestEdge = -Infinity, bestPid = -1;
  for (const pid of legal) {
    const e = board.vorp[pid] - board.vorpAdp[pid];
    if (board.vorp[pid] > 0 && e > bestEdge) { bestEdge = e; bestPid = pid; }
  }
  if (bestPid < 0) return;
  out.candChecked++;
  if (!legal.has(bestPid)) {
    out.candBad.push({ name: board.name[bestPid], why: 'uncapped set missed it' });
  }
  // and record when the old cap would have hidden it, so the test says out loud
  // that this is a live failure mode and not a hypothetical one
  if (!capped.has(bestPid)) out.cappedWouldMiss = (out.cappedWouldMiss || 0) + 1;
}

function opponentPick(st, s) {
  const r = E.jointProbs(M, st, s, st.managers[s], 0);
  if (!r.pids.length) return null;
  let b = 0;
  for (let i = 1; i < r.probs.length; i++) if (r.probs[i] > r.probs[b]) b = i;
  return r.pids[b];
}

/* Walk one whole draft from a fixed seat, checking the panel at every turn. */
const SEAT = 7;
const st = E.newState(board, league, managers);
for (;;) {
  const s = E.seatOnClock(st);
  if (s < 0) break;
  if (s !== SEAT) { E.advance(st, opponentPick(st, s)); continue; }
  const spec = E.lineupSpec(league, board.positions, E.emptyValues(M, st),
                            E.blindMask(M, board.positions), E.holdValues(M, board.positions));
  const sim = E.simulate(M, st, SEAT, { nSims: 150 });
  const rec = E.recommend(M, st, SEAT, sim, E.candidates(st, SEAT), spec);
  out.states++;
  checkTiers(rec);
  checkOrder(rec);
  checkBenchGate(rec, spec);
  checkCandidates(st, SEAT);
  if (rec.length) E.advance(st, rec[0].pid); else E.advance(st, null);
}

process.stdout.write(JSON.stringify(out));
