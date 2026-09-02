/* The headline test: play whole drafts taking the top recommendation every
 * time, and check the rosters are ones a person would actually field.
 *
 * This is the test that would have caught the failure that prompted the
 * rewrite. The old engine, run this way, produced five quarterbacks, one
 * running back, six tight ends against a cap of three, and no kicker or
 * defense at all — while all 70 real team-seasons in this league filled every
 * starting slot. No unit test noticed, because every individual number the
 * engine computed was internally consistent; only playing it out shows it.
 *
 * Opponents pick the model's argmax, which is deterministic, so a failure is
 * exactly reproducible.
 *
 *   node tests/mockdraft.mjs <payload.json> [nSims] [--verbose]
 */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const E = require('../tendies/web/engine.js');
const fs = require('node:fs');

const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const NSIMS = Number(process.argv[3] || 200);
const VERBOSE = process.argv.includes('--verbose');

const M = payload.model;
const league = M.league;
const board = E.makeBoard(payload.board, M.positions);
board.season = payload.meta.season;
const managers = (payload.meta.managers || []).slice(0, league.teams);
while (managers.length < league.teams) managers.push('');
const P = M.positions;

function opponentPick(st, s) {
  const r = E.jointProbs(M, st, s, st.managers[s], 0);
  if (!r.pids.length) return null;
  let best = 0;
  for (let i = 1; i < r.probs.length; i++) if (r.probs[i] > r.probs[best]) best = i;
  return r.pids[best];
}

/* The two rules a recommendation has to beat to be worth running: take the
 * best player left, by the market's price or by ours. Both are given the
 * cap-aware candidate set, so this is a fair fight — the old engine's sixth
 * tight end is not the bar. */
const STRATEGY = {
  model: (st, seat, sim) => E.recommend(M, st, seat, sim, E.candidates(st, seat, 8)),
  vorp: (st, seat) => E.candidates(st, seat, 8)
    .map((pid) => ({ pid })).sort((a, b) => board.vorp[b.pid] - board.vorp[a.pid]),
  adp: (st, seat) => E.candidates(st, seat, 8)
    .map((pid) => ({ pid })).sort((a, b) => board.adpRank[a.pid] - board.adpRank[b.pid]),
};

function playSeat(seat, how) {
  const st = E.newState(board, league, managers);
  const log = [];
  for (;;) {
    const s = E.seatOnClock(st);
    if (s < 0) break;
    if (s !== seat) { E.advance(st, opponentPick(st, s)); continue; }
    let sim = null;
    if (how === 'model') {
      sim = E.simulate(M, st, seat, { nSims: NSIMS });
      sim.index = new Map(sim.pids.map((pid, i) => [pid, i]));
    }
    const rec = STRATEGY[how](st, seat, sim);
    if (!rec.length) { E.advance(st, null); continue; }
    const pick = rec[0];
    log.push({
      round: Math.floor((E.pickNo(st) - 1) / league.teams) + 1,
      pos: P[board.pos[pick.pid]], name: board.name[pick.pid],
      now: pick.now || 0, ev: pick.ev || 0, bench: !!rec.bench,
    });
    E.advance(st, pick.pid);
  }
  return { roster: st.rosters[seat], log, st };
}

const spec = E.lineupSpec(league, P, E.emptyValues(M, E.newState(board, league, managers)),
                          E.blindMask(M, P));

const problems = [];
const rows = [];
let beat = 0;
for (let seat = 0; seat < league.teams; seat++) {
  const { roster, log } = playSeat(seat, 'model');
  const counts = {};
  for (const p of P) counts[p] = 0;
  for (const pid of roster) counts[P[board.pos[pid]]] += 1;

  const score = E.lineupValue(board, spec, roster);
  const base = {};
  for (const how of ['vorp', 'adp']) {
    base[how] = E.lineupValue(board, spec, playSeat(seat, how).roster);
  }
  if (score > base.vorp && score > base.adp) beat++;

  const open = E.openSlots(board, spec, roster);
  const fail = [];
  for (const p of P) {
    const need = league.starters[p] || 0;
    if (counts[p] < need) fail.push(`${p} ${counts[p]}<${need}`);
    if (counts[p] > (league.limits[p] === undefined ? 99 : league.limits[p])) {
      fail.push(`${p} ${counts[p]} over cap ${league.limits[p]}`);
    }
  }
  if (open > 0) fail.push(`${open} starting slot(s) unfilled`);
  // K and D/ST are roster legality, and real managers take them late: observed
  // K rounds 11-15 (median 15), D/ST 8-15 (median 14).
  for (const p of ['K', 'DST']) {
    const at = log.filter((r) => r.pos === p).map((r) => r.round);
    if (!at.length) fail.push(`no ${p}`);
    else if (at[0] < league.rounds - 4) fail.push(`${p} taken round ${at[0]}, too early`);
  }
  if (fail.length) problems.push({ seat, fail });
  rows.push({
    seat, counts, open, score, base,
    order: log.map((r) => r.pos).join(' '),
    benchFrom: log.findIndex((r) => r.bench) + 1,
  });
  if (VERBOSE) {
    console.error(`seat ${seat}: ` + P.map((p) => `${p} ${counts[p]}`).join(' · ') +
      `   lineup ${score.toFixed(1)} vs best-vorp ${base.vorp.toFixed(1)} ` +
      `best-adp ${base.adp.toFixed(1)}`);
    for (const r of log) {
      console.error(`  R${String(r.round).padStart(2)} ${r.pos.padEnd(4)} ` +
        `${r.name.padEnd(24)} now ${r.now.toFixed(1).padStart(7)} ` +
        `ev ${r.ev.toFixed(1).padStart(7)}${r.bench ? '  [bench]' : ''}`);
    }
  }
}

/* Per-seat wins and the mean margin say different things, and the mean is the
 * one that carries the claim. Best-available-by-VORP is greedy on exactly the
 * quantity being scored, so on any single seat it is a strong opponent and the
 * win/loss can turn on one tie-break; averaged over ten seats that noise
 * cancels and what is left is whether planning beats not planning. */
const mean = (f) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
const means = {
  model: mean((r) => r.score),
  vorp: mean((r) => r.base.vorp),
  adp: mean((r) => r.base.adp),
};

process.stdout.write(JSON.stringify({
  ok: !problems.length && means.model > means.vorp && means.model > means.adp,
  problems, beat, seats: league.teams, means, rows,
}, null, 1));
