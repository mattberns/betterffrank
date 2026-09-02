/* The plan must not touch the prediction path.
 *
 * Survival probabilities are the calibrated number the page exists to show
 * (ECE 0.009, gap CI +/-0.3pp), and they were fitted before the plan existed.
 * The second simulation leg runs inside the same loop and draws from the same
 * stream, so the only thing standing between "the plan is free" and "the
 * calibration silently moved" is that leg two happens AFTER the survival count
 * and only ever consumes its own path's random numbers.
 *
 * Bit-identical, not approximately equal. Anything else means the streams
 * crossed.
 *
 *   node tests/simstable.mjs <payload.json>
 */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const E = require('../tendies/web/engine.js');
const fs = require('node:fs');

const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const M = payload.model, league = M.league;
const board = E.makeBoard(payload.board, M.positions);
board.season = payload.meta.season;
const managers = (payload.meta.managers || []).slice(0, league.teams);
while (managers.length < league.teams) managers.push('');

let worst = 0, jitter = 0, checked = 0, planless = 0;
for (const nPicks of [0, 3, 17, 44, 96, 130]) {
  const st = E.newState(board, league, managers);
  for (let i = 0; i < nPicks; i++) {
    const s = E.seatOnClock(st);
    const r = E.jointProbs(M, st, s, st.managers[s], 0);
    if (!r.pids.length) { E.advance(st, null); continue; }
    let b = 0;
    for (let k = 1; k < r.probs.length; k++) if (r.probs[k] > r.probs[b]) b = k;
    E.advance(st, r.pids[b]);
  }
  for (let seat = 0; seat < league.teams; seat += 3) {
    const withPlan = E.simulate(M, st, seat, { nSims: 300 });
    const noPlan = E.simulate(M, st, seat, { nSims: 300, plan: false });
    const again = E.simulate(M, st, seat, { nSims: 300 });
    if (!noPlan.availPaths) planless++;
    for (let i = 0; i < withPlan.p.length; i++) {
      worst = Math.max(worst, Math.abs(withPlan.p[i] - noPlan.p[i]));
      jitter = Math.max(jitter, Math.abs(withPlan.p[i] - again.p[i]));
      checked++;
    }
  }
}
process.stdout.write(JSON.stringify({
  checked, worst, jitter, planless, ok: worst === 0 && jitter === 0 && checked > 500,
}));
