/* fastForward is the page's skip-ahead: the model makes real picks until a
 * target seat is on the clock or a target pick number is up. It is parallel
 * code to simulate()'s inner step on purpose (that stream is frozen by
 * simstable.mjs), so it needs its own proof of the same properties: argmax is
 * deterministic, sampling is a pure function of (state, seedBase), stops land
 * exactly where they claim, it never picks a taken player, and it consumes
 * nothing from simulate's stream.
 *
 *   node tests/fastforward.mjs <payload.json>
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

const fresh = () => E.newState(board, league, managers);
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const problems = [];

// 1. argmax is deterministic and stops with the target seat on the clock
const s1a = fresh(), s1b = fresh();
const ff1a = E.fastForward(M, s1a, { stopSeat: 4 });
const ff1b = E.fastForward(M, s1b, { stopSeat: 4 });
if (!same(ff1a, ff1b)) problems.push('argmax not deterministic');
if (E.seatOnClock(s1a) !== 4) problems.push(`stopSeat: on clock ${E.seatOnClock(s1a)}, want 4`);
if (ff1a.length !== 4) problems.push(`stopSeat from empty made ${ff1a.length} picks, want 4`);

// 2. sampling is a pure function of (state, seedBase), and the base matters
const s2a = fresh(), s2b = fresh(), s2c = fresh();
const ff2a = E.fastForward(M, s2a, { stopPick: 100, sample: true, seedBase: 7 });
const ff2b = E.fastForward(M, s2b, { stopPick: 100, sample: true, seedBase: 7 });
const ff2c = E.fastForward(M, s2c, { stopPick: 100, sample: true, seedBase: 8 });
if (!same(ff2a, ff2b)) problems.push('same seedBase gave two different drafts');
if (same(ff2a, ff2c)) problems.push('different seedBases gave identical 99-pick drafts');

// 3. stopPick leaves the target pick ON the clock, simming through every seat
const s3 = fresh();
const ff3 = E.fastForward(M, s3, { stopPick: 37 });
if (E.pickNo(s3) !== 37) problems.push(`stopPick 37 landed on pick ${E.pickNo(s3)}`);
if (ff3.length !== 36) problems.push(`stopPick 37 made ${ff3.length} picks, want 36`);
for (let s = 0; s < league.teams; s++) {
  const want = Math.floor(36 / league.teams);
  if (s3.rosters[s].length < want) {
    problems.push(`seat ${s} has ${s3.rosters[s].length} players after pick 36, want >= ${want}`);
  }
}

// 4. a full sampled draft: every pid real, unique, and legal to the end
const s4 = fresh();
const ff4 = E.fastForward(M, s4, { sample: true, seedBase: 3 });
if (s4.picks.length !== s4.seats.length) {
  problems.push(`full draft stopped at ${s4.picks.length}/${s4.seats.length}`);
}
if (E.seatOnClock(s4) !== -1) problems.push('full draft did not end');
if (ff4.some((p) => p === null)) problems.push('full draft on the real board produced a null pick');
const seen = new Set();
for (const pid of ff4) {
  if (pid === null) continue;
  if (seen.has(pid)) problems.push(`pid ${pid} picked twice`);
  seen.add(pid);
}

// 5. stopping where you already stand is a no-op
const s5 = fresh();
const ff5 = E.fastForward(M, s5, { stopSeat: E.seatOnClock(s5) });
if (ff5.length !== 0) problems.push(`no-op stop made ${ff5.length} picks`);
const s5b = fresh();
const ff5b = E.fastForward(M, s5b, { stopPick: 1 });
if (ff5b.length !== 0) problems.push(`stopPick at the current pick made ${ff5b.length} picks`);

// 6. fast-forwarding a fork consumes nothing from simulate's stream
const s6 = fresh();
E.fastForward(M, s6, { stopPick: 18 });          // a mid-draft state, argmax
const before = E.simulate(M, s6, 0, { nSims: 300 });
E.fastForward(M, E.forkState(s6), { sample: true, seedBase: 11 });
const after = E.simulate(M, s6, 0, { nSims: 300 });
let worst = 0;
for (let i = 0; i < before.p.length; i++) {
  worst = Math.max(worst, Math.abs(before.p[i] - after.p[i]));
}
if (worst !== 0) problems.push(`simulate moved by ${worst} after a fastForward`);

process.stdout.write(JSON.stringify({
  problems,
  checked: ff1a.length + ff2a.length + ff3.length + ff4.length,
  fullDraft: ff4.length,
  ok: problems.length === 0,
}));
