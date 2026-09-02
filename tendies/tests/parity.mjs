/* Replay the golden fixtures through the JS engine and print what it computes.
 *
 * Usage: node parity.mjs <payload.json> <fixtures.json>
 *
 * test_parity.py runs this with the node binary that ships inside Playwright
 * and asserts the output matches Python to 1e-9. Any drift between
 * features.py and engine.js shows up here as a numeric diff rather than as
 * confident nonsense on the page during a live draft.
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const E = require('../tendies/web/engine.js');

const [, , payloadPath, fixturesPath] = process.argv;
const payload = JSON.parse(readFileSync(payloadPath, 'utf8'));
const fixtures = JSON.parse(readFileSync(fixturesPath, 'utf8'));

const model = payload.model;
E.assertContract(model);
E.assertBoard(payload.board);
const board = E.makeBoard(payload.board, model.positions);
board.season = payload.meta.season;

const out = {};
for (const c of fixtures.cases) {
  const st = E.newState(board, model.league, c.managers || payload.meta.managers);
  for (const pid of c.picks) E.advance(st, pid);
  const seat = E.seatOnClock(st);
  const cs = E.choiceSet(st, seat, model);
  const pf = E.positionFeatures(st, seat, cs, model);
  const mi = model.managers[c.manager] !== undefined ? model.managers[c.manager] : null;
  const pp = E.positionProbs(model, st, seat, cs, mi, false);
  const jp = E.jointProbs(model, st, seat, c.manager, 0);
  const order = jp.probs
    .map((p, i) => [jp.pids[i], p])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);
  out[c.id] = {
    seat,
    positions: pf.positions,
    posFeatures: pf.X,
    posProbs: pp.probs,
    outside: jp.outside,
    top: order,
  };
}
process.stdout.write(JSON.stringify(out));
