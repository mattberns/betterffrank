/* The payload's manager blocks must be exactly as wide as the model says.
 *
 * Usage: node contract.mjs <payload.json>
 *
 * `assertContract` guards the FEATURE NAME LISTS, which is the drift that
 * matters between features.py and engine.js. It cannot see the manager blocks
 * at all. Those were sliced open-endedly out of the coefficient vector
 * (`gamma[off:]`, `theta[off:]`) until the layout refactor, so a coefficient
 * appended after the last block was silently absorbed into it and the page
 * applied one manager's reach coefficient as another's. A total length that
 * happens to come out right is exactly the failure this catches and the
 * numeric parity test does not: parity compares Python to JS, and both sides
 * would have read the same mis-sliced array.
 */
import { readFileSync } from 'node:fs';

const [, , payloadPath] = process.argv;
const payload = JSON.parse(readFileSync(payloadPath, 'utf8'));
const M = payload.model;
const nMgr = Object.keys(M.managers).length;
const fails = [];

const want = (name, got, expect) => {
  if (got !== expect) fails.push(`${name}: ${got} != ${expect}`);
};

// Every per-manager channel is exactly n_managers wide and names a real
// player feature. A channel that is absent from the fit is absent from the
// list entirely, so there is no zero-length case to allow here.
if (!Array.isArray(M.player.channels)) {
  fails.push('player.channels: missing or not a list');
} else {
  const seen = new Set();
  for (const ch of M.player.channels) {
    want(`player.channels[${ch.name}]`, ch.beta.length, nMgr);
    if (M.player.features.indexOf(ch.feature) < 0) {
      fails.push(`player.channels[${ch.name}]: feature ${ch.feature} not in player.features`);
    }
    if (seen.has(ch.name)) fails.push(`player.channels[${ch.name}]: duplicated`);
    seen.add(ch.name);
  }
}

const H = M.position.human;
if (H.managerAsc.length !== 0) {
  want('position.human.managerAsc', H.managerAsc.length, nMgr * H.managerPos.length);
}
if (H.managerEarly === undefined) {
  fails.push('position.human.managerEarly: missing from payload');
} else if (H.managerEarly.length !== 0) {
  want('position.human.managerEarly',
       H.managerEarly.length, nMgr * H.managerEarlyPos.length);
}

want('player.posAdp', M.player.posAdp.length, M.positions.length);
want('player.outside', M.player.outside.length, M.positions.length);
want('player.beta', M.player.beta.length, M.player.features.length);
want('position.human.beta', H.beta.length, M.position.features.length);
want('position.auto.beta', M.position.auto.beta.length, M.position.features.length);
want('position.human.asc', H.asc.length, M.position.ascPositions.length);
want('player.mean', M.player.mean.length, M.player.features.length);
want('player.sd', M.player.sd.length, M.player.features.length);
want('position.mean', M.position.mean.length, M.position.features.length);
want('position.sd', M.position.sd.length, M.position.features.length);

if (typeof M.earlyRounds !== 'number') fails.push('earlyRounds: missing from payload');
if (typeof M.endgameRounds !== 'number') fails.push('endgameRounds: missing');

if (fails.length) {
  console.error('payload contract violations:\n  ' + fails.join('\n  '));
  process.exit(1);
}
console.log(`ok — ${nMgr} managers, ${M.player.channels.length} channels, all blocks exact`);
