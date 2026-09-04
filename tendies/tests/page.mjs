/* Load page.js against a stub DOM and drive real render cycles.
 *
 * page.js is ~1200 lines of DOM glue with, until this file, no coverage at
 * all — and it is an IIFE under 'use strict', so a free identifier is a
 * ReferenceError at RUN time, not at parse time. `node --check` passes, the
 * page loads, and the first render that reaches the bad line throws inside a
 * requestAnimationFrame callback where nothing is watching: the panels simply
 * stay empty. That is exactly how `league` (a const local to rebuild()) got
 * read from buildRec() and blanked the survival columns and all three
 * sim-dependent panels on every pick that was not the user's own.
 *
 * So this is a SMOKE test, deliberately: it asserts that a full render cycle
 * completes without throwing and that each panel produced content, on the
 * clock and off it. It says nothing about whether the numbers are right —
 * ties.mjs, plan.mjs, lineup.mjs and mockdraft.mjs do that against the engine.
 * What it catches is the whole panel silently disappearing.
 *
 *   node tests/page.mjs <payload.json> [seat]
 */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.resolve(import.meta.dirname, '..');
const payloadPath = process.argv[2];
const SEAT = Number(process.argv[3] || 2);
const payload = JSON.parse(fs.readFileSync(payloadPath, 'utf8'));

/* ---------------------------------------------------------- the stub DOM ---
 * Permissive on purpose: getElementById mints an element for ANY id, so the
 * test cannot break merely because site.py added a panel. Listeners are
 * recorded so the test can dispatch the ones it needs. */
const thrown = [];
const timers = [];

function makeEl(id) {
  const listeners = {};
  const el = {
    id, innerHTML: '', textContent: '', value: '', checked: false,
    disabled: false, children: [], style: {}, dataset: {},
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      toggle(c, on) { if (on === undefined ? this._s.has(c) : !on) this._s.delete(c); else this._s.add(c); },
      contains(c) { return this._s.has(c); },
    },
    addEventListener(kind, fn) { (listeners[kind] = listeners[kind] || []).push(fn); },
    removeEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    appendChild() {}, remove() {}, focus() {}, blur() {}, closest() { return null; },
    getAttribute() { return null; }, setAttribute() {},
    fire(kind, ev) { for (const fn of listeners[kind] || []) fn(ev); },
    listeners,
  };
  return el;
}

const els = new Map();
const document = {
  getElementById(id) {
    if (!els.has(id)) els.set(id, makeEl(id));
    return els.get(id);
  },
  querySelectorAll() { return []; },
  querySelector() { return null; },
  addEventListener() {}, removeEventListener() {},
  createElement: (tag) => makeEl(tag),
  body: makeEl('body'),
};

const store = new Map();
const sandbox = {
  console,
  document,
  localStorage: {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  },
  requestAnimationFrame(fn) {
    // synchronous, and errors are CAPTURED rather than swallowed — the browser
    // logs them to a console nobody is reading mid-draft
    try { fn(); } catch (err) { thrown.push(err); }
    return 0;
  },
  cancelAnimationFrame() {},
  setTimeout(fn, ms) { timers.push(fn); return timers.length; },
  clearTimeout(h) { if (h) timers[h - 1] = null; },
  confirm: () => true,
  alert: () => {},
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

/** Run every pending setTimeout — the FINE refinement pass. */
function flushTimers() {
  for (let i = 0; i < timers.length; i++) {
    const fn = timers[i];
    timers[i] = null;
    if (fn) { try { fn(); } catch (err) { thrown.push(err); } }
  }
}

const run = (file) => vm.runInContext(fs.readFileSync(path.join(ROOT, file), 'utf8'),
                                      sandbox, { filename: file });
run('tendies/web/engine.js');
sandbox.TENDIES = payload;
run('tendies/web/page.js');          // bootstraps: rebuild() + render()
flushTimers();

/* ------------------------------------------------------------- the checks ---
 * PANELS are the ones a ReferenceError in the sim path empties. `league` is
 * checked by name because it is the panel the bug was found in and the one
 * whose content depends on the probe. */
const PANELS = ['board', 'recommend', 'wait', 'upcoming', 'roster', 'league'];
const fails = [];
const seatEl = document.getElementById('seat');

function checkRender(label) {
  for (const err of thrown) fails.push(`${label}: threw ${err.message}`);
  thrown.length = 0;
  for (const id of PANELS) {
    const html = document.getElementById(id).innerHTML;
    if (!html || !html.trim()) fails.push(`${label}: #${id} rendered empty`);
  }
  const rec = document.getElementById('recommend').innerHTML;
  if (rec && !/simulations/.test(rec)) {
    fails.push(`${label}: #recommend has no simulation footer — it did not finish`);
  }
  return { rec, hd: document.getElementById('recommend-hd').textContent };
}

const out = { seat: SEAT, states: [] };

// 1. Fresh draft, pick 1. Seat 0 is on the clock, so any other seat is OFF it
//    and exercises the probe path.
seatEl.fire('change', { target: { value: String(SEAT) } });
flushTimers();
let r = checkRender(`pick 1, seat ${SEAT} off the clock`);
out.states.push({ at: 'pick 1 off clock', hd: r.hd, rows: (r.rec.match(/class="rec"/g) || []).length });

// 2. Same state, but the user IS on the clock.
seatEl.fire('change', { target: { value: '0' } });
flushTimers();
r = checkRender('pick 1, seat 0 on the clock');
out.states.push({ at: 'pick 1 on clock', hd: r.hd, rows: (r.rec.match(/class="rec"/g) || []).length });
if (r.hd !== 'Take now') fails.push(`on the clock the panel is titled "${r.hd}", want "Take now"`);

// 3. Mid-draft: sim ahead to pick 47 the way the header bar does, then read
//    the panel from a seat that is not on the clock. This is the state the
//    ReferenceError was reported from.
document.getElementById('simto').value = '47';
document.getElementById('simgo').fire('click', {});
flushTimers();
seatEl.fire('change', { target: { value: String(SEAT) } });
flushTimers();
r = checkRender(`mid-draft, seat ${SEAT} off the clock`);
out.states.push({ at: 'mid-draft off clock', hd: r.hd, rows: (r.rec.match(/class="rec"/g) || []).length });
if (!/^Your pick/.test(r.hd)) {
  fails.push(`off the clock the panel is titled "${r.hd}", want "Your pick — #N"`);
}

/* 4. The P(avail) threshold slider, which filters "Take now" and NOTHING
 *    else: tightening it narrows the panel, the top of its range shows every
 *    candidate, and the board's row count must not move at any setting. */
const boardEl = document.getElementById('board');
const recEl = document.getElementById('recommend');
const riskEl = document.getElementById('risk');
const nBoard = () => (boardEl.innerHTML.match(/<tr data-pid=/g) || []).length;
const nRec = () => (recEl.innerHTML.match(/class="rec"/g) || []).length;
// The rendered list is capped at 8, so the KEPT count is read off the slider's
// own label — which is also the thing standing in for a "nothing here" cell.
const kept = () => {
  const m = document.getElementById('riskv').textContent.match(/(\d+)\s*$/);
  return m ? Number(m[1]) : null;
};

const setRisk = (v) => { riskEl.fire('input', { target: { value: String(v) } }); flushTimers(); };
setRisk(100);
const all = kept(), boardAll = nBoard(), labelAll = document.getElementById('riskv').textContent;
setRisk(50);
const k50 = kept(), board50 = nBoard(), label50 = document.getElementById('riskv').textContent;
setRisk(5);
const k5 = kept(), board5 = nBoard();
out.risk = { all, k50, k5, rows50: nRec(), boardAll, board50, board5, labelAll, label50 };

// tightening the threshold narrows the panel, and 50% is a real cut, not a
// rounding of "everything" — the first version kept 273 of 332 here
if (!(k5 <= k50 && k50 < all)) {
  fails.push(`panel does not narrow: all=${all} 50%=${k50} 5%=${k5}`);
}
if (!(k50 < all / 2)) {
  fails.push(`50% keeps ${k50} of ${all} candidates — the filter is not biting`);
}
if (all < 1) fails.push('panel is empty at the top of the slider range');
// ...and the board is untouched at every setting. This is the constraint that
// was got wrong the first time round: the filter belongs to one panel.
if (!(boardAll === board50 && board50 === board5)) {
  fails.push(`the slider moved the BOARD: ${boardAll}/${board50}/${board5} rows`);
}
if (!/^50% . \d+$/.test(label50)) fails.push(`label reads "${label50}", want "50% · N"`);
if (!/^all . \d+$/.test(labelAll)) fails.push(`label reads "${labelAll}", want "all · N"`);

/* 5. No prose left in the panel. Every explanatory block carried one of these
 *    classes; `warn` covers the K/DST banner, `tie-note` the tie and
 *    assumed-path blocks. */
setRisk(100);
const recHtml = recEl.innerHTML;
for (const cls of ['tie-note', 'warn']) {
  if (recHtml.includes(`"${cls}`) || recHtml.includes(` ${cls}`)) {
    fails.push(`#recommend still renders a .${cls} text cell`);
  }
}
for (const phrase of ['Not this pick', 'Assumes', 'finishing lineup', 'are a tie',
                      'take one now', 'grey = projected']) {
  if (recHtml.includes(phrase)) fails.push(`#recommend still says "${phrase}"`);
}
for (const err of thrown) fails.push(`risk slider: threw ${err.message}`);
thrown.length = 0;

out.ok = fails.length === 0;
out.fails = fails;
console.log(JSON.stringify(out, null, 1));
process.exit(out.ok ? 0 : 1);
