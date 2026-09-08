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

/* 6. The card shows the five numbers it is supposed to and nothing it is not.
 *    A card that quietly loses BOONE (the newest of them, and the one fed by a
 *    join that can come back all-null) still renders and still looks fine. */
for (const lb of ['vorp', 'ecr', 'boone', 'avail', '+2']) {
  if (!recHtml.includes(`<span class="lb">${lb}</span>`)
      && !recHtml.includes(`class="lb">${lb}<`)) {
    fails.push(`the card has no ${lb.toUpperCase()} stat`);
  }
}
for (const gone of ['ADP ', 'insurance', 'lose by waiting', '&middot; EV ', 'bench']) {
  if (recHtml.includes(gone)) fails.push(`the card still shows "${gone.trim()}"`);
}
/* At least one card must show a real Boone rank rather than an em dash — the
 * hint join is the part that can silently come back empty. Conditional on the
 * PAYLOAD carrying any, because that decides whether there is anything to
 * show: hints are attached in cli.cmd_site for the live board only, and the
 * pytest fixture payload is built from the newest season with DRAFT PICKS
 * (2025), whose board has no hint columns at all. Run this file against the
 * payload inlined in docs/index.html to exercise the populated case. */
const boardHasBoone = payload.board.some((r) => r.boone !== null && r.boone !== undefined);
out.boone = { inPayload: boardHasBoone };
if (boardHasBoone) {
  const vals = [...recHtml.matchAll(/class="lb">boone<\/span><b>([^<]+)/g)].map((m) => m[1]);
  out.boone.shown = vals.slice(0, 8);
  if (!vals.some((v) => /^\d+$/.test(v))) {
    fails.push(`no card shows a Boone rank: ${JSON.stringify(vals.slice(0, 8))}`);
  }
}
for (const err of thrown) fails.push(`risk slider: threw ${err.message}`);
thrown.length = 0;

/* 7. LATE-ROUND QB. One dropdown with four consequences, and each of them is
 *    a place it could quietly do nothing:
 *      - your board loses its quarterbacks while YOU are on the clock;
 *      - it keeps them while anyone else is, or the pick another team just
 *        made could not be recorded — a failure that would only show up
 *        mid-draft, with no way out of it;
 *      - "Take now" loses them as candidates;
 *      - and the PLAN under it stops scheduling one before the round, which is
 *        the half a candidate filter alone would leave contradicting itself.
 *
 *    Top of round 3, seat 0 on the clock, sampling OFF so the jump is the
 *    model's argmax and the state is identical on every run. That state is
 *    chosen because the unconstrained answer USES a quarterback: three of them
 *    are in the panel and the plan takes one at #60. Run it a round later and
 *    the seat has already drafted one, and every check below passes vacuously.
 *    Nothing here asserts on P(avail): the rule binds your seat and not the
 *    market, so the survival numbers are supposed to be untouched.
 */
const teams = payload.model.league.teams;
const lateEl = document.getElementById('lateqb');
const setLate = (v) => { lateEl.fire('change', { target: { value: String(v) } }); flushTimers(); };
const nQbBoard = () => (boardEl.innerHTML.match(/pos-qb/g) || []).length;
const nQbRec = () => (recEl.innerHTML.match(/pos-qb/g) || []).length;
// plan steps that schedule a quarterback, as overall pick numbers
const planQbTurns = () => [...recEl.innerHTML.matchAll(/#(\d+) QB/g)].map((m) => Number(m[1]));
const roundOf = (pick) => Math.floor((pick - 1) / teams) + 1;

document.getElementById('reset').fire('click', {});
flushTimers();
document.getElementById('simsample').fire('change', { target: { checked: false } });
seatEl.fire('change', { target: { value: '0' } });
flushTimers();
document.getElementById('simto').value = String(2 * teams + 1);
document.getElementById('simgo').fire('click', {});
flushTimers();
if (document.getElementById('recommend-hd').textContent !== 'Take now') {
  fails.push('late-QB: seat 0 is not on the clock at the top of round 3');
}
setLate(0);
const off = { board: nQbBoard(), rec: nQbRec(), plan: planQbTurns() };
setLate(9);
const on9 = { board: nQbBoard(), rec: nQbRec(), plan: planQbTurns() };
setLate(13);
const on13 = { board: nQbBoard(), rec: nQbRec(), plan: planQbTurns() };
out.lateQb = { off, on9, on13 };

/* Whether the rule had anything to bite on is REPORTED, not asserted: the
 * checks below are invariants of the page and hold on any payload, but "a
 * quarterback was a candidate here with the rule off" is a property of the
 * board being rendered. It holds on the pytest fixture (2025), where the
 * caller asserts it, and does not on the live 2026 board, where the model
 * wants no passer in the first eight rounds at all. Failing here would mean a
 * test that breaks when the market changes. */
out.lateQb.testable = { board: off.board > 0, rec: off.rec > 0, plan: off.plan.length > 0 };
for (const [name, st9] of [['round 9', on9], ['round 13', on13]]) {
  if (st9.board !== 0) fails.push(`late-QB ${name} left ${st9.board} quarterbacks on your board`);
  if (st9.rec !== 0) fails.push(`late-QB ${name} left ${st9.rec} quarterbacks in Take now`);
}
for (const t of on9.plan) {
  if (roundOf(t) < 9) fails.push(`round 9: the plan still wants a QB at #${t} (round ${roundOf(t)})`);
}
for (const t of on13.plan) {
  if (roundOf(t) < 13) fails.push(`round 13: the plan still wants a QB at #${t} (round ${roundOf(t)})`);
}
// and the rule must MOVE the plan's quarterback rather than merely leave it
// alone: unconstrained, this board takes one well before either round
for (const [name, r, st9] of [['round 9', 9, on9], ['round 13', 13, on13]]) {
  if (st9.plan.length && off.plan.length && st9.plan[0] === off.plan[0]
      && roundOf(off.plan[0]) < r) {
    fails.push(`${name}: the plan's QB stayed at #${off.plan[0]}`);
  }
}
// ...and off the clock the board keeps them, or another team's quarterback
// could never be recorded
seatEl.fire('change', { target: { value: String(SEAT) } });
flushTimers();
out.lateQb.offClockBoard = nQbBoard();
if (out.lateQb.offClockBoard === 0) {
  fails.push('late-QB hid quarterbacks while another seat was on the clock');
}
// back off: the board is whole again
seatEl.fire('change', { target: { value: '0' } });
flushTimers();
setLate(0);
out.lateQb.restored = nQbBoard();
if (out.lateQb.restored !== off.board) {
  fails.push(`turning the rule off left ${out.lateQb.restored} of ${off.board} quarterbacks`);
}
for (const err of thrown) fails.push(`late-QB dropdown: threw ${err.message}`);
thrown.length = 0;

/* 8. SORTING the two survival columns. Both are read as "who am I about to
 *    lose", so the first click has to put the LEAST likely to last on top, and
 *    a blank — a player outside the simulation's watched top ~60 — must sink
 *    to the bottom whichever way the column points. The second half is the one
 *    a sentinel value gets wrong: 1e9 is "worst" ascending and "best"
 *    descending, so the reversed click led the table with rows that carry no
 *    measurement at all.
 *
 *    The header click is dispatched with a fake target rather than a real DOM
 *    node: the handler only ever asks `closest('th.sortable')`, so a target
 *    that answers that question is the whole contract. */
const clickHead = (key) => {
  boardEl.fire('click', {
    target: {
      closest: (sel) => (sel === 'th.sortable' ? { dataset: { key } } : null),
    },
  });
  flushTimers();
};
/* [value, ...] down the rendered column: '' for a blank cell. The two survival
 * columns are the last two `td.surv` of each row, in P(next), P(+2) order. */
const survColumn = (which) => [...boardEl.innerHTML.matchAll(/<tr data-pid="\d+">[\s\S]*?<\/tr>/g)]
  .map((m) => {
    const cells = [...m[0].matchAll(/<td class="num surv[^"]*">([^<]*)<\/td>/g)].map((c) => c[1]);
    const v = cells[which] === undefined ? '' : cells[which].trim();
    return v === '—' || v === '&mdash;' ? '' : v;
  });

const checkSorted = (label, col, wantAsc) => {
  const vals = survColumn(col);
  if (vals.length < 20) { fails.push(`${label}: only ${vals.length} rows to sort`); return; }
  const nums = [], blanks = [];
  vals.forEach((v, i) => (v === '' ? blanks : nums).push(i));
  if (!nums.length) { fails.push(`${label}: no measured rows in the column`); return; }
  if (!blanks.length) { fails.push(`${label}: no blank rows — the sink is untested`); return; }
  if (Math.min(...blanks) < Math.max(...nums)) {
    fails.push(`${label}: a blank sorts above a measured row `
      + `(first blank at ${Math.min(...blanks)}, last value at ${Math.max(...nums)})`);
  }
  const seq = nums.map((i) => Number(vals[i].replace('%', '')));
  for (let i = 1; i < seq.length; i++) {
    if (wantAsc ? seq[i] < seq[i - 1] : seq[i] > seq[i - 1]) {
      fails.push(`${label}: not ${wantAsc ? 'ascending' : 'descending'} at row ${i}: `
        + `${seq[i - 1]}% then ${seq[i]}%`);
      break;
    }
  }
  return seq;
};

for (const [key, col] of [['p', 0], ['p2', 1]]) {
  clickHead(key);                       // first click: the column's natural order
  const asc = checkSorted(`${key} first click`, col, true);
  clickHead(key);                       // second click: reversed
  const desc = checkSorted(`${key} second click`, col, false);
  out[`sort_${key}`] = { asc: (asc || []).slice(0, 6), desc: (desc || []).slice(0, 6) };
}
for (const err of thrown) fails.push(`survival sort: threw ${err.message}`);
thrown.length = 0;

out.ok = fails.length === 0;
out.fails = fails;
console.log(JSON.stringify(out, null, 1));
process.exit(out.ok ? 0 : 1);
