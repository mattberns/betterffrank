/* Draft-day UI. All modelling lives in engine.js; this file is DOM only. */
'use strict';
(function () {
  const E = window.TendiesEngine;
  const D = window.TENDIES;
  const M = D.model;
  E.assertContract(M);

  const board = E.makeBoard(D.board, M.positions);
  board.season = D.meta.season;
  const KEY = 'tendies-' + D.meta.league_id + '-' + D.meta.season;
  // sims, then plan paths. The second number drives the "+2 picks" horizon and
  // the plan DP; it is small on the interactive pass and grows when idle.
  const FAST = [600, 48], FINE = [2400, 200];
  const WAIT_P = 0.8;        // the panel's threshold
  const WAIT_N = 14;         // rows per section before it says "+N more"
  // declared up here because load() validates against it, and load() runs first
  const SORT_KEYS = ['adp', 'ecr', 'edge', 'name', 'pos', 'team', 'vorp', 'p', 'p2'];

  const el = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const posName = (p) => M.positions[p];
  const posCls = (p) => posName(p).toLowerCase().replace('/', '');
  const tag = (p) => `<span class="tag pos-${posCls(p)}">${posName(p)}</span>`;

  // INVARIANT: ui.simmed.length === ui.picks.length at every save().
  // simmed[i] === 0 means picks[i] was recorded by hand (draft / off-board);
  // simmed[i] === k > 0 means entry i was produced by sim jump k. Every
  // mutation of ui.picks mutates ui.simmed at the same index — push with push,
  // pop with pop, clear with clear. "undo sim" pops the trailing run that
  // shares one nonzero id. v stays 1: old stored blobs backfill zeros, old
  // page builds ignore the extra fields.
  const defaults = () => ({
    v: 1,
    teams: M.league.teams,
    rounds: M.league.rounds,
    seat: 0,
    managers: (D.meta.managers || []).slice(0, M.league.teams),
    autoMode: new Array(M.league.teams).fill('prior'),
    picks: [],
    simmed: [],
    simSample: true,
    sort: { key: 'adp', dir: 1 },
  });

  let ui = load();
  let st = null;
  let simmedPids = new Set();   // pids on rosters via a sim jump, for badging
  let spec = null;
  let sim = null;
  let simFine = false;
  let idleHandle = null;
  let planCache = null;
  // the player card is answered for whoever owns the question, which is usually
  // NOT your seat; those sims and plans are built on demand and kept for the
  // life of one render
  const seatSims = new Map();
  const seatPlans = new Map();

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(KEY) || 'null');
      if (!raw || raw.v !== 1) return defaults();
      const d = defaults();
      d.teams = raw.teams || d.teams;
      d.rounds = raw.rounds || d.rounds;
      d.seat = Math.min(Math.max(0, raw.seat | 0), d.teams - 1);
      if (Array.isArray(raw.managers)) d.managers = raw.managers.slice(0, d.teams);
      if (Array.isArray(raw.autoMode)) d.autoMode = raw.autoMode.slice(0, d.teams);
      // every pid is validated against the board; a corrupt entry drops the
      // draft — and drops its provenance flag with it, keeping the pair aligned
      if (Array.isArray(raw.picks)) {
        const flags = Array.isArray(raw.simmed) ? raw.simmed : [];
        raw.picks.forEach((p, i) => {
          if (p === null || (p >= 0 && p < board.n)) {
            d.picks.push(p);
            d.simmed.push(flags[i] > 0 ? flags[i] | 0 : 0);
          }
        });
      }
      d.simSample = raw.simSample !== false;
      if (raw.sort && SORT_KEYS.indexOf(raw.sort.key) >= 0) {
        d.sort = { key: raw.sort.key, dir: raw.sort.dir < 0 ? -1 : 1 };
      }
      while (d.managers.length < d.teams) d.managers.push('');
      while (d.autoMode.length < d.teams) d.autoMode.push('prior');
      return d;
    } catch (err) { return defaults(); }
  }

  function save() { try { localStorage.setItem(KEY, JSON.stringify(ui)); } catch (err) {} }

  function rebuild() {
    const league = Object.assign({}, M.league, { teams: ui.teams, rounds: ui.rounds });
    st = E.newState(board, league, ui.managers);
    simmedPids = new Set();
    const seen = new Set();
    for (let i = 0; i < ui.picks.length; i++) {
      const pid = ui.picks[i];
      if (pid !== null && seen.has(pid)) continue;
      if (pid !== null) {
        seen.add(pid);
        if (ui.simmed[i] > 0) simmedPids.add(pid);
      }
      E.advance(st, pid);
    }
    // the empty-slot floors are read off the board AS IT STANDS, so the spec is
    // rebuilt with the state rather than cached across picks
    spec = E.lineupSpec(league, M.positions, E.emptyValues(M, st), E.blindMask(M, M.positions),
                        E.holdValues(M, M.positions));
    planCache = null;
    seatSims.clear(); seatPlans.clear();
  }

  function autoOverride() {
    const o = {};
    ui.autoMode.forEach((mode, s) => {
      if (mode === 'auto') o[s] = 1;
      else if (mode === 'human') o[s] = 0;
    });
    return o;
  }

  /** The seat's autodraft weight as the model believes it: an explicit
   *  auto/human setting wins, 'prior' falls back to the fitted chain. */
  function autoWeightFor(s) {
    const mode = ui.autoMode[s];
    if (mode === 'auto') return 1;
    if (mode === 'human') return 0;
    return M.autoChain && M.autoChain.p_start[ui.managers[s]] !== undefined
      ? M.autoChain.p_start[ui.managers[s]]
      : (M.autoChain ? M.autoChain.league_start : 0);
  }

  function runSim(n, paths) {
    sim = E.simulate(M, st, ui.seat, {
      nSims: n, planPaths: paths, autoOverride: autoOverride(),
    });
    sim.index = new Map(sim.pids.map((pid, i) => [pid, i]));
    planCache = null;
  }

  /** The plan DP for the current state, computed at most once per render. */
  function planNow() {
    if (!planCache && sim) planCache = E.plan(M, st, ui.seat, sim, spec);
    return planCache;
  }

  /* A card about someone else's pick needs THEIR simulation: the survival
   * horizons are their turns, not yours, and the plan is over their remaining
   * picks. One extra sim per seat per render, at the interactive budget. */
  function simFor(seat) {
    if (seat === ui.seat && sim) return sim;
    if (seatSims.has(seat)) return seatSims.get(seat);
    const s = E.simulate(M, st, seat, {
      nSims: FAST[0], planPaths: FAST[1], autoOverride: autoOverride(),
    });
    s.index = new Map(s.pids.map((pid, i) => [pid, i]));
    seatSims.set(seat, s);
    return s;
  }

  function planFor(seat, sm) {
    if (seat === ui.seat) return planNow();
    if (seatPlans.has(seat)) return seatPlans.get(seat);
    const pl = sm ? E.plan(M, st, seat, sm, spec) : null;
    seatPlans.set(seat, pl);
    return pl;
  }

  /** Which roster this player is a question about: his owner if he is drafted,
   *  otherwise whoever is on the clock. Not your seat — you are usually not the
   *  one picking. */
  /** Positions the seat on the clock may draft right now (cap + endgame
   *  squeeze, per E.legalPositions), as a Set of names; null when no one is on
   *  the clock. Every draft path checks this: the board filter, draft(), and
   *  the player card's button. */
  function legalNow() {
    const s = E.seatOnClock(st);
    if (s < 0) return null;
    return new Set(E.legalPositions(st, s).map(posName));
  }

  function subjectSeat(pid) {
    for (let s = 0; s < ui.teams; s++) if (st.rosters[s].indexOf(pid) >= 0) return s;
    const on = E.seatOnClock(st);
    return on >= 0 ? on : ui.seat;
  }

  /* --------------------------------------------------------------- render */

  // every panel whose content depends on the simulation (Take now, Safe to
  // wait on, Upcoming) dims together while the sim recomputes
  function setComputing(on) {
    document.querySelectorAll('.simdep').forEach((p) => p.classList.toggle('computing', on));
  }

  function render() {
    closeInspect();
    renderStatus();
    renderSeats();
    renderLeague();
    renderRoster();
    renderBoard();
    if (gridOpen) renderGrid();   // stays open across a pick; it is the live view
    setComputing(true);
    if (idleHandle) { clearTimeout(idleHandle); idleHandle = null; }
    requestAnimationFrame(() => {
      runSim(FAST[0], FAST[1]); simFine = false;
      renderBoard();          // the survival columns depend on the sim
      renderPanels();
      setComputing(false);
      idleHandle = setTimeout(() => {
        runSim(FINE[0], FINE[1]); simFine = true; renderBoard(); renderPanels();
      }, 350);
    });
  }

  function renderStatus() {
    const seat = E.seatOnClock(st);
    const pick = E.pickNo(st);
    const round = Math.floor((pick - 1) / ui.teams) + 1;
    const mine = E.nextPickFor(st, ui.seat, seat === ui.seat ? st.picks.length + 1 : st.picks.length);
    const mgr = seat >= 0 ? (ui.managers[seat] || 'seat ' + (seat + 1)) : '—';
    const nSim = ui.simmed.reduce((n, x) => n + (x > 0 ? 1 : 0), 0);
    el('status').innerHTML = (seat < 0
      ? '<strong>Draft complete</strong>'
      : `<strong>Pick ${pick}</strong> <span class="dim">(round ${round})</span>
         &middot; on the clock: <strong>${esc(label(mgr))}</strong>
         ${seat === ui.seat ? '<span class="you">YOU</span>' : ''}
         &middot; your next pick: <strong>${mine || '—'}</strong>`)
      + (nSim ? ` <span class="dim">&middot; ${nSim} simmed</span>` : '');
    el('simundo').disabled = !(ui.simmed.length && ui.simmed[ui.simmed.length - 1] > 0);
    el('simnext').disabled = seat < 0 || seat === ui.seat;
  }

  function label(m) { return (D.meta.labels && D.meta.labels[m]) || m || 'unknown'; }

  /** Rebuilt on every render, because dragging the draft order renames seats. */
  function renderSeats() {
    el('seat').innerHTML = ui.managers.map((m, s) =>
      `<option value="${s}"${s === ui.seat ? ' selected' : ''}>${s + 1}. ${esc(label(m))}</option>`
    ).join('');
  }

  function renderLeague() {
    const rows = [];
    for (let s = 0; s < ui.teams; s++) {
      const on = E.seatOnClock(st) === s;
      const ns = st.rosters[s].reduce((n, p) => n + (simmedPids.has(p) ? 1 : 0), 0);
      rows.push(`<tr data-seat="${s}" class="${on ? 'onclock' : ''}${s === ui.seat ? ' mine' : ''}">
        <td class="dim hdl" draggable="true" title="drag to change the draft order">${s + 1}</td>
        <td>${esc(label(ui.managers[s]))}${ns
          ? ` <span class="dim small">${ns} sim</span>` : ''}</td>
        <td><select data-seat="${s}" class="automode">
          ${['prior', 'human', 'auto'].map((m) =>
            `<option value="${m}"${ui.autoMode[s] === m ? ' selected' : ''}>${m}</option>`).join('')}
        </select></td></tr>`);
    }
    // Reordering replays the recorded picks against the new order, so picks
    // already made change hands. That is the right behaviour — it is how you
    // fix an order you got wrong three picks in — but it should not be a
    // surprise, and it is undone by dragging back.
    const note = ui.picks.length
      ? `<div class="dim small pad">${ui.picks.length} pick${ui.picks.length === 1 ? '' : 's'}
         recorded — reordering now reassigns them to whoever holds the slot.</div>`
      : '';
    el('league').innerHTML = `<table><tbody>${rows.join('')}</tbody></table>${note}`;
  }

  /* Slot by slot, not a flat list: the same greedy assignment every EV number
   * on the page is computed from (E.lineupSlots), so the panel and the model
   * never disagree about who starts. Whoever the assignment leaves over is the
   * bench. Slots are reordered for DISPLAY only — flex between TE and DST/K —
   * the engine's own order (and the sums tests assert) is untouched. */
  function renderRoster() {
    const r = st.rosters[ui.seat] || [];
    const slots = E.lineupSlots(board, spec, r, null);
    const ord = (s) => (s.pos < 0 ? 4.5
      : ({ QB: 0, RB: 1, WR: 2, TE: 3, DST: 5, K: 6 })[posName(s.pos)] ?? 7);
    const shown = slots.slice().sort((a, b) => ord(a) - ord(b));
    const assigned = new Set(shown.filter((s) => s.pid >= 0).map((s) => s.pid));
    const row = (lab, pid, value) => {
      const nm = pid < 0 ? '<span class="dim">&mdash;</span>'
        : `${tag(board.pos[pid])} <span class="rnm${
            simmedPids.has(pid) ? ' sim' : ''}">${esc(board.name[pid])}</span>`;
      return `<tr${pid >= 0 ? ` data-pid="${pid}"` : ' class="hole"'}>
        <td class="slab">${lab}</td><td class="rn">${nm}</td>
        <td class="num val dim">${value === null ? '' : value.toFixed(0)}</td></tr>`;
    };
    const rows = shown.map((s) => row(s.label, s.pid, s.value));
    // bench capacity = picks the draft gives you beyond the starting slots;
    // unfilled bench rows show as empty, same as an unfilled starter
    const bench = r.filter((pid) => !assigned.has(pid));
    const benchN = Math.max(bench.length, ui.rounds - shown.length);
    for (let i = 0; i < benchN; i++) {
      rows.push(i < bench.length ? row('BN', bench[i], board.vorp[bench[i]]) : row('BN', -1, null));
    }
    el('roster').innerHTML = `<table>${rows.join('')}</table>`;
  }

  /* ------------------------------------------------------------ the board */

  let filterPos = 'ALL';
  let query = '';

  const sv = (s, pid) => (s && s.index && s.index.has(pid) ? s.p[s.index.get(pid)] : null);
  const sv2 = (s, pid) => (s && s.p2 && s.index && s.index.has(pid) ? s.p2[s.index.get(pid)] : null);
  const survOf = (pid) => sv(sim, pid);
  const surv2Of = (pid) => sv2(sim, pid);
  const edgeOf = (pid) => (board.hasEcr[pid] ? board.adpRank[pid] - board.ecr[pid] : null);

  /* Each comparator is written in its NATURAL direction — smallest ADP first,
   * largest VORP first — and `dir` flips it. So the header arrow has to be read
   * off the column, not off `dir`, or half of them point the wrong way. */
  const SORTS = {
    adp: (a, b) => board.adpRank[a] - board.adpRank[b],
    ecr: (a, b) => (board.hasEcr[a] ? board.ecr[a] : 1e9) - (board.hasEcr[b] ? board.ecr[b] : 1e9),
    edge: (a, b) => (edgeOf(b) === null ? -1e9 : edgeOf(b)) - (edgeOf(a) === null ? -1e9 : edgeOf(a)),
    name: (a, b) => (board.name[a] < board.name[b] ? -1 : board.name[a] > board.name[b] ? 1 : 0),
    pos: (a, b) => board.pos[a] - board.pos[b],
    team: (a, b) => (board.team[a] < board.team[b] ? -1 : board.team[a] > board.team[b] ? 1 : 0),
    vorp: (a, b) => board.vorp[b] - board.vorp[a],
    p: (a, b) => (survOf(b) === null ? -1 : survOf(b)) - (survOf(a) === null ? -1 : survOf(a)),
    p2: (a, b) => (surv2Of(b) === null ? -1 : surv2Of(b)) - (surv2Of(a) === null ? -1 : surv2Of(a)),
  };
  const DESC_NATURAL = { edge: 1, vorp: 1, p: 1, p2: 1 };

  function cmpFor(key, dir) {
    const f = SORTS[key] || SORTS.adp;
    return (a, b) => {
      const d = f(a, b);
      // ADP breaks every tie, so the order is total and the table never
      // reshuffles under an equal-valued sort the way a stable-sort-on-zeros did
      return (d ? d * dir : 0) || (board.adpRank[a] - board.adpRank[b]);
    };
  }

  const COLS = [
    { key: 'adp', head: 'ADP', cls: 'num' },
    { key: 'ecr', head: 'ECR', cls: 'num' },
    { key: 'edge', head: 'EDGE', cls: 'num',
      title: 'ADP minus ECR: positive = experts rate him above his market price' },
    { key: 'name', head: 'Player', cls: '' },
    { key: 'pos', head: 'Pos', cls: '' },
    { key: 'team', head: 'Tm', cls: '' },
    { key: 'vorp', head: 'VORP', cls: 'num', title: 'value over replacement, priced off ECR' },
    { key: 'p', head: 'P(next)', cls: 'num', title: 'P(still available at your next pick)' },
    { key: 'p2', head: 'P(+2)', cls: 'num',
      title: 'P(still available at the pick after that) — coarser, and optimistic: see the panel note' },
    { key: null, head: '', cls: '' },      // the fit button
  ];

  function renderBoard() {
    // only what the seat on the clock can legally roster is on offer
    const seat = E.seatOnClock(st);
    const legal = legalNow();
    const openN = seat < 0 ? 0 : E.openPositions(st, seat).length;
    if (legal && filterPos !== 'ALL' && !legal.has(filterPos)) {
      filterPos = 'ALL';   // the active filter's position just became illegal
    }
    [...el('filters').children].forEach((c) => {
      c.classList.toggle('on', c.dataset.pos === filterPos);
      c.disabled = !!legal && c.dataset.pos !== 'ALL' && !legal.has(c.dataset.pos);
    });
    const avail = [];
    for (let pid = 0; pid < board.n; pid++) {
      if (st.taken[pid]) continue;
      if (legal && !legal.has(posName(board.pos[pid]))) continue;
      if (filterPos !== 'ALL' && posName(board.pos[pid]) !== filterPos) continue;
      if (query && board.name[pid].toLowerCase().indexOf(query) < 0) continue;
      avail.push(pid);
    }
    avail.sort(cmpFor(ui.sort.key, ui.sort.dir));
    const shown = avail.slice(0, 200);

    const rows = shown.map((pid) => {
      const p = survOf(pid), p2 = surv2Of(pid);
      // ADP is where the market has him; ECR is what the experts think he is
      // worth. The gap is the actionable part, so it gets its own column and a
      // colour: green = experts rate him above his price.
      const gap = edgeOf(pid);
      const gapCls = gap === null ? '' : gap >= 8 ? 'safe' : gap <= -8 ? 'gone' : 'dim';
      return `<tr data-pid="${pid}">
        <td class="dim num">${board.adpRank[pid]}</td>
        <td class="num">${board.hasEcr[pid] ? board.ecr[pid] : '—'}</td>
        <td class="num ${gapCls}">${gap === null ? '' : (gap > 0 ? '+' : '') + gap}</td>
        <td class="nm">${esc(board.name[pid])}${board.rookie[pid]
          ? ' <span class="rk" title="rookie — NFL entry year">R</span>' : ''}</td>
        <td>${tag(board.pos[pid])}</td>
        <td class="dim">${esc(board.team[pid])}</td>
        <td class="num">${board.vorp[pid].toFixed(0)}</td>
        <td class="num surv ${p === null ? '' : survClass(p)}">${pct(p)}</td>
        <td class="num surv ${p2 === null ? '' : survClass(p2)}">${pct(p2)}</td>
        <td class="fitc"><button class="fit"
          title="what he does to your roster — does not draft him">fit</button></td>
      </tr>`;
    });

    const head = COLS.map((c) => {
      if (!c.key) return '<th></th>';
      const on = ui.sort.key === c.key;
      const up = DESC_NATURAL[c.key] ? ui.sort.dir < 0 : ui.sort.dir > 0;
      return `<th class="sortable ${c.cls}${on ? ' sorted' : ''}" data-key="${c.key}"
        ${c.title ? `title="${esc(c.title)}"` : ''}>${c.head}${
        on ? `<span class="arw">${up ? '&#9650;' : '&#9660;'}</span>` : ''}</th>`;
    }).join('');

    const forced = legal && legal.size < openN
      ? `<div class="warn pad small">Every remaining pick is owed to an open starting slot —
         ${esc(label(ui.managers[seat]))} can only draft ${[...legal].join(' &middot; ')}.</div>`
      : '';
    el('board').innerHTML = forced + `<table><thead><tr>${head}</tr></thead>
      <tbody>${rows.join('')}</tbody></table>` +
      (avail.length > shown.length
        ? `<div class="pad dim small">${avail.length - shown.length} more — search or filter</div>` : '');
  }

  const pct = (v) => (v === null || v === undefined ? '—' : Math.round(v * 100) + '%');
  function survClass(p) { return p < 0.25 ? 'gone' : p < 0.6 ? 'risky' : 'safe'; }

  /* -------------------------------------------------------------- panels */

  function renderPanels() {
    renderUpcoming();
    renderRecommend();
    renderWait();
  }

  function renderUpcoming() {
    const seat = E.seatOnClock(st);
    if (seat < 0) { el('upcoming').innerHTML = ''; return; }
    const mine = E.nextPickFor(st, ui.seat, seat === ui.seat ? st.picks.length + 1 : st.picks.length);
    const cards = [];
    const probe = E.forkState(st);
    let guard = 0;
    while (E.pickNo(probe) < (mine || 0) && guard++ < 24) {
      const s = E.seatOnClock(probe);
      if (s < 0) break;
      const aw = autoWeightFor(s);
      const r = E.jointProbs(M, probe, s, probe.managers[s], aw);
      const order = r.probs.map((p, i) => [r.pids[i], p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
      const posMix = r.positions.map((p, i) => [posName(p), r.posProbs[i]])
        .sort((a, b) => b[1] - a[1]).slice(0, 4)
        .map(([n, v]) => `${n} ${Math.round(v * 100)}%`).join(' &middot; ');
      const thin = (D.meta.pickCounts && (D.meta.pickCounts[probe.managers[s]] || 0) < 40);
      cards.push(`<div class="card">
        <div class="card-hd">#${E.pickNo(probe)} ${esc(label(probe.managers[s]))}
          ${aw > 0.5 ? '<span class="badge auto">auto</span>' : ''}
          ${thin ? '<span class="badge pooled">pooled</span>' : ''}</div>
        ${order.map(([pid, p]) => `<div class="opt">
            <span class="bar" style="width:${Math.round(p * 100)}%"></span>
            ${tag(board.pos[pid])}
            <span class="onm" data-pid="${pid}">${esc(board.name[pid])}</span>
            <span class="op">${Math.round(p * 100)}%</span></div>`).join('')}
        <div class="mix dim">${posMix}</div></div>`);
      // advance the probe by its most likely pick, so later cards see a
      // plausible board rather than today's
      let best = 0;
      for (let i = 1; i < r.probs.length; i++) if (r.probs[i] > r.probs[best]) best = i;
      E.advance(probe, r.pids.length ? r.pids[best] : null);
    }
    el('upcoming').innerHTML = cards.join('') ||
      '<div class="dim pad">You are on the clock.</div>';
  }

  function renderRecommend() {
    const seat = ui.seat;
    if (!sim || !sim.pids.length) { el('recommend').innerHTML = ''; return; }
    // The whole legal board, not a fixed slice of it. `candidates` used to cap
    // at 8 per position in ADP order, which silently removed the biggest
    // expert-vs-market edges from a list ranked by expert-vs-market edge.
    const all = E.recommend(M, st, seat, sim, E.candidates(st, seat), spec);
    const rec = all.slice(0, 8);
    const onClock = E.seatOnClock(st) === seat;

    // Say which question is being answered. The old copy here read "Starters
    // covered", which was not the condition the engine tests and was flatly
    // false whenever the plan meant to fill an open slot at a later turn — the
    // usual middle-round case. It told a manager with an empty WR2 that his
    // starters were covered.
    const head = all.bench
      ? 'Nothing here improves the lineup you would <b>finish</b> with — every slot this pick '
        + 'could fill, the plan below already fills as cheaply later. So this is a bench pick: '
        + 'ranking by <b>insurance</b> (what he is worth if a starter at his position is lost) '
        + 'plus <b>market edge</b> (where the experts have him above his ADP price). '
        + 'Not lineup points.'
      : (onClock
        ? 'Value of the lineup you would <b>finish</b> with: what he adds now, plus the best plan '
          + 'for every turn after this one.'
        : 'Not your pick — showing value if it were.');

    // The plan is the part a single recommendation hides: it is why a receiver
    // now is fine when the back is coming at your next turn, and why it is not.
    //
    // It is shown in BENCH mode too, and that is the mode that needs it most:
    // "nothing improves your lineup" is only believable alongside the turns
    // that do the filling. Hiding it there (as this did until 2026-09-02) left
    // the one claim the recommendation rests on with no evidence on screen.
    const plan = (all.plan || []).filter((s) => s.pos >= 0).slice(0, 6);
    const planHtml = plan.length
      ? `<div class="planline dim small">plan: ${plan.map((s) =>
          `<span class="${s.modelled ? '' : 'proj'}">#${s.turn} ${posName(s.pos)}</span>`)
          .join(' → ')}${plan.some((s) => !s.modelled)
          ? ' <span class="proj">(grey = projected from 2021-25 draft flow)</span>' : ''}</div>`
      : '';

    // When the curve cannot separate the top of the list, say so instead of
    // letting the rank numbers imply an order that is not there.
    //
    // Measured over the ROWS ON SCREEN, not over the whole tier: `all.tieBand`
    // spans every tied candidate, and its widest member is some deep
    // quarterback carrying a +/-70 band that says nothing about the eight names
    // shown. The claim in this sentence is "#1 cannot be shown better than the
    // last one listed", so the number quoted is that pair's own band.
    const topTier = rec.length ? rec.filter((r) => r.tier === rec[0].tier) : [];
    const nTied = topTier.length;
    const band = nTied > 1 ? topTier[nTied - 1].tieBand : 0;
    const more = (all.tied || 0) - nTied;
    const tieHtml = nTied > 1
      ? `<div class="pad small tie-note">Top ${nTied} are a <b>tie</b>: the VORP curve cannot
         show the first is better than the ${nTied === 2 ? 'second' : nTied + 'th'}
         (${isFinite(band) ? '±' + band.toFixed(1) + ' pts at '
             + (all.z || 1.96).toFixed(2) + ' SE'
           : 'no standard errors in this payload'})${more > 0
             ? `, and ${more} more below them` : ''}. Still ordered by the best estimate —
         a tie is not indifference — but overruling it on <b>lose by waiting</b>, a bye
         clash or a handcuff costs you nothing this list can measure.</div>`
      : '';

    const urgent = rec.length && rec[0].need > 0.5
      ? `<div class="warn pad small">No ${rec[0].need >= 1.5 ? 'kicker or defense' : 'kicker/defense'}`
        + ' is projected to be available at your remaining turns — take one now.</div>'
      : '';

    // Rank badges follow the TIER, not the row: every member of a tie shows the
    // same number, with an `=` so it reads as a tie rather than a typo. Two
    // players the curve cannot separate must not be labelled 1 and 2.
    const tiers = [];
    for (const r of rec) if (!tiers.includes(r.tier)) tiers.push(r.tier);
    const tierSize = (t) => rec.filter((r) => r.tier === t).length;

    const body = rec.map((r) => {
      const rank = tiers.indexOf(r.tier) + 1;
      const shared = tierSize(r.tier) > 1;
      // per row, because a turn can genuinely offer both: someone who improves
      // the lineup now, and below him the best of the players who cannot
      const nums = r.bench
        ? `<span class="proj">bench</span> &middot; insurance <b>${r.insurance.toFixed(1)}</b>
           &middot; edge <b>${r.edge >= 0 ? '+' : ''}${r.edge.toFixed(1)}</b>`
        : `now <b>${r.now.toFixed(1)}</b>
           &middot; then <b>${r.later.toFixed(1)}</b>
           &middot; EV <b>${r.ev.toFixed(1)}</b>`;
      return `<div class="rec">
        <div class="rec-hd"><span class="rk${shared ? ' rk-tie' : ''}">${rank}${
          shared ? '=' : ''}</span>${tag(board.pos[r.pid])}
          <span class="onm" data-pid="${r.pid}">${esc(board.name[r.pid])}</span></div>
        <div class="rec-bd dim">ADP ${board.adpRank[r.pid]} &middot; ECR ${
          board.hasEcr[r.pid] ? board.ecr[r.pid] : '—'} &middot; VORP ${board.vorp[r.pid].toFixed(0)}${
          isFinite(board.vorpSe[r.pid]) && board.vorpSe[r.pid] > 0
            ? ' <span class="proj">±' + board.vorpSe[r.pid].toFixed(0) + '</span>' : ''}<br>
          ${nums}
          &middot; lose by waiting <b>${r.cost.toFixed(1)}</b>
          &middot; P(avail) <b class="${survClass(r.pAvail)}">${Math.round(r.pAvail * 100)}%</b></div>
      </div>`;
    }).join('');

    el('recommend').innerHTML =
      `<div class="dim pad small">${head}</div>${urgent}${tieHtml}${planHtml}${body}` +
      `<div class="dim pad small">${sim.nSims} simulations${simFine ? '' : ' (refining…)'}
        &middot; +/-${(100 * 0.5 / Math.sqrt(sim.nSims)).toFixed(1)}pp at 50%
        &middot; plan over ${sim.availPaths} paths</div>`;
  }

  /* Who you can afford to pass on. Same survival numbers as the board, read the
   * other way round: not "will he last" per player, but "what is still there"
   * per turn. Restricted to positions the seat can still legally draft, since a
   * capped position is not a choice. */
  function renderWait() {
    if (!sim || !sim.pids.length) { el('wait').innerHTML = '<div class="dim pad">—</div>'; return; }
    const turns = sim.turns || [];
    const open = new Set(E.openPositions(st, ui.seat));
    const moe = (n) => (100 * Math.sqrt(WAIT_P * (1 - WAIT_P) / Math.max(1, n))).toFixed(1);

    const section = (title, sub, probe, warn) => {
      const hits = sim.pids
        .filter((pid) => !st.taken[pid] && open.has(board.pos[pid]) && probe(pid) >= WAIT_P)
        .sort((a, b) => board.vorp[b] - board.vorp[a]);
      const rows = hits.slice(0, WAIT_N).map((pid) => `<div class="wr">
        ${tag(board.pos[pid])}
        <span class="onm" data-pid="${pid}">${esc(board.name[pid])}</span>
        <span class="wv">${board.vorp[pid].toFixed(0)}</span>
        <span class="op ${survClass(probe(pid))}">${pct(probe(pid))}</span></div>`).join('');
      return `<div class="wsec"><div class="whd">${title}
          <span class="dim">${hits.length} player${hits.length === 1 ? '' : 's'}</span></div>
        <div class="dim small pad0">${sub}</div>
        ${warn || ''}
        ${rows || '<div class="dim pad0">nothing at ' + Math.round(WAIT_P * 100)
          + '% — the board thins out before then</div>'}
        ${hits.length > WAIT_N ? `<div class="dim pad0">+${hits.length - WAIT_N} more</div>` : ''}
      </div>`;
    };

    const out = [];
    if (turns.length > 1) {
      out.push(section(`Still there at #${turns[1]}`,
        `your next pick &middot; ${sim.nSims} sims, +/-${moe(sim.nSims)}pp`,
        (pid) => {
          const v = survOf(pid);
          return v === null ? -1 : v;
        }));
    }
    if (turns.length > 2 && turns[2] === turns[1] + 1) {
      // Turn of the snake: nothing happens between the two picks, so the second
      // list would be a copy of the first. Say that instead of printing it
      // twice and letting it look like a coincidence.
      out.push(`<div class="wsec"><div class="whd">Still there at #${turns[2]}</div>
        <div class="dim small pad0">back-to-back with #${turns[1]} — no one picks in between,
        so the same players are there. Your first real wait is #${turns[3] || '—'}.</div></div>`);
    } else if (turns.length > 2 && sim.p2) {
      out.push(section(`Still there at #${turns[2]}`,
        `the pick after &middot; ${sim.p2Paths} paths, +/-${moe(sim.p2Paths)}pp`,
        (pid) => {
          const v = surv2Of(pid);
          return v === null ? -1 : v;
        },
        '<div class="dim small pad0 proj">read as an upper bound: fewer paths, a calibration '
        + 'fitted at one turn, and it assumes you take nobody in between.</div>'));
    }
    el('wait').innerHTML = out.join('') ||
      '<div class="dim pad">No further turns.</div>';
  }

  /* ------------------------------------------------------- the player card */

  /* The lineup you would field if you took him, slot by slot, with the plan for
   * the turns after. Everything here is read out of the same engine the
   * recommendation uses — no second opinion.
   *
   * Opened from the row's `fit` button, never from the row itself: the row's
   * click drafts, and that is the action being used nine times out of ten. */
  function inspect(pid) {
    if (pid === null || pid === undefined || pid < 0 || pid >= board.n) return;
    const seat = subjectSeat(pid);
    const sm = simFor(seat);
    const roster = st.rosters[seat];
    const taken = !!st.taken[pid];
    const p = board.pos[pid];
    const before = E.lineupValue(board, spec, roster);
    const slots = E.lineupSlots(board, spec, roster, taken ? null : pid);
    const after = slots.reduce((a, s) => a + s.value, 0);
    const gap = edgeOf(pid);

    const rows = slots.map((s) => {
      const nm = s.pid < 0
        ? '<span class="proj">empty — streamed</span>'
        : `${esc(board.name[s.pid])}${s.probe ? ' <span class="you">NEW</span>' : ''}`;
      return `<tr class="${s.probe ? 'probe' : ''}${s.pid < 0 ? ' hole' : ''}">
        <td class="slab">${s.label}</td><td>${nm}</td>
        <td class="num">${s.value.toFixed(1)}</td></tr>`;
    }).join('');

    // What the rest of the draft looks like from here, and what the turn is
    // worth. Both come straight from the DP the recommendation ranks on.
    let planHtml = '', evHtml = '', capHtml = '';
    const pl = planFor(seat, sm);
    if (pl && !taken) {
      const step = pl.space.stepTo[pl.i0 * spec.nPos + p];
      const capped = step < 0;
      const a = E.planValueAt(pl, capped ? pl.i0 : step);
      const idle = E.planValueAt(pl, pl.i0);
      const now = after - before;
      const path = E.planPath(pl, capped ? pl.i0 : step).filter((s) => s.pos >= 0).slice(0, 6);
      evHtml = `<div class="ilrow">adds now <b>${now >= 0 ? '+' : ''}${now.toFixed(1)}</b>
        &middot; rest of the draft <b>+${a.v.toFixed(1)}</b>
        &middot; ${seat === ui.seat ? 'you' : 'he'} would finish at
        <b>${(before + now + a.v).toFixed(1)}</b>
        <span class="dim">(passing this turn: ${(before + idle.v).toFixed(1)})</span></div>`;
      if (capped) {
        capHtml = `<div class="warn pad small">You are at the roster cap for ${posName(p)},
          so this is a bench pick — it cannot change the lineup you finish with.</div>`;
      }
      if (a.f > idle.f + 1e-9) {
        capHtml += `<div class="warn pad small">Taking him leaves ${a.f.toFixed(1)} mandatory
          slot(s) the plan cannot fill, against ${idle.f.toFixed(1)} if you pass.</div>`;
      }
      planHtml = path.length
        ? `<div class="ilrow dim">then the plan wants ${path.map((s) =>
            `<span class="${s.modelled ? '' : 'proj'}">#${s.turn} ${posName(s.pos)}</span>`).join(' → ')}</div>`
        : '';
    }

    const onClock = E.seatOnClock(st);
    let draftHtml = '';
    if (!taken && onClock >= 0) {
      const legal = legalNow();
      if (!legal || legal.has(posName(p))) {
        draftHtml = `<div class="ilrow">
        <button class="idraft" data-pid="${pid}">draft ${esc(board.name[pid])}</button>
        <span class="dim small">to ${esc(label(ui.managers[onClock]))}</span></div>`;
      } else {
        const capped = E.openPositions(st, onClock).indexOf(p) < 0;
        draftHtml = `<div class="warn pad small">${esc(label(ui.managers[onClock]))} cannot draft
        a ${posName(p)} right now: ${capped ? 'the position is at its roster cap'
          : 'every remaining pick is owed to an open starting slot'}.</div>`;
      }
    }
    const mine = seat === ui.seat;
    const who = mine ? 'Your' : esc(label(ui.managers[seat])) + '&rsquo;s';
    const pn = sv(sm, pid), p2n = sv2(sm, pid);
    const turns = (sm && sm.turns) || [];
    el('inspect').innerHTML = `<div class="ibx">
      <div class="ihd">${tag(p)} <b>${esc(board.name[pid])}</b>
        <span class="dim">${esc(board.team[pid])}${board.bye[pid] ? ' · bye ' + board.bye[pid] : ''}</span>
        <span class="sp"></span><button class="ix">&times;</button></div>
      <div class="ilrow dim">ADP ${board.adpRank[pid]}
        &middot; ECR ${board.hasEcr[pid] ? board.ecr[pid] : '—'}
        ${gap === null ? '' : `&middot; edge ${gap > 0 ? '+' : ''}${gap}`}
        &middot; VORP <b>${board.vorp[pid].toFixed(0)}</b>${
          isFinite(board.vorpSe[pid]) && board.vorpSe[pid] > 0
            ? ` <span class="proj">±${board.vorpSe[pid].toFixed(0)}</span>` : ''}
        ${pn === null || !turns[1] ? ''
          : `&middot; there at #${turns[1]} <b class="${survClass(pn)}">${pct(pn)}</b>`}
        ${p2n === null || !turns[2] ? ''
          : `&middot; at #${turns[2]} <b class="${survClass(p2n)}">${pct(p2n)}</b>`}</div>
      ${taken ? `<div class="warn pad small">Already ${mine ? 'on your roster'
        : 'drafted by ' + esc(label(ui.managers[seat]))} — showing that roster as it
        stands.</div>` : ''}
      ${capHtml}
      <h3>${who} starting lineup${taken ? '' : (mine ? ' if you take him' : ' if he takes him')}</h3>
      <table class="slots"><tbody>${rows}</tbody></table>
      <div class="ilrow">lineup <b>${after.toFixed(1)}</b>
        <span class="dim">from ${before.toFixed(1)}</span></div>
      ${evHtml}${planHtml}
      ${draftHtml}
      <div class="ilrow dim small">An empty slot is priced at what you would actually field
        there, not at zero — the best player who goes undrafted, off the 2021-25 flow. At QB
        and TE that price is already zero: their VORP is measured against streaming to begin
        with. That is why filling an RB or WR hole with a below-replacement player is worth
        something, and why leaving the QB slot open costs almost nothing.</div>
    </div>`;
    el('inspect').hidden = false;
  }

  function closeInspect() { el('inspect').hidden = true; el('inspect').innerHTML = ''; }

  /* -------------------------------------------------------- the draft board */

  /* The grid every draft room has on the wall: a column per team in draft
   * order, a row per round, the pick in the cell. It answers the questions the
   * player list cannot — who has been hoarding backs, whether the run you are
   * worried about already happened, what the wrap looks like from your seat.
   *
   * It reads `st`, never `ui.picks`: rebuild() drops duplicate pids without
   * advancing, so the two can be different lengths and only st.picks is aligned
   * with st.seats. A cell is drawn from its PICK INDEX for the same reason —
   * the seat order is whatever st.seats says, not an assumed snake, so a
   * reordered league or a non-snake format draws correctly with no extra code.
   *
   * Cells open the player card (they never draft). Drafting out of turn is the
   * one thing this view must not make easy: the cell you would click is a
   * future pick belonging to someone else. */
  let gridOpen = false;

  /** overall pick number -> "3.05", the way a draft room says it */
  function pickLabel(i) {
    return (Math.floor(i / ui.teams) + 1) + '.' + String((i % ui.teams) + 1).padStart(2, '0');
  }

  function gridCell(i, s) {
    const mine = s === ui.seat ? ' mycol' : '';
    if (i === null) return `<td class="gc${mine}"></td>`;
    const lab = pickLabel(i);
    if (i >= st.picks.length) {
      const now = i === st.picks.length;
      return `<td class="gc${mine}${now ? ' now' : ''}">
        <div class="gm"><span>${lab}</span></div>
        ${now ? '<div class="gn dim">on the clock</div>' : ''}</td>`;
    }
    const pid = st.picks[i];
    if (pid === null) {
      return `<td class="gc off${mine}"><div class="gm"><span>${lab}</span></div>
        <div class="gn">off board</div></td>`;
    }
    const p = board.pos[pid];
    // Where the pick landed against the market: positive = he was still there
    // this many slots past his ADP. The same quantity the EDGE column reports,
    // measured against the pick actually spent instead of against ECR.
    const d = (i + 1) - board.adpRank[pid];
    const dCls = d >= 8 ? 'v' : d <= -8 ? 'r' : '';
    return `<td class="gc g-${posCls(p)}${mine}${simmedPids.has(pid) ? ' simmed' : ''}"
      data-pid="${pid}" title="pick ${i + 1} &middot; ADP ${board.adpRank[pid]} &middot; ECR ${
        board.hasEcr[pid] ? board.ecr[pid] : '—'}">
      <div class="gm"><span>${lab}</span><span class="sp"></span>
        <span class="gd ${dCls}">${d > 0 ? '+' : ''}${d}</span></div>
      <div class="gn">${esc(board.name[pid])}</div>
      <div class="gm"><span>${posName(p)} &middot; ${esc(board.team[pid])}</span>
        <span class="sp"></span><span>${board.vorp[pid].toFixed(0)}</span></div></td>`;
  }

  function renderGrid() {
    const teams = ui.teams, rounds = ui.rounds;
    const at = [];                       // at[round][seat] = overall pick index
    for (let r = 0; r < rounds; r++) at.push(new Array(teams).fill(null));
    for (let i = 0; i < st.seats.length; i++) {
      const r = Math.floor(i / teams), s = st.seats[i] - 1;
      if (r < rounds && s >= 0 && s < teams) at[r][s] = i;
    }

    const head = [];
    for (let s = 0; s < teams; s++) {
      const nm = esc(label(ui.managers[s]));
      head.push(`<th class="${s === ui.seat ? 'mine' : ''}" title="${nm}">${nm}${
        s === ui.seat ? ' <span class="you">YOU</span>' : ''}</th>`);
    }

    const rows = [];
    for (let r = 0; r < rounds; r++) {
      const i0 = r * teams;
      // read the direction off the seat list rather than off r % 2, so the
      // gutter still tells the truth if the format is ever not a snake
      const dir = teams > 1 && st.seats[i0 + 1] !== undefined
        ? (st.seats[i0] < st.seats[i0 + 1] ? '&rarr;' : '&larr;') : '';
      const tds = [];
      for (let s = 0; s < teams; s++) tds.push(gridCell(at[r][s], s));
      rows.push(`<tr><td class="rlab">${r + 1}<div class="dirn">${dir}</div></td>
        ${tds.join('')}</tr>`);
    }

    // Per column: what he has taken, and what his lineup is worth as it stands.
    // The lineup number is E.lineupValue — the same one the player card and
    // every EV on the page are built from, empty slots priced at what he would
    // actually stream there rather than at zero.
    const foot = [];
    for (let s = 0; s < teams; s++) {
      const counts = M.positions.map((nm, p) => [nm, st.counts[s][p]])
        .filter(([, n]) => n > 0).map(([nm, n]) => `${n}${nm}`).join(' ');
      const lv = E.lineupValue(board, spec, st.rosters[s] || []);
      foot.push(`<td><span class="gcount">${counts || '&mdash;'}</span>
        <span class="lv">${lv.toFixed(0)}</span></td>`);
    }

    const seat = E.seatOnClock(st);
    const done = st.picks.length;
    el('grid').innerHTML = `<div class="gbx">
      <div class="ghd"><b>Draft board</b>
        <span class="dim">${done} of ${teams * rounds} picks made${seat < 0 ? ''
          : ' &middot; ' + esc(label(ui.managers[seat])) + ' on the clock at '
            + pickLabel(done)}</span>
        <span class="sp"></span><button class="ix">&times;</button></div>
      <div class="gsc"><table class="gt">
        <thead><tr><th class="rlab">rd</th>${head.join('')}</tr></thead>
        <tbody>${rows.join('')}</tbody>
        <tfoot><tr><td class="rlab"></td>${foot.join('')}</tr></tfoot>
      </table></div>
      <div class="gleg">
        ${['QB', 'RB', 'WR', 'TE'].map((n) =>
          `<span><span class="sw" style="background:var(--pos-${n.toLowerCase()})"></span>${n}</span>`).join('')}
        <span><b class="safe">+n</b> / <b class="gone">&minus;n</b> = slots past
          (or before) his ADP</span>
        <span>bottom number = VORP &middot; footer = starting lineup as it stands</span>
        <span>click a pick for the player card &middot; <b>esc</b> closes</span>
      </div></div>`;
  }

  function openGrid() { gridOpen = true; el('grid').hidden = false; renderGrid(); }
  function closeGrid() { gridOpen = false; el('grid').hidden = true; el('grid').innerHTML = ''; }

  /* ---------------------------------------------------------------- events */

  function draft(pid) {
    if (st.taken[pid]) return;
    const legal = legalNow();
    if (legal && !legal.has(posName(board.pos[pid]))) return;
    ui.picks.push(pid); ui.simmed.push(0); save(); rebuild(); render();
  }

  /* Skip ahead: the model makes every pick until the stop condition — either
   * {stopSeat} (you are on the clock) or {stopPick} (that pick is up next,
   * simming through your own turns if the target passes them). The whole jump
   * is appended before ONE save/rebuild/render, and every appended entry
   * shares one sim id so "undo sim" can rewind it wholesale. */
  function simJump(stop) {
    const probe = E.forkState(st);           // an error cannot corrupt st
    const added = E.fastForward(M, probe, Object.assign({
      sample: ui.simSample !== false,
      seedBase: Date.now() >>> 0,            // re-rolls on undo-then-resim
      // you are at the keyboard, so 'prior' autopilot odds do not apply to
      // your own seat; an explicit auto/human setting on it is still honored
      autoWeight: (s) => (s === ui.seat && ui.autoMode[s] === 'prior') ? 0 : autoWeightFor(s),
    }, stop));
    if (!added.length) return;
    const id = ui.simmed.reduce((m, x) => Math.max(m, x), 0) + 1;
    for (const pid of added) { ui.picks.push(pid); ui.simmed.push(id); }
    save(); rebuild(); render();
  }

  /** "37" or round.pick "3.05" → overall pick number; null when out of range
   *  or not ahead of the pick on the clock. */
  function parsePickTarget(text) {
    const t = text.trim();
    const total = ui.teams * ui.rounds;
    let n = null;
    const m = t.match(/^(\d+)\.(\d+)$/);
    if (m) {
      const r = +m[1], p = +m[2];
      if (r >= 1 && r <= ui.rounds && p >= 1 && p <= ui.teams) n = (r - 1) * ui.teams + p;
    } else if (/^\d+$/.test(t)) n = +t;
    return (n !== null && n > E.pickNo(st) && n <= total) ? n : null;
  }

  el('board').addEventListener('click', (e) => {
    const th = e.target.closest('th.sortable');
    if (th) {
      const key = th.dataset.key;
      ui.sort = ui.sort.key === key ? { key, dir: -ui.sort.dir } : { key, dir: 1 };
      save(); renderBoard();
      return;
    }
    const tr = e.target.closest('tr[data-pid]');
    if (!tr) return;
    const pid = Number(tr.dataset.pid);
    // the fit button is the ONLY thing on a row that does not draft
    if (e.target.closest('.fit')) { inspect(pid); return; }
    draft(pid);
  });

  /* Drag a seat number to change the draft order. The order is often wrong
   * until ESPN publishes it, and retyping ten managers is not a fix. Picks
   * already recorded stay where they are in the sequence, so they reassign to
   * whoever now holds that slot — which is the point, and is undone by dragging
   * back. `ui.seat` follows the person, not the slot. */
  let dragSeat = null, overRow = null;
  const clearOver = () => {
    if (overRow) overRow.classList.remove('dropb', 'dropa');
    overRow = null;
  };

  function moveSeat(from, to, before) {
    if (from === to) return;
    const order = [];
    for (let s = 0; s < ui.teams; s++) if (s !== from) order.push(s);
    let ti = order.indexOf(to);
    if (ti < 0) ti = order.length; else if (!before) ti += 1;
    order.splice(ti, 0, from);
    ui.managers = order.map((s) => ui.managers[s]);
    ui.autoMode = order.map((s) => ui.autoMode[s]);
    ui.seat = order.indexOf(ui.seat);
    save(); rebuild(); render();
  }

  el('league').addEventListener('dragstart', (e) => {
    const tr = e.target.closest('tr[data-seat]');
    if (!tr || !e.target.closest('.hdl')) { e.preventDefault(); return; }
    dragSeat = Number(tr.dataset.seat);
    tr.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    try { e.dataTransfer.setData('text/plain', String(dragSeat)); } catch (err) {}
  });
  el('league').addEventListener('dragover', (e) => {
    if (dragSeat === null) return;
    const tr = e.target.closest('tr[data-seat]');
    if (!tr) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const r = tr.getBoundingClientRect();
    const before = (e.clientY - r.top) < r.height / 2;
    if (overRow !== tr) clearOver();
    overRow = tr;
    tr.classList.toggle('dropb', before);
    tr.classList.toggle('dropa', !before);
  });
  el('league').addEventListener('drop', (e) => {
    if (dragSeat === null) return;
    const tr = e.target.closest('tr[data-seat]');
    e.preventDefault();
    const before = overRow === tr && tr.classList.contains('dropb');
    clearOver();
    const src = dragSeat; dragSeat = null;
    if (tr) moveSeat(src, Number(tr.dataset.seat), before);
  });
  el('league').addEventListener('dragend', () => {
    dragSeat = null; clearOver();
    const d = el('league').querySelector('.dragging');
    if (d) d.classList.remove('dragging');
  });

  // a name in any of the read-only panels opens the card; none of them draft,
  // so there is nothing to disambiguate there
  for (const id of ['recommend', 'upcoming', 'wait', 'roster']) {
    el(id).addEventListener('click', (e) => {
      const t = e.target.closest('[data-pid]');
      if (t) inspect(Number(t.dataset.pid));
    });
  }
  el('inspect').addEventListener('click', (e) => {
    const go = e.target.closest('.idraft');
    if (go) { const pid = Number(go.dataset.pid); closeInspect(); draft(pid); return; }
    if (e.target.closest('.ibx') && !e.target.closest('.ix')) return;
    closeInspect();
  });
  // a pick in the grid opens the card, exactly like a name in the side panels;
  // nothing in this view drafts
  el('grid').addEventListener('click', (e) => {
    if (e.target.closest('.ix')) { closeGrid(); return; }
    const td = e.target.closest('.gc[data-pid]');
    if (td) { inspect(Number(td.dataset.pid)); return; }
    if (!e.target.closest('.gbx')) closeGrid();
  });
  el('gridbtn').addEventListener('click', () => (gridOpen ? closeGrid() : openGrid()));

  document.addEventListener('keydown', (e) => {
    // esc peels one layer: the card sits on top of the grid, so it goes first
    if (e.key === 'Escape') {
      if (!el('inspect').hidden) closeInspect();
      else closeGrid();
      return;
    }
    if (e.key !== 'b' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable) return;
    gridOpen ? closeGrid() : openGrid();
  });

  // undo stays strictly single-pick, even into a simmed tail (it just shrinks
  // the jump): it is the panic key during a live draft, and a mis-tap must
  // never silently rewind forty picks. "undo sim" is the wholesale rewind.
  el('undo').addEventListener('click', () => {
    ui.picks.pop(); ui.simmed.pop(); save(); rebuild(); render();
  });
  el('offboard').addEventListener('click', () => {
    ui.picks.push(null); ui.simmed.push(0); save(); rebuild(); render();
  });
  el('reset').addEventListener('click', () => {
    if (!confirm('Clear all picks? Seats and managers are kept.')) return;
    ui.picks = []; ui.simmed = []; save(); rebuild(); render();
  });
  el('simnext').addEventListener('click', () => simJump({ stopSeat: ui.seat }));
  function simGo() {
    const n = parsePickTarget(el('simto').value);
    if (n === null) { el('simto').classList.add('bad'); return; }
    el('simto').value = '';
    simJump({ stopPick: n });
  }
  el('simgo').addEventListener('click', simGo);
  el('simto').addEventListener('keydown', (e) => { if (e.key === 'Enter') simGo(); });
  el('simto').addEventListener('input', (e) => e.target.classList.remove('bad'));
  el('simsample').checked = ui.simSample !== false;   // once — the bar never re-renders
  el('simsample').addEventListener('change', (e) => {
    ui.simSample = e.target.checked; save();
  });
  el('simundo').addEventListener('click', () => {
    const id = ui.simmed.length ? ui.simmed[ui.simmed.length - 1] : 0;
    if (!id) return;
    while (ui.simmed.length && ui.simmed[ui.simmed.length - 1] === id) {
      ui.picks.pop(); ui.simmed.pop();
    }
    save(); rebuild(); render();
  });
  el('search').addEventListener('input', (e) => {
    query = e.target.value.toLowerCase().trim(); renderBoard();
  });
  el('search').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const first = el('board').querySelector('tr[data-pid]');
    if (first) {
      e.target.value = ''; query = '';
      draft(Number(first.dataset.pid));
    }
  });
  el('filters').addEventListener('click', (e) => {
    if (!e.target.dataset.pos) return;
    filterPos = e.target.dataset.pos;
    [...el('filters').children].forEach((c) => c.classList.toggle('on', c === e.target));
    renderBoard();
  });
  el('league').addEventListener('change', (e) => {
    if (!e.target.classList.contains('automode')) return;
    ui.autoMode[Number(e.target.dataset.seat)] = e.target.value;
    save(); render();
  });
  el('seat').addEventListener('change', (e) => {
    ui.seat = Number(e.target.value); save(); render();
  });

  rebuild();
  render();
})();
