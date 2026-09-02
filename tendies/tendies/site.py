"""Render the draft-day page: one self-contained HTML file.

House pattern borrowed from the parent repo's `bff/site.py` — no CDN, no web
fonts, no libraries, an inline data blob and vanilla JS — but this page is
generated here and writes only into `tendies/docs/`. Nothing in `bff/` or the
published site is touched.

The JS lives in real files under `web/` rather than in Python string constants,
which is the one deliberate departure. Keeping it in strings would make the
Python/JS parity test impossible to run, and that test is the only thing
standing between "the model is right" and "the page is right".
"""

from __future__ import annotations

import json
from pathlib import Path

CSS = """
:root{
  --bg:#fbfaf8; --panel:#fff; --text:#1a1a19; --dim:#6b6a67; --faint:#9d9b97;
  --rule:#e6e3dd; --rule-strong:#cfcbc3; --accent:#1f5f4a;
  --pos-qb:#7c4dbe; --pos-rb:#1f7a4d; --pos-wr:#1f5f9e; --pos-te:#b06a1f;
  --pos-k:#8a8a8a; --pos-dst:#6b6a67;
  --safe:#1f7a4d; --risky:#b06a1f; --gone:#b03a2e;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 var(--sans)}
h1{font-size:15px;margin:0;letter-spacing:.02em}
.wrap{display:grid;grid-template-columns:256px 1fr 340px;grid-template-rows:auto 1fr;
  gap:10px;padding:10px;height:100vh}
.bar{grid-column:1/-1;display:flex;align-items:center;gap:12px;background:var(--panel);
  border:1px solid var(--rule);border-radius:6px;padding:8px 12px}
.bar .sp{flex:1}
button,select,input{font:12px var(--sans);border:1px solid var(--rule-strong);
  background:var(--panel);color:var(--text);border-radius:4px;padding:4px 8px;cursor:pointer}
button:hover{border-color:var(--accent)}
input{cursor:text}
.col{display:flex;flex-direction:column;gap:10px;min-height:0}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  display:flex;flex-direction:column;min-height:0;overflow:hidden}
.panel>h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
  margin:0;padding:8px 10px;border-bottom:1px solid var(--rule);font-weight:600}
.panel>div:not(.tools){overflow:auto;min-height:0}
.panel>.tools{flex:0 0 auto}
table{width:100%;border-collapse:collapse;font-size:12px}
th{position:sticky;top:0;background:var(--panel);text-align:left;color:var(--faint);
  font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.06em;
  padding:6px 8px;border-bottom:1px solid var(--rule)}
td{padding:4px 8px;border-bottom:1px solid var(--rule)}
tbody tr{cursor:pointer}
tbody tr:hover{background:#f2efe9}
.num{text-align:right;font-family:var(--mono)}
.nm{font-weight:500}.rk{font-size:9px;font-weight:700;vertical-align:super;opacity:.65;letter-spacing:.5px}
.dim{color:var(--dim)}
.small{font-size:11px}
.pad{padding:8px 10px}
.tag{display:inline-block;min-width:30px;text-align:center;font-size:10px;font-weight:600;
  padding:1px 4px;border-radius:3px;color:#fff}
.pos-qb{background:var(--pos-qb)}.pos-rb{background:var(--pos-rb)}
.pos-wr{background:var(--pos-wr)}.pos-te{background:var(--pos-te)}
.pos-k{background:var(--pos-k)}.pos-dst{background:var(--pos-dst)}
.surv.safe{color:var(--safe)}.surv.risky{color:var(--risky)}.surv.gone{color:var(--gone)}
b.safe{color:var(--safe)}b.risky{color:var(--risky)}b.gone{color:var(--gone)}
tr.onclock{background:#f0f6f3}
tr.mine td:nth-child(2){font-weight:700}
.you{background:var(--accent);color:#fff;font-size:10px;padding:1px 5px;border-radius:3px}
#roster ul{margin:0;padding:6px 10px;list-style:none}
#roster li{padding:2px 0}

/* sim / skip-ahead controls and provenance badges */
#simto{width:110px}
input.bad{border-color:var(--gone)}
button:disabled{opacity:.45;cursor:default;border-color:var(--rule)}
#roster li.sim{color:var(--dim)}
#roster li.sim::after{content:"sim";font-size:9px;font-weight:600;letter-spacing:.04em;
  color:var(--faint);border:1px solid var(--rule);border-radius:3px;padding:0 3px;margin-left:5px}
.card{border-bottom:1px solid var(--rule);padding:7px 10px}
.card-hd{font-weight:600;font-size:12px;margin-bottom:4px}
.badge{font-size:9px;padding:1px 4px;border-radius:3px;margin-left:4px;
  text-transform:uppercase;letter-spacing:.04em}
.badge.auto{background:#f3e3c8;color:#7a4f10}
.badge.pooled{background:#e9e7e2;color:var(--dim)}
.opt{position:relative;padding:2px 0;display:flex;align-items:center;gap:6px;font-size:12px}
.opt .bar{position:absolute;left:0;top:0;bottom:0;background:#eef3f0;z-index:0;border:0;padding:0}
.opt>*{position:relative;z-index:1}
.opt .onm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.opt .op{font-family:var(--mono);color:var(--dim)}
.mix{font-size:10px;margin-top:3px}
.rec{border-bottom:1px solid var(--rule);padding:7px 10px}
.rec-hd{display:flex;align-items:center;gap:6px}
.rec-hd .rk{color:var(--faint);font-family:var(--mono);width:14px}
.rec-hd .rk-tie{width:18px}
.rec-bd{font-size:11px;margin-top:2px;font-family:var(--mono)}
/* A tie is not a ranking. Where the VORP curve cannot separate the top of the
   list the panel says so and shares one rank number across the tied rows, so
   the reader is never handed a 1-2-3 the data does not support. */
.tie-note{background:#eef1f6;color:#2f3f57;border-bottom:1px solid var(--rule)}
/* the plan: what the recommendation intends for the REST of the draft. Turns
   the simulation covers are solid; the ones projected from the historical
   positional flow are greyed, so the page never presents the two as equally
   well known. */
.planline{padding:5px 10px;border-bottom:1px solid var(--rule);font-family:var(--mono)}
.planline .proj,.rec-bd .proj{color:var(--faint)}
.warn{background:#f7ece0;color:#7a4f10;border-bottom:1px solid var(--rule)}
#panels.computing{opacity:.55}
#filters button{padding:2px 7px;font-size:11px}
#filters button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tools{display:flex;gap:6px;padding:7px 10px;border-bottom:1px solid var(--rule);
  align-items:center;flex-wrap:wrap}
#search{flex:1;min-width:90px}
footer{grid-column:1/-1;color:var(--faint);font-size:11px;padding:0 4px}

/* sortable headers */
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--text)}
th.sorted{color:var(--accent)}
.arw{font-size:8px;margin-left:3px;vertical-align:1px}

/* drag a seat number to change the draft order (League panel only — the board
   rows draft on click, so nothing there is draggable) */
#league td.hdl{cursor:grab}
#league td.hdl:active{cursor:grabbing}
#league tr:hover td.hdl{color:var(--accent);font-weight:600}
tr.dragging{opacity:.4}
tr.dropb td{box-shadow:inset 0 2px 0 var(--accent)}
tr.dropa td{box-shadow:inset 0 -2px 0 var(--accent)}

/* the one thing on a board row that does not draft */
td.fitc{width:30px;padding:2px 6px 2px 0;text-align:right}
/* faint but always visible: an invisible button on a row whose click drafts is
   a trap, not a clean design */
.fit{font-size:10px;padding:1px 5px;color:var(--faint);border-color:var(--rule)}
tr:hover .fit{color:var(--dim);border-color:var(--rule-strong)}
.fit:hover{color:var(--accent);border-color:var(--accent)}
.onm[data-pid]{cursor:pointer}
.onm[data-pid]:hover{text-decoration:underline dotted;text-underline-offset:2px}
#roster li[data-pid]{cursor:pointer}
#roster li[data-pid]:hover{background:#f2efe9}

/* "safe to wait on" */
.wsec{border-bottom:1px solid var(--rule);padding:6px 10px}
.whd{font-weight:600;font-size:12px;display:flex;gap:6px;align-items:baseline}
.whd .dim{font-weight:400;font-size:11px}
.pad0{padding:1px 0}
.wr{display:flex;align-items:center;gap:6px;padding:2px 0;font-size:12px}
.wr .onm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.wr .wv{font-family:var(--mono);color:var(--faint);font-size:11px}
.wr .op{font-family:var(--mono);min-width:34px;text-align:right}

/* the player card: what your roster looks like if you take him */
#inspect{position:fixed;inset:0;background:rgba(26,26,25,.42);z-index:50;
  display:flex;align-items:center;justify-content:center;padding:20px}
#inspect[hidden]{display:none}
.ibx{background:var(--panel);border:1px solid var(--rule-strong);border-radius:8px;
  width:min(520px,100%);max-height:88vh;overflow:auto;
  box-shadow:0 12px 40px rgba(0,0,0,.18)}
.ihd{display:flex;align-items:center;gap:6px;padding:10px 12px;
  border-bottom:1px solid var(--rule);font-size:14px}
.ihd .sp{flex:1}
.ix{border:0;background:none;font-size:18px;line-height:1;color:var(--faint);padding:0 4px}
.ix:hover{color:var(--text)}
.ibx h3{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
  margin:10px 12px 4px;font-weight:600}
.ilrow{padding:4px 12px}
.slots{width:100%}
.slots td{padding:3px 12px}
.slots .slab{font-family:var(--mono);color:var(--dim);width:52px}
.slots tr.probe{background:#eef3f0}
.slots tr.probe td{font-weight:600}
.slots tr.hole td{color:var(--dim)}
.idraft{background:var(--accent);color:#fff;border-color:var(--accent);
  font-weight:600;padding:5px 10px;margin-right:6px}
.idraft:hover{filter:brightness(1.12)}
"""


def render(payload: dict, web: Path, out: Path) -> Path:
    engine = (web / "engine.js").read_text()
    page = (web / "page.js").read_text()
    blob = json.dumps(payload, separators=(",", ":"))
    meta = payload.get("meta", {})
    note = meta.get("note", "")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tendies &middot; {meta.get('season', '')} draft</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='13'>&#127944;</text></svg>">
<style>{CSS}</style></head><body>
<div class="wrap">
  <div class="bar">
    <h1>tendies</h1>
    <span id="status" class="small"></span>
    <span class="sp"></span>
    <label class="small dim">you are&nbsp;<select id="seat"></select></label>
    <button id="simnext" title="model picks every seat until you are on the clock">sim to my pick</button>
    <input id="simto" placeholder="to pick, e.g. 37 or 3.05" autocomplete="off">
    <button id="simgo">go</button>
    <label class="small dim"><input type="checkbox" id="simsample"> sampled</label>
    <button id="simundo" title="rewind the whole last sim jump">undo sim</button>
    <button id="undo">undo</button>
    <button id="offboard">off-board</button>
    <button id="reset">reset</button>
  </div>

  <div class="col">
    <div class="panel" style="flex:0 0 auto">
      <h2>League &mdash; drag a number to fix the draft order</h2><div id="league"></div>
    </div>
    <div class="panel" style="flex:0 0 auto">
      <h2>Your roster</h2><div id="roster"></div>
    </div>
    <div class="panel">
      <h2>Safe to wait on</h2><div id="wait"></div>
    </div>
  </div>

  <div class="panel">
    <h2>Board &mdash; click a player to draft him; sort on any header; <em>fit</em> shows his roster fit</h2>
    <div class="tools">
      <input id="search" placeholder="search, Enter drafts the top row" autocomplete="off">
      <span id="filters">
        <button data-pos="ALL" class="on">ALL</button><button data-pos="QB">QB</button>
        <button data-pos="RB">RB</button><button data-pos="WR">WR</button>
        <button data-pos="TE">TE</button><button data-pos="K">K</button>
        <button data-pos="DST">DST</button>
      </span>
    </div>
    <div id="board"></div>
  </div>

  <div class="col" id="panels">
    <div class="panel">
      <h2>Upcoming picks &mdash; most likely 3 each</h2><div id="upcoming"></div>
    </div>
    <div class="panel">
      <h2>Take now</h2><div id="recommend"></div>
    </div>
  </div>

  <footer>{note}</footer>
</div>
<div id="inspect" hidden></div>
<script>window.TENDIES={blob};</script>
<script>{engine}</script>
<script>{page}</script>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
