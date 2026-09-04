# tendies

Draft-day prediction for an ESPN fantasy football league. Two layers:

1. **Extraction** — pull every season's draft, link the teams year over year
   into stable franchises, and join each drafted player to that season's
   market ADP.
2. **Prediction** — a nested-logit choice model over the live draft state that
   answers "who does this manager take next", plus a Monte-Carlo simulation of
   who survives to your next turn, rendered as a self-contained draft-day page.
3. **Recommendation** — what *you* should do about it: a dynamic program over
   your remaining turns that maximises the lineup you will finish the draft
   with, not the one you hold.

Results and method: **[REPORT.md](REPORT.md)**. Short version: the model gets
**20.0% top-1 / 44.0% top-3** against a best-available-by-ADP baseline of
15.3% / 32.0%, and survival probabilities score Brier 0.092 against a 0.171
base rate. Nine per-manager channels were built and **six ship**: reaching past
ADP, following the expert list, appetite for rookies, age preference, and the
two positional ones. Stacking and re-drafting your own players priced at zero
and last season's durability was actively harmful, so all three are
implemented, exported and switched off. Even the six point two ways at once —
they buy 0.015 nats of likelihood while costing 0.4pp of top-1 — so the signal
is still mostly in the draft state, not the personalities. (An earlier headline said tendencies were worth +0.33pp
of top-1; that was measured with a broken switch that left one channel fitted
while reporting it off — see REPORT.) The
recommendation fills every starting slot from all ten seats and beats
cap-aware best-available — by expert value and by market price — on 10 of 10.

Self-contained project (its own `pyproject.toml` and `.venv`); it reads the
parent `betterffrank` repo only for the ADP boards, and those paths are flags.

## Use

```bash
cd tendies
uv sync

# once per machine: opens Chrome, you sign in to ESPN, cookies land in .auth/
# (git-ignored). Private leagues answer 401 without them.
uv run python -m tendies login

# fetch -> tidy -> link franchises -> join ADP -> write data/processed/
uv run python -m tendies build
uv run python -m tendies report           # tendency cuts off the dataset
uv run python -m tendies traits           # per-manager traits + reliability
uv run python -m tendies ecr-check        # audit the expert ranks before pinning them
uv run python -m tendies streaming        # derive each position's replacement level

# the model
uv run python -m tendies model            # fit; prints coefficients
uv run python -m tendies evaluate --b4 --ablate   # walk-forward vs baselines,
                                          # and what each manager channel buys
uv run python -m tendies hints            # resolve boone/etr ids (writes data/raw/*.json)
uv run python -m tendies site --season 2026   # -> docs/index.html
uv run pytest tests/                      # incl. Python/JS parity
```

Open `docs/index.html` in any browser. Click a player to draft him to whoever
is on the clock; the board shows P(available at your next turn), the right
panel shows each upcoming manager's three most likely picks, and the bottom
panel ranks candidates by the lineup you would *finish* with — what each adds
now plus the best plan for every turn after this one — and shows the plan
itself, so you can see why a receiver now is fine when the back is coming at
your next turn. Once a pick can no longer improve the lineup it is labelled and
scored as a bench pick instead (insurance plus market edge, never as lineup
points).

**That panel is always about YOUR pick**, which most of the time is not the
pick on the clock. Off the clock it retitles itself `Your pick — #43` and is
computed on a probe: the model plays every seat in between at its argmax (the
same `fastForward` behind the *sim to my pick* button, seeded off the state so
it is deterministic) and the recommendation is priced at your turn on the board
that path leaves. That is one modal path, not a distribution — a player the
model hands to someone else drops off the list even where he had a real chance
of reaching you, and the probabilistic read on the same question is the board's
`P(next)` column and **Safe to wait on**. Until 2026-09-04 the panel instead
priced today's board "as if this pick were yours", which led with players who
would be gone (a first recommendation at `P(avail) 23%`) and, worse, silently
gave the plan **one turn too many** — `myTurns` starts at the pick on the clock
whoever owns it, so a 15-turn plan was priced as 16 and every `then` and `EV`
on screen was a full round inflated (top EV 662.8 against the correct 570.3 at
pick 1).

**The panel carries no prose.** It is the slider, the ranked rows, the plan and
a provenance footer; the four explanatory blocks it used to open with are gone,
including the `No K/DST projected available — take one now` banner (the
ordering still puts those first, so the signal survives as position in the
list). A tie still reads as a shared rank badge with an `=`.

**A card is rank, position, name, an ETR tag when there is one, and four
labelled numbers**: `VORP` (with its ±SE), `ECR`, `BOONE`, `AVAIL`. The label
sits above the value so the numbers line up as a row the eye runs across; the
old card was two wrapped lines of `key value · key value` and nothing lined up
with anything. Off it came ADP, `insurance`/`edge`, `now`/`then`/`EV`, `lose by
waiting` and the `bench` marker — so **the ranking basis is no longer on
screen**. The card shows market and expert facts plus availability, which are
the same quantities whether the engine scored the pick as a starter or as a
bench pick, and the ordering is what those two modes actually change. The
player card behind `fit` still breaks out adds-now, the plan and the finishing
lineup.

**`P(avail)<` slider** — its own row under the panel title, so a drag is not
fighting an `innerHTML` rebuild. It keeps only candidates *less* likely than
the threshold to reach the turn after the one being ranked: the picks you
stand to lose by waiting. Preset 50%, persisted, full right shows every
candidate, and the label carries the kept count (`50% · 9`, `all · 248`) so an
empty list explains itself without a text cell. It re-filters a cached list —
dragging never re-solves the plan DP — and it touches nothing but this panel.

Two things about the threshold worth knowing before you draft on it. It keeps a
useful 7-14 candidates on the **long** side of the snake and **0** on the short
side, because nobody is under 50% to survive a four-pick gap; if you want
something at every turn, 75% keeps 4-6 throughout. And the filter has to test
simulation membership before `pAvail`, not `pAvail` alone: `recommend` falls
back to `pAvail = 0` for anyone outside the simulation's watched top ~60, which
is harmless where it is used (`cost = now * (1 - pAvail)` degrades to `now`)
and exactly backwards as a filter — the first cut read every round-14 receiver
as 0% to survive and kept 273 of 332 candidates.

The first cut of it also read `league` — a `const` local to `rebuild()` — from
the new probe builder. `node --check` passes, the page loads, and the
ReferenceError lands inside a `requestAnimationFrame` callback, so the only
symptom is that the survival columns and all three sim-dependent panels come up
blank on every pick that is not yours. `tests/page.mjs` exists because of that:
it loads `page.js` against a stub DOM and drives real render cycles on and off
the clock, asserting each panel produced content and that the panel retitles
itself. It is a smoke test — it says nothing about whether the numbers are
right, which is `ties`/`plan`/`lineup`/`mockdraft`'s job — and what it catches
is a panel silently disappearing. State persists in localStorage, so a reload mid-draft costs nothing.
The board only offers what the seat on the clock can legally roster: position
caps always, and at the endgame — once every remaining pick is owed to an
unfilled starting slot — only the positions that fill one (a banner says so;
simulated picks obey the same rule).

**Board view** (`Ctrl-B`, plain `b` outside the search box, or the button in
the header bar; `Tab` closes it and any open card and jumps to the search box;
`Ctrl-Z` or `Ctrl-U` undoes the last pick) is the grid a draft room
keeps on the wall: a column per team in draft order, a row per round, the pick
in the cell, coloured by position. Each cell carries the pick as `3.05`, the
player's VORP, and how far past his ADP he actually went — `+12` means he was
still there twelve slots after the market price, `−9` that someone reached.
Simulated picks keep their `sim` badge, the footer totals each team's positions
and the starting lineup it would field today, and clicking a cell opens the same
player card the rest of the page uses. Nothing in it drafts.

The header bar can also **skip ahead**: *sim to my pick* has the model make
every other seat's pick until you are on the clock, and the pick box (overall
`37` or round.pick `3.05`) jumps to any later pick, simming through your own
turns too if the target passes them. Picks are sampled from the model's
distribution by default (untick *sampled* for its single most likely pick each
time, which is reproducible but chalky); each seat's picks respect its
auto/human setting in the League panel. Simmed picks carry a `sim` badge on
the roster, and *undo sim* rewinds the whole last jump at once — plain *undo*
still steps back one pick at a time.

**Candidates the value curve cannot separate share one rank number and are
labelled a tie**, with the band quoted. That is not hedging: a slot mean
averages ~14 seasons, so across WR30-39 the trend measures +1.50 pts/slot
(SE 1.66, t 0.90) and the whole spread of available backs at a round-8 pick is
+19.1 ± 10.6 — a ranked list of eight of them would be a ranked list of noise.
Ties are still ordered by the best estimate, because indistinguishable is not
indifferent; every row also shows what you *lose by waiting*, which is the
number to overrule a tie with. See [REPORT.md](REPORT.md) § the resolution floor.

Another league or a different range — flags, not edits:

```bash
uv run python -m tendies --league-id 12345678 --first-season 2015 --last-season 2025 build
```

| flag | meaning |
| --- | --- |
| `--league-id` | ESPN league id (the `leagueId` in the fantasy URL) |
| `--first-season` / `--last-season` | inclusive range; seasons before the league existed, and a scheduled-but-unheld draft, are skipped with a message |
| `--refresh` | re-fetch instead of reading the cached raw JSON |
| `--no-browser` | never open Chrome; fail if stored cookies do not work (CI) |
| `--adp` | primary ADP board (default: **FantasyPros half-PPR**) |
| `--adp-fallback PATH[=TAG]` | extra board in its own `*_<tag>` columns and next in the fill chain; repeatable, order is the fill order |
| `--primary-tag` / `--no-fallback` | name the primary board in `adp_source` / use it alone |
| `--miss-through` | report unmatched picks inside this many overall picks |

Headless environments skip the browser entirely by exporting `ESPN_S2` and
`ESPN_SWID`.

## What it writes

| file | contents |
| --- | --- |
| `data/raw/espn/<league>/league_<season>.json` | the exact API payload, cached |
| `data/raw/espn/players/players_<season>.json` | that season's player universe (public endpoint) |
| `data/processed/draft_picks_<league>.parquet` / `.csv` | **the dataset** — one row per pick |
| `data/processed/franchises_<league>.csv` | franchise ↔ managers ↔ team names ↔ seasons |
| `data/processed/seasons_<league>.csv` | per-season league name, size, draft type and date |
| `data/raw/boone.json`, `data/raw/etr.json` | third-party hint lists, with resolved player ids written back in place by `tendies hints` |

One row per pick:

| group | columns |
| --- | --- |
| identity | `season, league_id, franchise_id, franchise_label, manager, team_id, team_name` |
| the pick | `overall_pick, round, pick_in_round, bid_amount, keeper, auto_pick` |
| the player | `player_id, player_name, position, pro_team, gsis_id` |
| primary board (FP half-PPR) | `adp, adp_rank, adp_pos_rank, adp_delta, adp_match, adp_name` |
| fallback boards (FP full-PPR, then FFC half) | `adp_*_ppr`, `adp_*_ffc` |
| coalesced | `adp_rank_filled, adp_delta_filled, adp_source` |

`adp_delta = overall_pick - adp_rank`: **positive = taken later than the
market** (value), negative = a reach. Null where the player has no ADP row.
`gsis_id` is the nflverse player id, so picks join straight to the parent
repo's actuals and features.

## How the four hard parts work

**Auth.** The browser authenticates and nothing else. Playwright opens Chrome
with a persistent profile under `.auth/`, waits for `espn_s2` + `SWID`, stores
them, and every later run is a plain HTTPS call — a full rebuild needs no GUI.

**Two ESPN endpoints.** ESPN serves the current season from
`/seasons/{y}/.../leagues/{id}` and past seasons from `/leagueHistory/{id}`.
The client tries both, so a season is just a number to the caller. Requests
retry on 429/5xx, and a payload that claims a completed draft while carrying
zero picks is rejected rather than cached — that exact throttled response
silently erased 2025 from a build during development.

**Year-over-year team linking.** ESPN's `team_id` is a slot in one season's
league and is reused when a manager leaves, so it is not an identity. The
stable key is the owner's SWID GUID. Franchises are the connected components
of "these two team-seasons share an owner" (union-find), which survives team
renames, co-owners, and a co-owner taking over. `franchise_id` is stable
(`F01`, `F02`, … ordered by first season) and `franchise_label` is the
manager's real name where ESPN has one. Two accounts belonging to one person
are reported, not silently merged.

**ADP matching.** Names are the only shared key, so matching runs in passes
and every pick records which one hit it in `adp_match`: `name` (normalized
full name, position agrees), `alias` (an explicit entry in `ALIASES` in
`adp_match.py` — extend it), `initial` (unique first-initial + last name +
position within the season), or `none`. Kickers and defenses have no ADP by
construction and are excluded from the coverage rate `build` prints.

Boards are consulted as an ordered chain: the primary board fills `adp_rank`,
each fallback fills its own `adp_*_<tag>` columns, and `adp_rank_filled` /
`adp_source` say what was used and where it came from. The primary columns
stay pure — no board is ever silently substituted for another.

The sharp edge is **legal name changes**. ADP sources key on a player id and
restate old boards under the player's CURRENT name, while ESPN's draft history
keeps the name as drafted: every board calls the 2019-2021 Robby Anderson
"Robbie Chosen". That is an `ALIASES` entry, and it is the first thing to
check when a prominent player shows up unmatched.

## Second opinions on the board (Boone, ETR)

Two hand-keyed hint lists ride along on the board as a tiebreaker for the
places where VORP, ADP and ECR all say the same thing:

| column | source | on how many of the 350 board rows |
| --- | --- | --- |
| `BOONE` | Justin Boone's overall ranks, `data/raw/boone.json` (303 names) | 268 |
| `ETR` | Establish The Run's take/avoid calls and the round each applies to, `data/raw/etr.json` | 35 |

`BOONE` shows the rank and colours it on the gap against ADP, on the same ±8
thresholds `EDGE` uses, so a green 12 beside an ADP of 30 reads the same way in
both columns. `ETR` shows `Take R4` / `Avoid R2`. Both sort. A blank is a real
"no opinion", not a zero: Boone's list is 303 deep and 82 board rows are not on
it, ETR calls 35 players.

**Ids are resolved once, not at render time.** These are name lists typed by
hand — ETR writes "Ceedee Lamb", "Travis Ettiene", "Jeremiah Love"; Boone
writes "JAC" for the defense the board calls "Jacksonville Jaguars DST" — so
`tendies hints` matches each name against the board and writes `id` (the
board's `gsis_id`, or `null` for the K and D/ST rows FantasyPros gives no id)
and `key` (the board's own canonical name key) back into the JSON beside it.
`cli.cmd_site` then LEFT-joins on `gsis_id` first and `key` second, the same
ladder `board.link_picks` uses, and asserts the board's row count does not
move. A rename on either side therefore turns up as an unresolved row in
`tendies hints` — a number that has to change — rather than as a hint that
quietly stops appearing. The 35 Boone names and 0 ETR names that resolve to
nothing are printed in full; all 35 are at his rank 208 or worse and are
genuinely absent from the FantasyPros board.

`hints.py` deliberately has no initial-and-surname fallback pass, which
`adp_match` has. It was tried: its one extra match was Boone's "Trevor
Etienne" (CAR, not on the board) landing on Travis Etienne Jr. (NO). For a
hint column a wrong player is worse than a blank one.

**These columns are presentation only.** They are attached in `cmd_site`,
downstream of `train`, and nothing in the choice model, the VORP curve or the
simulation can see them — the same rule the parent repo applies to VONA. A
hint list is one analyst's opinion with no walk-forward evidence behind it, and
there is no honest way to tune on it: ETR's rows are 2026-only and Boone's
ranks have no history in this repo at all. The season the two files describe is
pinned in `hints.HINT_SEASON`, so a file swapped for next year's board cannot
silently price this year's.

## Caveats

- **Rookie status is NFL entry year**, read from
  `../data/raw/context/roster_<season>.parquet` and matched by `gsis_id` then by
  normalised name. The two obvious alternatives both fail on the rookie class
  itself: nflverse has no `gsis_id` for a player who has not played, and the
  FantasyPros-id route matches 77.5% of the 2026 board while finding zero
  rookies. Coverage is complete where it matters — 0-2.5% of skill rows
  unresolved, **none** inside the top 100 of any board, which `rookies.check`
  asserts. Defenses are always 0 by construction.
- **ADP comes from FantasyPros** (`bff/fp_adp.py` in the parent repo), which
  replaced FantasyFootballCalculator because FFC's boards are pruned at the
  publisher. FP half-PPR runs 246-426 deep per season versus FFC's 117-191, so
  it covers every pick in a 10-team draft. Sanity-checked against FFC on
  shared players: Spearman 0.94-0.98, top-50 mean rank gap 3-4.
- FantasyPros' own historical pages still drop the occasional real player —
  Austin Ekeler is on neither 2025 FP board, at ADP 90 on FFC's — which is why
  the superseded FFC half board stays in the fallback chain. Current coverage
  for this league is **100% of skill-position picks, all 7 seasons**.
- Kickers and defenses have no ADP by construction and are excluded from the
  coverage rate.
- **The live season prices off REAL-TIME ADP, not AVG, and that is pinned per
  season** (`REALTIME_ADP_SEASONS`). The FantasyPros board carries both: AVG is
  a mean pick over the whole sampling window, real-time is a dense rank over
  only the recent drafts, so it reprices news the average is still averaging
  away. Measured 2026-09-02: Josh Jacobs AVG 48.0 (rank 49), real-time 96,
  half-PPR ECR 152 — the experts and the recent market agree and AVG is the
  outlier, so drafting off AVG reaches ~50 picks early. 2026 uses real-time
  (272 of 355 rows; the 83 the recent sample has never taken keep their AVG,
  none above AVG rank 178), every training season stays on AVG. Only the 2025
  and 2026 captures even carry the column, and 2025's "recent drafts" are
  drafts happening *today* rather than in its preseason — the same in-season
  -information trap `ECR_SOURCE_BY_SEASON` exists to keep out. Rebuild the
  parent parquet to match: `uv run python -m bff.fp_adp --build --format half
  --realtime`. **Real-time ages by the hour** — re-fetch the capture the
  morning of the draft.
- **Expert ranks are half-PPR wherever a preseason board exists, and pinned per
  season.** The board is half-PPR, so the ECR that prices it should be too —
  scored against full-PPR ECR the 2026 board shows a mean rank gap of 10.8 where
  the matched half-PPR series shows 8.3. Half prices 2018, 2020-2025 (Wayback
  preseason captures) and 2026 (a live pull 8 days before kickoff); full-PPR
  prices 2012-2017 and 2019, where the only half boards are FantasyPros' live
  `?year=` pages. Those serve the LATEST REVISION, not a frozen pre-draft
  snapshot, and the parent repo measures them at +0.0201 mean Spearman against
  outcomes — the size of its entire published edge over ADP. The tell is team
  codes: a revised board lists today's rosters, agreeing with that season's
  week-1 roster on 0.4-5% of rows where a real capture agrees on 95.5-99.7%.
  `config.ECR_SOURCE_BY_SEASON` is the pin, and it is a decision, not a
  preference: an undeclared season raises rather than falling back, and a season
  pinned to half whose rows are not preseason raises too. Audit the series with
  `python -m tendies ecr-check` before changing it.
- **VORP is points above the best option you would field at that position for
  FREE**, and that one definition has two arms because the free option differs by
  position. QB and TE are streamable — one starter each, ~14 drafted, so punting
  the slot means rotating the waiver pool on matchup, not rostering one man — and
  their replacement is a simulated STREAMING TOTAL (QB7, TE6). RB and WR are not;
  ~49 and ~56 come off the board and their replacement is the measured BEST
  UNDRAFTED player (RB43, WR51). `python -m tendies streaming` derives all four
  from this league's own drafts.
  The mixed definition that preceded this (QB/TE streaming-aware, RB/WR at
  roster-demand depth RB25/WR30) understated every back by 38 points and every
  receiver by 30, which floated quarterbacks and tight ends to the top of the
  board: greedy-best-available-by-VORP used to finish *behind* simply following
  ADP in a mock draft, and now finishes clearly ahead of it.
- **The QB/TE numbers carry real uncertainty and it is not small.** The streaming
  sim assumes you lose the season's single biggest waiver breakout to one of your
  nine rivals (`drop_top=1`). Assume you win it and replacement is QB3/TE3, which
  prices quarterbacks and tight ends near zero all the way down; assume you lose
  the top two and it is QB16/TE15. `tendies streaming` prints the whole sweep —
  read it before treating QB7/TE6 as settled.
- **An occupied QB/TE slot earns `vorp.HOLD_GAIN` (QB 15, TE 6) on top of
  `max(VORP, 0)`** (2026-09-04). Holding a body and streaming around him beats
  streaming alone even when he is below the floor — measured paired within
  season over 2013-2025 by `tendies streaming` (TE +6 to +13, QB +13 to +19) —
  and without it a below-floor tight end priced identically to no tight end and
  could never earn a pick on value. Uniform within the position, so nothing
  reorders; on the ten-seat mock it moved no pick on either board and lifted
  every finished lineup by exactly +21. RB/WR get none: their one-slot stream
  baseline banks RB14-25 / WR27-40, which is the proof they are not streamable,
  and a bench body there is priced by the insurance term instead.
- Raw payloads are cached and never re-fetched without `--refresh`, so reruns
  are offline and reproducible.
- `data/processed/` is git-ignored because the outputs carry league members'
  real names; drop that line from `.gitignore` if you want them committed.

## The model layer

| module | responsibility |
| --- | --- |
| `board.py` | season player universe: FantasyPros half-PPR **plus K and D/ST**, restored from the raw capture that the parent repo's build filters out |
| `rookies.py` | NFL entry year per player, from the parent repo's raw nflverse rosters — the only source that resolves the CURRENT rookie class |
| `attrs.py` | the other player attributes managers have a taste for: age, last season's durability, and who drafted him last year |
| `traits.py` | per-manager tendencies and their split-half reliability; the descriptive half of what the model fits |
| `vorp.py` | 10-team half-PPR value curve, indexed by **ECR** (not ADP) on both sides — the curve's own slots as well as the lookup — and fitted isotonically rather than smooth-then-clamp. Ships a **standard error with every value**, so consumers can tell the resolved top of the board from the unresolved back of it |
| `streaming.py` | where each position's replacement level comes from: a streaming simulation for QB/TE, the measured best-undrafted slot for RB/WR. Also measures what holding a below-floor QB/TE body is worth over punting (TE +6-13, QB +13-19 a season; shipped as `vorp.HOLD_GAIN` QB 15 / TE 6 — see REPORT, "Why the page will not draft a tight end"). Prints; the record behind `vorp.REPL_RANKS` and `vorp.HOLD_GAIN` |
| `depletion.py` | how the board empties, measured: mean cumulative picks by position at each point of a draft. Prices an unfilled starting slot, and stands in for the simulation past the next two turns |
| `league.py` | per-season rules read from ESPN — this league moved from 1 FLEX/7 bench to 2 FLEX/5 bench, and hardcoding today's settings silently corrupts older rows |
| `replay.py` | the draft-state engine; one code path serves training, simulation and the live page |
| `features.py` | the feature contract, mirrored by `web/engine.js` and asserted at page load |
| `choice_model.py` / `train.py` | penalised nested logit, sequential estimation |
| `autopick.py` | autodraft as a two-state Markov chain |
| `simulate.py` | seeded forward simulation + survival recalibration |
| `evaluate.py` / `calibrate.py` | walk-forward scoring, baselines, reliability |
| `export.py` / `site.py` / `web/` | the train/serve boundary and the page |

**Why the JS lives in `web/*.js` rather than Python strings** (the parent
repo's pattern): the Python/JS parity test cannot run against a string
literal, and that test is the only thing standing between "the model is right"
and "the page is right".
