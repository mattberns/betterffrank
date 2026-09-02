# tendies — predicting who your leaguemates draft

One league (ESPN 20482930), 2019-2025, 10-team snake, half-PPR, **1,070 picks**.
The question: given a live draft state, which player does the manager on the
clock take? And downstream: who survives to your next turn, and what should you
do about it.

Everything below is walk-forward — season *s* is predicted by a model fitted
only on seasons before it. Six scored folds (2020-2025), 906 picks.

---

## Headline

| | top-1 | top-3 | skill-only top-1 | skill-only top-3 |
| --- | --- | --- | --- | --- |
| B0 best available by ADP | 15.3% | 32.0% | 17.2% | 35.8% |
| B1 + position caps and roster limits | 15.3% | 32.0% | 17.2% | 35.8% |
| B2 + take an empty K/D-ST slot in the last two rounds | 17.2% | 35.5% | 17.2% | 35.8% |
| **model** | **20.0%** | **44.0%** | **20.2%** | **45.4%** |

Position accuracy 47.8%. Mean NLL 3.4533 nats over a choice set of ~120.

These moved twice on 2026-09-01, as two rounds of tendency work landed:
21.1% / 42.8% before, 21.5% / 42.2% after `is_rookie` / `early_qb` / `early_te`,
and 20.0% / 44.0% after `age` / `was_mine` / `durability`, their five new
per-manager channels, and the pruning of three of those channels back off.
Top-1 fell 1.5pp and top-3 rose 1.8pp across the two rounds. Read the
direction, not the digits: at n = 906 the paired SE on a top-1 difference is
roughly a point, and NLL — the proper scoring rule, and the only one of the
three not decided by argmax ties — improved throughout (3.4623 → 3.4533). See
"Do managers have tendencies?" for the part of that move the manager channels
are responsible for, which is not the whole of it and does not point the same
way on every metric.

By round, against B2 — the gain is concentrated exactly where a human is least
sure:

| | model top-1 / top-3 | B2 top-1 / top-3 |
| --- | --- | --- |
| rounds 1-3 | 36.1 / 64.4 | 33.3 / 59.4 |
| rounds 4-8 | 18.5 / 46.5 | 16.2 / 34.7 |
| rounds 9+ | 14.2 / 33.8 | 11.2 / 26.1 |

Rounds 1-3 have almost no headroom: the market is nearly sufficient and the
residual is taste noise on a handful of picks per manager per season.

**Report B2, not B0.** "Don't draft a third kicker, and do draft your first one
before the draft ends" costs nothing and is worth +1.9pp top-1 on its own. A
model that only beat B0 would have earned nothing.

**Report skill-only alongside.** K and D/ST are on this board but were absent
from the ADP board behind the original 17.2% figure. Adding kickers raises the
combined headline without any modelling.

---

## What the model is

`P(player) = P(position | state) × P(player | position, state)` — a nested
logit, fitted by penalised maximum likelihood (`choice_model.py`, scipy
L-BFGS-B, analytic gradients).

Why factorised: given the position, the choice is far sharper (top-1 41% vs
17%), and a flat conditional logit over ~250 players asserts that removing the
best RB shifts demand proportionally to every other position, which is plainly
false. Nesting relaxes exactly that.

The inclusive-value coefficient **λ ≈ 0.00** (bounded to [0, 1.2], and it sits
on the lower bound) is the interesting fitted number. λ = 1 would mean the two
stages collapse to a single joint conditional logit over every available
player; λ = 0 means they are independent. The data picks 0.

Read literally: managers choose a POSITION from roster need, round and
scarcity, and only then take the best player at it — the *quality* of what is
available at a position does essentially nothing to the position decision.
The joint conditional logit, which is what you get by modelling this as one
big choice over all players, is strongly rejected. λ costs one parameter and
it settled an architectural question, which is the entire reason to fit a
nested logit rather than assert a factorisation.

**Out-of-window picks are not dropped.** Each position carries an outside
option with its own constant, so the 2.2% of picks outside the candidate
window score against it. Dropping them would condition the likelihood on "the
pick was a top-20 name" — selection on the label.

Coefficients read the way you would hope: `first_at_pos +1.35`, `log_count
−1.01` (diminishing returns on stacking a position), `endgame_k +3.28`,
position constants ordering K < D/ST < QB < TE < WR ≈ RB.

### Which features actually earn their place

Leave-one-out over the walk-forward folds, zeroing one feature everywhere and
refitting. Positive Δnll means removing it *hurt* — i.e. it was doing work:

| feature | Δnll | reading |
| --- | --- | --- |
| `u_adp` | **+0.169** | ADP-following within a position; the single load-bearing feature |
| `ecr_lean` | **+0.058** | expert-vs-market disagreement genuinely predicts, within position |
| `endgame_k` / `endgame_dst` | +0.043 / +0.037 | the last-two-rounds structure |
| `log_count` / `first_at_pos` | +0.010 / +0.009 | roster need |
| `n_gone` | +0.008 | positional scarcity before your next turn |
| `flex_need` | +0.004 | |
| `run5`, `run10`, `bye_clash`, `posrank_gap`, `vorp_gap`, `best_vorp` | −0.007 to −0.001 | noise; removal neither helps nor hurts beyond error |

Six more landed on 2026-09-01 with the tendency work and are not in that pass.
Position stage: `early_qb` +0.377, `early_te` +0.323 — quarterbacks and tight
ends are taken somewhat more often in rounds 1-6 than the rest of the model
expects. Player stage: `is_rookie` −0.083, `age` +0.022, `was_mine` **+0.371**,
`durability` −0.106.

Two of those pooled coefficients are worth a sentence each. **`was_mine` is the
largest of the six**: this league really does re-draft its own players, by about
as much as it avoids stacking an NFL team (`same_team` −0.393). And
**`durability` comes out NEGATIVE**, meaning that at equal ADP the player who
missed time last season is preferred — which reads backwards until you remember
the model is conditioning on market price. The absentee's ADP has already been
marked down for the absence, so what is left is the bounce-back candidate.

They exist mainly to give the per-manager deviations something to deviate FROM;
the per-manager half is priced in "Do managers have tendencies?" below.

That pass also caught a modelling error worth recording. Three features —
`log_h` (picks until your next turn), `rounds_left` and `starters_filled` —
came back at **exactly** 0.0000 on every metric. They are properties of the
OCCASION, identical across every position in a choice set, so they cancel in
the conditional-logit softmax: structurally unidentified, not merely weak. They
have been removed (14 position features, not 17), and removing them changed
every reported number by exactly zero, which is the confirmation. Occasion-level
information can only enter through an interaction with position — which is what
`endgame_k` / `endgame_dst` and, since 2026-09-01, `early_qb` / `early_te` are.
"He takes quarterbacks early" is an occasion-level claim about a position, so
that shape is the only one available to it; a bare round term would cancel
exactly the way `rounds_left` did.

Note the deliberate asymmetry between the two pairs. `endgame_*` counts
BACKWARD from the end, because "the last two rounds" has to mean the same thing
in the 16-round 2019-20 drafts and the 15-round ones since. `early_*` counts
FORWARD from round 1, because "round 6" already does.

### VORP is priced off ECR, not ADP — on both sides of the curve

The value curve is indexed by within-position **expert consensus rank**, with
ADP only as a fallback for players the ECR boards do not reach. ADP is where
the market ended up; ECR is the better estimate of what a player is worth, and
the parent repo measured the gap directly — ECR beats ADP by +0.0145 mean
spearman_vorp on its 2018-2025 test window, 7 seasons of 8, widening to +0.055
by 2025.

What that changes: Saquon Barkley goes at ADP 13 on the 2026 board but the
experts have him 26th, so he is priced as RB10 rather than as a top-five back.
The page shows both columns and the gap between them, because that gap is the
actionable part.

Three corrections landed here later. The first two were cases of the code not
doing what its own docstring said; the third was the code doing exactly what it
said, on data that changed underneath it.

**The curve's own slots were indexed by MARKET rank.** `attach_vorp` looks a
player up by his expert slot, but the history the curve is built from
(`fp_adp_ppr.parquet`) carries no ECR column, so `slot_key`'s
fallback-for-unranked-players quietly applied to every row of it. The curve was
answering "what did the market's k-th pick score" while being asked "what is
the expert's k-th ranked player worth". ECR is now joined onto the curve
history too. Cost: top-1 20.97% → 20.53%, top-3 43.16% → 42.83% — 0.3 of a
standard error (1.35pp at n=906), so the correctness argument decides it rather
than the metric.

**The board was half-PPR and the expert ranks were full-PPR.** Comparing a
half-PPR ADP against a full-PPR ECR manufactures an edge everywhere receptions
matter, which is most of the board: against the matched half-PPR series the
2026 board's mean |ADP − ECR| gap is 8.3, against the full-PPR series it is
10.8, and the extra 2.5 places is a scoring-format artifact. `load_ecr` was
changed to prefer half-PPR per season, which at the time covered 2026 alone.

**Then the half-PPR series grew, and preferring it turned out to be the wrong
verb.** On 2026-09-01 `ecr_half.parquet` was rebuilt from 2026-only to
2013-2026. Because `load_ecr` took half wherever half existed, every training
season repriced with no code change and nothing printed — top-1 20.53% →
20.86%, top-3 42.83% → 43.71%, NLL 3.4768 → 3.4271, 535 of 906 picks moving —
and `docs/index.html` was re-rendered on top of it 39 seconds later. The gain
was not free: those boards came from FantasyPros' live `?year=` pages, which
serve the latest revision rather than a frozen pre-draft snapshot.
`bff/fp_ecr.check_preseason` measures the difference at **+0.0201** mean
Spearman against outcomes, positive in 7 of 8 seasons, which is the size of the
parent model's entire published edge over ADP.

The fix is a per-season **pin** in `config.ECR_SOURCE_BY_SEASON` rather than a
preference, plus a rule that a season pinned to half must carry a preseason
`source` in the parquet. An undeclared season now raises. Which series prices
which season is a decision recorded in the repo and stamped into every artifact
as `ecrSource`, not an emergent property of what a file happens to contain.

Provenance was then re-established per season, and `python -m tendies ecr-check`
audits it against evidence rather than against the file's own label. The
discriminating test is CONTEMPORANEITY: FantasyPros restates historical pages
under current team identity, so a revised board lists today's rosters. Measured
against `ctx_rosters_week1`, revised seasons agree on **0.4-5%** of rows and
real captures on **95.5-99.7%**. Nothing else separates them — Spearman against
the PPR twin reads 0.96-0.996 on both, which is why it is only a warning.

| pinned | seasons | why |
| --- | --- | --- |
| half | 2018, 2020-2025 | Wayback preseason captures |
| half | 2026 | live pull, 8-day lead on kickoff |
| ppr | 2012-2017, 2019 | no preseason half board exists |

2019 is the one that hurts, because it is the league's first draft and so a real
board season rather than curve history. Its only capture is dated 2019-09-13 and
the page header says "Week 2" — it has seen Week 1 results — and the archive
holds nothing for that page in August or early September 2019. It is never
scored (folds are 2020-2025); it is training data whose share shrinks every
fold, carrying a format gap where the two series correlate 0.95-0.99.

What the clean switch is actually worth, walk-forward on the same 906 picks:

| | top-1 | top-3 | NLL |
| --- | --- | --- | --- |
| full-PPR expert ranks (the published baseline) | 20.53% | 42.83% | 3.4768 |
| **half-PPR where attested (shipped)** | **21.08%** | **42.83%** | **3.4575** |
| half-PPR everywhere, including revised boards | 20.86% | 43.71% | 3.4271 |

So the format correction is worth +0.55pp of top-1 (0.4 of a standard error at
n=906, and 3 seasons up against 3 down, so not a result on its own) and 0.019
nats of likelihood, with top-3 unchanged to four decimals. The third row is the
useful one: **all** of the top-3 gain and most of the likelihood gain that the
accidental switch appeared to deliver came from the revised boards, not from the
scoring format. As with the curve-slot correction above, the correctness
argument decides this, not the metric.

The market rank still drives the BEHAVIOUR features (`u_adp`, `adp_lead`) —
what a manager *will do* is a different question from what a player is *worth*,
and managers draft close to ADP. That split is visible in the numbers above:
each move *toward* expert pricing costs a fraction of a point of prediction
accuracy, because a more market-flavoured VORP predicts market-following
managers slightly better. That is not evidence it is a better value estimate.

---

## Autodraft is 17% of all picks, and it is a Markov chain

Per manager it runs from 6.5% (Streeter) to 68% (Harwood), 57% (Clark), 50%
(Kelley). It is **not** a per-pick coin flip:

- P(auto | previous pick was auto) = **0.906** (160 transitions)
- P(auto | previous was human) = **0.024** (840 transitions)
- 47 of 70 team-seasons contain none; 13 of the rest are a pure suffix —
  someone left the room. Two are 100% auto from pick one.

So the state is drawn once per simulated path and carried forward. Re-drawing
it every pick collapses to the marginal rate and understates the variance of
the survival probabilities the page displays. The page exposes a per-manager
**prior / human / auto** control, because at 11pm you can often *see* who has
walked away.

Separating auto from human picks matters for the fit too: within a position,
autodraft takes the top available 38.9% of the time against humans' 40.2% —
indistinguishable. The difference is entirely in *which position* gets taken,
so the player stage is shared and only the position stage is split.

---

## Do managers have tendencies?

This is what the project is named for, so it gets a straight answer: **several
are real and stable as TRAITS, four of nine channels earn their keep in the
likelihood, and the apparatus as a whole still buys less than a point of
anything.**

There are four per-manager channels, each independently droppable
(`train.MANAGER_BLOCKS`):

| channel | stage | scales | what it says | params |
| --- | --- | --- | --- | --- |
| `manager_pos` | position | — | how often he takes a QB / RB / TE at all | 13 × 3 |
| `manager_timing` | position | — | whether his QB / TE comes in rounds 1-6 | 13 × 2 |
| `manager_reach` | player | `u_adp` | how steeply he follows ADP inside a position | 13 |
| `manager_rookie` | player | `is_rookie` | appetite for first-year players | 13 |
| `manager_expert` | player | `ecr_lean` | drafts off the expert list or off the market | 13 |
| `manager_stack` | player | `same_team` | stacks his own NFL teams or diversifies | 13 |
| `manager_age` | player | `age` | prefers young or old for the position | 13 |
| `manager_mine` | player | `was_mine` | re-drafts the players he had last season | 13 |
| `manager_durability` | player | `durability` | avoids or ignores last season's absentees | 13 |

All but `pos` and `reach` were added on 2026-09-01, in two rounds. The seven
player-stage channels are one piece of algebra declared once in
`train.PLAYER_CHANNELS`: a design column equal to the STANDARDISED feature,
switched on for that manager's occasions. `reach` is not a special case —
`acc += z * (posAdp[p] + reach)` is exactly a coefficient on standardised
`u_adp` — and unifying it is what made adding five more a table entry rather
than seven edits across seven files.

That is 91 player-stage manager coefficients on ~1,066 choice occasions, plus
65 in the position stage on ~889. They are all penalised toward the pooled
model (`tau` 0.25, or 0.15 for timing), which is the only reason the fit is
identified at all — and the ablation below is what says whether the penalty was
generous enough.

### What each channel is worth

Leave-one-out over the walk-forward folds — each row is a full refit, not a
zeroing. `d_nll > 0` means dropping the block HURT, so it was doing work. Each Δ is
against the all-nine fit at the top, which is why the shipped row at the bottom
reads negative: it drops three blocks at once. NLL is the column to read: it is a proper scoring rule over the whole distribution,
where top-1 and top-3 on 906 picks turn on a handful of argmax ties.

| | top-1 | top-3 | NLL | Δnll |
| --- | --- | --- | --- | --- |
| all nine channels | 20.09% | 43.71% | 3.4604 | — |
| − `manager_reach` | 19.87% | 43.27% | 3.4649 | **+0.0045** |
| − `manager_expert` | 19.98% | 43.82% | 3.4628 | **+0.0024** |
| − `manager_age` | 20.53% | 44.48% | 3.4627 | **+0.0023** |
| − `manager_rookie` | 19.87% | 43.93% | 3.4624 | **+0.0020** |
| − `manager_pos` | 19.98% | 43.38% | 3.4619 | +0.0015 |
| − `manager_timing` | 20.09% | 43.71% | 3.4606 | +0.0002 |
| − `manager_mine` | 20.20% | 43.93% | 3.4600 | −0.0004 |
| − `manager_stack` | 20.09% | 43.60% | 3.4598 | −0.0006 |
| − `manager_durability` | 20.31% | 43.71% | 3.4544 | **−0.0060** |
| **all off (true B4)** | **20.42%** | **44.26%** | **3.4687** | **+0.0083** |
| **six shipped (`DEFAULT_BLOCKS`)** | **19.98%** | **44.04%** | **3.4533** | **−0.0071** |

**Four channels earn their place, two are inert, and one is actively harmful.**
Reach tolerance leads, as it did before; of the five added second,
expert-following and age preference land third and fourth overall, stacking and
loyalty are indistinguishable from zero, and durability is the worst channel in
the table by a factor of six — thirteen more parameters fitted on a signal that
is not there.

**The three losing channels are pruned and the six survivors are what ships.**
`train.DEFAULT_BLOCKS` is `ALL_BLOCKS - {stack, mine, durability}`, giving
19.98% / 44.04% / **3.4533** — the best likelihood of any configuration tried.
Their POOLED features stay in the model, and two of them are among its larger
coefficients (`same_team` −0.393, `was_mine` +0.371); what is dropped is only
the claim that managers differ from one another on them.

Two caveats on that prune, both real. **It is selection on the six evaluated
folds**, which is the optimism this report lists first, so the 3.4533 is not an
out-of-sample number in the way the baselines are. And **the apparatus still
points two ways at once**: turning every manager channel off gives 20.42% /
44.26%, better than the pruned model on both top-1 and top-3, while costing
0.0154 nats. The manager terms sharpen the predicted distribution and move the
argmax wrongly on a few picks; those are different things and 906 picks cannot
say which matters more. NLL is the metric this repo trusts, because top-1 on
this sample turns on a handful of argmax ties, but a reader who only cares
about the top-3 list the page renders should know the pooled model wins that
column. `blocks=train.ALL_BLOCKS` restores all nine.

`--b4` used to be measured wrong and the old +0.33pp figure is void: the switch
was a boolean that only reached the position stage, so `reach[m]` was fitted
underneath a run that reported every manager deviation as off. `train(blocks=…)`
replaces it and `test_manager_blocks_drop_cleanly` pins it shut.

### Are the traits themselves real?

Separately from whether they predict: does a manager's trait repeat across his
own seasons? `traits.py` computes thirteen traits on his odd seasons and again
on his even ones and correlates the two across the seven managers whose halves
both carry 25+ human picks. `full_r` is Spearman-Brown, reported only where r is
positive because the step-up formula is meaningless otherwise; `p` is a seeded
two-sided permutation test on the manager labels.

| trait | r | full_r | p | reading |
| --- | --- | --- | --- | --- |
| mean ADP delta | **+0.90** | +0.95 | 0.010 | how far from the market he drafts is his most stable trait |
| reach rate (≤ −10) | **+0.79** | +0.88 | 0.034 | real |
| rookie rate | **+0.74** | +0.85 | 0.048 | real |
| RB share, rounds 1-8 | +0.70 | +0.83 | 0.093 | real; Hawxhurst .523 vs S. Patterson .327 against a .398 league average |
| **mean age vs position** | **+0.53** | +0.69 | 0.248 | the best of the second round of traits |
| TE share, rounds 1-8 | +0.42 | +0.59 | 0.351 | weak |
| mean round of first TE | +0.36 | +0.53 | 0.387 | weak |
| expert lean (ADP − ECR) | +0.32 | +0.48 | 0.496 | weak as a trait, useful as a channel — see below |
| QB share, rounds 1-8 | −0.00 | — | 1.000 | noise |
| mean round of first QB | −0.20 | — | 0.680 | noise |
| mean durability | −0.30 | — | 0.492 | noise |
| stack rate | −0.44 | — | 0.352 | noise |
| reunion rate | −0.76 | — | 0.009 | see below |

With seven managers the SE on r is about 0.45, so read the ORDER, not the third
decimal. The order broadly agrees with the ablation — reach and rookie high in
both, QB timing at the bottom of both — with one instructive exception.

**Expert lean is a weak trait and a useful channel, and that is not a
contradiction.** The trait is the mean (ADP − ECR) of the players he took, which
does not know what was on the board; the channel is his coefficient on
`ecr_lean` INSIDE the choice set he actually faced. Conditioning on availability
is the difference, and it is why the descriptive and fitted layers are both
worth having rather than one being a cheaper version of the other.

**Reunion rate comes back significantly NEGATIVE (r = −0.76, p = 0.009), and a
negative reliability is not a weak trait — it is evidence the quantity is not a
property of the person at all.** The most likely mechanism is mechanical: the
players a manager can re-draft are the ones he took last year who are still on
the board, and how many that is depends on how his previous draft aged, not on
his loyalty. A manager who took breakouts re-drafts few of them at their new
price. Note the POOLED effect is real and large — `g_was_mine` is +0.371, so
this league genuinely does re-draft its own players — it just is not a trait
that distinguishes one manager from another.

Where the two layers do corroborate, they do it at the player level, which is
the check worth having. Ryan Patterson has the highest rookie rate (.159) and
the largest positive fitted deviation (`rookie[F08]` +0.344); William Streeter
has the lowest (.020) and a negative one (−0.190).

### Three rules this analysis has to follow, because it got them wrong before

1. **Human picks only.** For three managers autodraft is half their picks, and
   the first version of this analysis measured ESPN's robot as their taste.
   `cmd_report` still had no `auto_pick` filter until 2026-09-01.
2. **The timing traits need their own rule.** "Round of his first QB" on human
   picks alone is wrong in a specific way: a manager whose first QB was an
   autodraft in round 7 would score as though he waited for his next human QB
   in round 12. So they are measured over ALL of a team-season's picks, a
   team-season whose first pick at the position was an autodraft is dropped,
   and a team-season with no such pick at all is right-censored at one past the
   last round — "never" is the extreme of "late", not missing data.
3. **Split by season within a manager's career.** Splitting picks instead puts
   both halves inside the same draft, so a trait scores as reliable for
   agreeing with itself about one season's board; measured, that collapses RB
   share to +0.13 and inflates reach rate. Odd/even rather than first/second
   half, because chronological halves confound a stable trait with drift.

### The numbers this replaces

An earlier version of this section published RB share +0.664, reach rate +0.032
and mean ADP delta −0.222, computed by hand with no code behind them, and
concluded that reach rate does not repeat. **That conclusion does not survive.**
The trait DEFINITIONS reproduce exactly (Hawxhurst .523, S. Patterson .327,
league average .398, pinned by a test), but no split method reproduces those
correlations — four candidates span −0.43 to +0.90 on the same data. The two
ADP-derived ones cannot reproduce even in principle: they were measured against
the FantasyFootballCalculator board that `config.DEFAULT_FALLBACK_BOARDS` has
since superseded with FantasyPros. The table above is the pinned-method
replacement.

### Where the new player attributes come from

`rookies.py` supplies entry year; `attrs.py` supplies the other three, and each
has one thing worth knowing.

**`age`** — `birth_date` from the same raw nflverse roster files, taken at 1
September of the draft season and **centred within (season, position)**. The
centring is the point: the player stage only ever chooses WITHIN a position, so
a 27-year-old running back and a 27-year-old quarterback are not the same
proposition. A missing birth date centres to 0, imputing the typical age rather
than asserting anything (97 of 3,045 skill rows, at most 3 inside any board's
top 100). Age and `is_rookie` are not redundant: rookie is a rare 8% indicator
at the extreme tail, age carries information on every pick.

**`durability`** — share of the PRIOR season's games played, from
`actuals.parquet`, centred the same way. This is the ONLY feature in the model
built from the outcome table, so it carries its own leakage test
(`test_durability_cannot_see_its_own_season`, which rebuilds a board with every
actuals row from season t onward deleted and requires the column not to move).
Three cases, and the middle one is why it exists: a prior-season row gives
`games / schedule` clipped to [0, 1] — the clip is load-bearing, because a
player traded mid-season can dodge two byes and appear in 18 games against a
17-game schedule; **no prior-season row and not a rookie means he played zero
fantasy-relevant games** and is scored 0.0, which is exactly the signal a
drafter reacts to; a rookie has no prior season and is placed AT the centre and
left to `is_rookie` to describe, because scoring him 0.0 would call every rookie
maximally fragile.

**`was_mine`** — `prev_owner` on the board is the franchise that drafted the
player in the previous season's draft; `was_mine` is that compared against the
seat on the clock, so the board carries one string per player and the
seat-specific part stays in the feature builder. 81-89% of a season's picks
reappear on the next season's board. Two consequences: board seasons 2018 and
2019 predate the league's first draft and carry the column entirely null, so
`was_mine` has zero variance in fold 2020's training window and its coefficients
are fitted at 0 there (the same situation the parent repo documents for `vs_adp`
in its own first fold); and this is a LEAGUE fact rather than a market fact, so
unlike everything else on the board it cannot be computed for a season the
league did not play.

### What a rookie is here

`rookie = (NFL entry year == this season)`, from the parent repo's raw nflverse
roster files (`data/raw/context/roster_<season>.parquet`), matched by `gsis_id`
and then by normalised name. 82 of 1,070 picks, 9-18 a season.

The source matters more than it sounds, because the two obvious routes both
fail on precisely the population being measured. `gsis_id` is null for a player
who has not played yet — `dataset.check_ecr` already says so in its own comment
— and joining the raw board's FantasyPros id to the crosswalk matches 77.5% of
the 2026 board while finding **zero** rookies: the 82 unmatched rows ARE the
rookie class. The roster files resolve 37 of them, with **zero** unresolved
inside the top 100 of any board, which is what `rookies.check` asserts. A match
rate that looks fine overall can be 0% on the subgroup you care about, which is
why `rookie_match` records which pass hit every row.

One caveat on the specification. λ ≈ 0.009 in this league, so the inclusive
value carries essentially nothing from the player stage into the position
stage. A player-stage rookie term therefore reorders players INSIDE a position
and has almost no effect on which position gets taken. That is the right
behaviour for this trait, but it means "he likes rookies" can never explain a
positional choice in this model.

---

## Survival probabilities — the number the page actually shows

For each pick, simulate the intervening picks forward (Monte Carlo, seeded from
the state) and ask who is still there at your next turn. Backtested
walk-forward on 2022-2025, 6,720 (prediction, outcome) pairs over 112 pick
occasions, restricted to the top 60 available — the players a drafter could
plausibly be waiting on.

| | raw simulation | **as shipped** |
| --- | --- | --- |
| Brier (base rate 0.1712) | 0.0923 | **0.0910** |
| ECE | 0.0246 | **0.0090** |
| worst bin error (MCE) | 0.1144 | **0.0210** |
| mean predicted vs actual | 0.7974 / 0.7808 | **0.7810 / 0.7808** |
| calibration gap, 95% CI | [−1.9pp, −1.4pp] | **[−0.28pp, +0.26pp]** |

The raw simulation **over-predicts survival**, by 6-11pp through the middle of
the range. The mechanism: picks are drawn independently given the state, so
simulated drafts contain fewer positional runs than real ones, and a player who
is "next up" everywhere survives more often in simulation than in life.

Corrected with a two-parameter map on the survival logit, fitted on 2022-2023
and tested on 2024-2025 (ECE 0.0174 → 0.0067 out of sample). Coefficients
fitted on two seasons (a=−0.295, b=1.122) and on four (−0.296, 1.119) agree to
three decimals, which is what makes it a bias worth correcting rather than
noise worth fitting.

After correction the gap's confidence interval straddles zero and no
reliability bin is off by more than 2pp, at either horizon band. When the page
says 70%, 70% is what happens.

---

## The recommendation: what to do about it

Predicting the room is half the job. The other half — which player *you* should
take — was wrong in a way no unit test caught, because every number the engine
computed was internally consistent. Playing whole drafts out is what showed it:
taking the top recommendation fifteen times built **five quarterbacks, one
running back, six tight ends against a cap of three, and no kicker or defense**.
All 70 real team-seasons in this league filled every starting slot.

Four separate defects, and only one of them was the one that looked obvious.

**An unfilled starting slot was scored as zero.** VORP is measured *over
replacement*, so 0 means "a replacement-level player", not "nobody". After the
running-back cliff every remaining back has negative VORP, so filling an empty
RB2 with a −24 back scored −24 while a sixth wide receiver you would never
start scored 0. The engine was actively avoiding filling slots. The floor is
now what you would actually field — the best player at that position who goes
**undrafted**, read off the measured positional flow (below), and at a
streamable position floored at zero (see "What replacement means", below). On
the 2026 board that is QB 0.0, RB 0.0, WR +2.5, TE 0.0, and `lineupValue`
becomes `Σ max(occupant, EMPTY_slot)`: a roster never scores below streaming,
because in reality you would bench the player and stream. This also collapses the
cutoff machinery to one number per position — an empty slot is just a slot
whose cutoff is EMPTY — and `tests/lineup.mjs` brute-forces both the cutoff
identity and the optimality of greedy slot assignment against every legal
assignment.

**`recommend` ignored position caps.** It iterated the top of the ADP watch
list and never consulted `openPositions`, which is where the six tight ends
came from. The candidate set is now every available player at every position
the seat may legally draft.

**And the replacement candidate set was still filtered on the wrong axis**
(found 2026-09-02). It kept the top 8 per position, and `available` walks
`byPos`, which is in ADP order — so it took the eight *shallowest ADPs* at each
position while `recommend` ranks bench picks by MARKET EDGE, which is how far
the experts have a player *above* his ADP. A big edge implies a deep ADP, so
the filter removed exactly the players the score was looking for. Walking a
whole draft, the old slice hid the single biggest market edge on the board in
**8 of the 12 states** where the seat was on the clock (`tests/ties.mjs`,
measured on both the live 2026 board and the 2025 fixture). Scoring the full
board is free — `recommend` is O(1) per candidate once the plan and cutoffs are
built — so it now does.

**From round 7 on, every candidate scored exactly 0.** The sort ran over an
all-zero array, so V8's stable sort handed back the input order: for nine of
fifteen rounds the "recommendation" was the ADP list. Fixed by classifying each
candidate against the value of spending the turn on nobody, and ranking those
that cannot beat it by insurance and market edge instead.

**Greedy is not enough even with the floors**, because a pick spends a TURN as
well as filling a slot. The plan now values the lineup you will *finish* with,
over all remaining turns at once: 544 reachable states (position counts capped
at what the lineup can use, bounded by the flex count), solved backwards, seven
actions per turn. One backward pass prices every candidate as a table lookup,
so the turn's opportunity cost falls out for free — a bench pick and a starter
pick leave the same turns behind and are compared on the same footing. It costs
0.5 ms in node, ~2 ms in the browser, and `tests/plan.mjs` checks it against
exhaustive enumeration on a draft small enough to enumerate (value to 6e-14,
feasibility exactly).

Two things the DP has to get right that a simpler version would not. Availability
is carried five deep per position: with only the best, a plan taking backs at two
consecutive turns values both at the *same* back, because the later turn's board
never had the earlier pick removed. And **kickers are a legality constraint, not
a price** — K and D/ST have zero *differential* value, which is what their flat
curve honestly encodes, but you must field one and VORP cannot see that. The DP
compares `(unfillable mandatory slots, value)` lexicographically.

### The far horizon is measured, not assumed

Simulating to the end of the draft costs 5 s in the browser, so turns beyond the
next two come from a **measured positional flow table** — the mean cumulative
picks by position at each point of a draft, over this league's own 2021-25
seasons.

```
avg cumulative drafted by pick   QB    RB    WR    TE     K   DST
  100                           9.8  37.0  42.8   9.8   0.0   0.6
  140                          13.2  46.6  54.4  12.8   4.0   9.0
  150                          13.4  47.8  55.0  13.8  10.0  10.0
```

The obvious cheap alternative — assume players leave in ADP order — is wrong
exactly where it is needed: kicker ADPs run 120-257 and defenses 125-372, so
under ADP depletion every kicker is gone by pick 150 and Tyler Shough (ADP 313)
gets drafted. Applied as an *increment* from the current pick, the table also
self-corrects for a draft that has already run RB-heavy. Checked against a full
model-driven rollout (N=400) at four states, the tail's bias in best-available
VORP at turns 3+ is QB −3.3, RB +3.7, WR +1.3, TE −0.5 — small enough to leave
the third model-driven turn unbought, though individual cells reach ±18 on the
deep running-back tail, where the curve is flat and the number matters least.

### What replacement means

VORP is **points above the best option you would field at that position for
free**. That is one definition, and it is the only one under which the numbers
are comparable across positions — but it has two arms, because the free option
is not the same kind of thing everywhere.

QB and TE are **streamable**. One starter each, ~14 of each drafted, so a
manager who punts the position does not roster one man all year; he starts
whoever looks best that week. Their replacement is a simulated streaming total.
RB and WR are **not**: ~49 backs and ~56 receivers come off the board, the pool
behind them is startable in name only, and no matchup rotation fixes it. Their
replacement is the measured best undrafted player. `tendies streaming` derives
all four from this league's own drafts:

| pos | replacement | how | curve value |
| --- | --- | --- | --- |
| QB | QB7 | streaming banks 257.2 pts | 255 |
| TE | TE6 | streaming banks 118.8 pts | 118 |
| RB | RB43 | mean best undrafted, 2019-2025 (RB42.6) | 89 |
| WR | WR51 | mean best undrafted, 2019-2025 (WR50.7) | 99 |

**This replaced a mixed definition, and the mix was doing real damage.** Until
2026-09-01 QB/TE were streaming-aware while RB and WR sat at roster-demand depth
— RB25/WR30, the last starter — which is 38 and 30 points better than what is
actually free. Every back read 38 too low and every receiver 30 too low, so the
board sorted quarterbacks and tight ends to the top of a column that meant a
different thing in each row: at pick 53 of the 2026 board Drake Maye showed 40
and Joe Burrow 30 against Christian Watson 19 and Quinshon Judkins 12. On the
repaired board the same five read Judkins 54, Watson 48, Maye 40, Burrow 30,
Kraft 9.

The evidence that this was a defect and not a preference is in the mock draft:
greedy best-available-**by-VORP** used to finish *behind* simply following ADP
(225.7 against 229.0 mean lineup value) and now finishes clearly ahead of it
(512.7 against 451.5). A value column that loses to ADP when you draft off it
was not measuring value.

The same repair removes a double-count. The empty-slot floor priced a punted QB
as "roster the best undrafted quarterback all season" (−5.5) and a punted TE at
−4.5, but QB/TE replacement is *already* a streaming total, so that charged the
streaming twice and paid the drafter for it. At a streamable position the floor
is now `max(0, best undrafted)`, and on the 2026 board all four floors land at
QB 0.0, RB 0.0, WR +2.5, TE 0.0 — the two mechanisms agree instead of arguing.
The `max` still lets the live board raise the floor if the room genuinely punts
a position.

**The QB/TE numbers carry uncertainty worth stating.** The streaming policy
form-streams the free pool, which hoards the year's breakout for free — it picks
up Mahomes in week 2 of 2018 and keeps him, which in a ten-team league one of
nine rivals would not allow. `drop_top` removes the season's biggest free-pool
scorers before streaming; `drop_top=1` is what ships, and the sweep is wide:

| | drop 0 | drop 1 (shipped) | drop 2 |
| --- | --- | --- | --- |
| QB | 305.9 pts = QB3 | 257.2 = QB7 | 239.9 = QB16 |
| TE | 137.2 pts = TE3 | 118.8 = TE6 | 100.2 = TE15 |

A reader who believes he wins the breakout waiver should be running QB3/TE3,
which prices quarterbacks and tight ends near zero all the way down the board.
That is a genuine disagreement about waiver skill, not a rounding choice, and
`tendies streaming` prints the whole grid rather than the shipped column alone.
RB/WR carry no such knob: their arm is a count off real drafts, tight across
seasons (RB 40-46, WR 47-53).

The walk-forward pick model barely notices any of it — `best_vorp` is a
per-position level so re-basing moves it, `vorp_gap` is within-position and is
invariant. Top-1 0.1998 → 0.1998, top-3 0.4404 → 0.4382, NLL 3.4533 → 3.4580,
position accuracy 0.4779 → 0.4790, all inside noise at n=906. That is expected:
this changes what the board is worth, not who the room takes.

### Does it work

Playing all fifteen rounds from each of the ten seats on the live 2026 board:
every starting slot filled on all ten, no cap exceeded, K taken in round 14-15
and D/ST in 14-15 with **no tuning for timing at all** — that falls out of the
DP spending its cheapest turns on the positions whose gain is zero, and the
cheapest turns are the last. Final lineup value beats cap-aware best-available
by our own VORP and by the market's on **10 seats out of 10**.

| | QB | RB | WR | TE | K | D/ST |
| --- | --- | --- | --- | --- | --- | --- |
| recommendation, mean | 3.0 | 3.1 | 5.2 | 1.7 | 1.0 | 1.0 |
| real managers, mean | 1.4 | 4.9 | 5.6 | 1.4 | 1.0 | 1.0 |
| real, observed range | 1-4 | 2-8 | 3-7 | 1-3 | 1 | 1-2 |

Every cell is inside the observed range. It is not trying to imitate managers,
and the gap it does not close is quarterbacks: three on every seat against a
real-manager mean of 1.4. That is **not** the replacement level talking — the
same walk on the pre-repair board took 2.9 — it is the late-round tie-break. By
round 12 nothing left moves lineup value, so `recommend` falls through to
ranking on insurance and market edge, and a third quarterback wins that on a
board where every remaining back and receiver is at the flat tail of his curve.

**Fixed 2026-09-02.** Three things were wrong at once, and the spare
quarterbacks were the visible symptom of all three.

*The endgame ordered on float dust.* By round 12 every candidate's score is
`vorp - ins[p]` with `vorp == ins[p]`, which lands at ~1e-10 rather than at 0.
The comparator then ordered the last rounds on that residue — one board ranked
a 148th-ADP back over a 127th-ADP receiver on 7e-11 of a point. `cmp` now
quantises at `SCORE_EPS`, which is what lets the rows be seen as the tie they
are. This is the same failure the section above describes at round 7 ("the sort
ran over an all-zero array, so V8's stable sort handed back the input order"),
surviving in the one stretch where all-zero is the honest answer rather than a
bug.

*Nothing then broke the tie except array order,* which is ADP order. There is
no upside model to break it with — deliberately, this data has an isotonic mean
curve and no per-player distribution — so it breaks on roster shape instead,
which needs no new number: among picks all worth nothing, prefer a position
where losing a starter would COST you a pick to cover. At a streamable position
it would not, which is what streamable means, so a spare back or receiver beats
a third quarterback as a lottery ticket.

*And `insuranceWeights` was circular,* which is its own entry below.

Measured over ten seats on the live 2026 board and on the 2025 fixture, the
three together leave final lineup value **unchanged** — 544.9 and 528.2, still
10 seats out of 10 — while moving the mix:

| | QB | RB | WR | TE | K | D/ST |
| --- | --- | --- | --- | --- | --- | --- |
| before, live 2026 board | 2.0 | 3.6 | 4.7 | 2.7 | 1.0 | 1.0 |
| after, live 2026 board | 1.0 | 5.8 | 5.2 | 1.0 | 1.0 | 1.0 |
| after, 2025 fixture | 1.0 | 5.7 | 5.3 | 1.0 | 1.0 | 1.0 |
| real managers, mean | 1.4 | 4.9 | 5.6 | 1.4 | 1.0 | 1.0 |

The spare tight ends went with the quarterbacks. Note the 2026 board is a
rolling window and did move during the session that produced these numbers, so
read the *before/after* pair, not the absolute levels.

What the replacement repair did move is tight ends: 22 taken across the ten
seats before, 17 after, with the capital going to RB (28 → 31) and WR
(51 → 52).

**What is approximate, and stated rather than buried.** The DP values future
acquisitions as slot fills, so it cannot see that a back acquired later would
displace a receiver already in your flex; displacement is handled exactly for
the pick you are actually making (through the cutoff machinery) and
approximately beyond it. The second simulation leg advances your own turn as a
pick of nobody, leaving the board a player or two over-available — which is what
the depth index absorbs. And the bench ranking is deliberately **not** an upside
model: there is no per-player distribution in this data, only an isotonic mean
curve, so it is insurance value plus the expert-versus-market gap, labelled as
such and never presented as lineup points.

### Why the page will not draft a tight end, and the fix that failed

Raised 2026-09-02 from a real board: round 12, an **empty TE slot**, Kelce,
Kincaid, Goedert and Likely all still available, and the panel recommending a
dart-throw running back. Two mechanisms stack, and the first is a genuine gap in
the objective.

**An occupied slot is priced `max(his season total, the floor).`** Every
available tight end was VORP −10.2 against an empty-slot floor of 0.0, so
`max(−10.2, 0) = 0`: rostering Travis Kelce priced *identically to leaving the
slot empty*. Not slightly better — identically. There is no gradient at all, so
no tight end below the streaming line can ever earn a pick, and Kelce, Goedert,
Likely and Hunter Henry are one number.

**And that line is the streaming total.** `REPL_RANKS[TE] = 6` is 118.8 points;
the best draftable tight end is worth ~108. So by construction every tight end
left on the board is below replacement.

The first mechanism looks straightforwardly wrong, because `max` of season
totals is not the season total of WEEKLY maxima — roster a tight end and you do
not stop streaming, you start whichever of him and the best free tight end looks
better that week. Holding both should be worth strictly more than either. So a
per-position option-value constant was derived, measured with the same
no-hindsight policy `stream_total` uses (prior-weeks form only, `drop_top=1`),
and **it did not survive.** Pooled over every drafted slot and season, in the
region where `max()` actually flattens the player (his own total at or below the
floor):

| | n | gain | t | per season |
| --- | --- | --- | --- | --- |
| TE | 64 | **+1.61 ± 3.80** | 0.42 | −1 −29 +29 +46 −29 +36 −32 +1 |
| QB | 42 | **+5.70 ± 6.07** | 0.94 | −1 +46 +79 −71 +2 −9 +5 +2 |

Across *all* drafted slots the gain is **negative** (TE −5.10 ± 2.84), because
the form rule benches a stud on two hot weeks from the waiver pool where a real
manager would not. A single slot in isolation reads +4 to +9 — TE13 reads +4.7,
which is what made this look shippable — and that is one draw from a noisy
surface. The constant would have to be ~6 points to change a decision, and 6 is
1.6 SE from zero. Nothing was added to the objective; the derivation prints
under `tendies streaming` so the null is reproducible rather than folklore.

**What this does not settle**, and it is the part that matters: the floor's own
uncertainty is **35 points of curve** — the `n_owned` sweep spans TE4 to TE15 —
so a test with an SE of 3.8 cannot tell you whether TE6 is the right
replacement rank. At `drop_top=2` replacement is TE15, every one of those tight
ends is *positive*, and the panel wants one immediately. The parent repo reached
this same fork and went the other way on purpose: its notes record the TE
streaming sim as "central ~TE5-6 but optimistic for the thin TE pool" and it
ships TE8. `tendies` took the sim's central value. That is a live judgment call
about how often you win a waiver breakout, not a bug, and it is the lever that
decides whether mid-round tight ends are draftable at all.

Worth keeping in view: the panel is right that you cannot *miss out*. All five
draftable tight ends sat on one curve block, and Jake Ferguson was 86% to
survive fifteen picks, Juwan Johnson 94%, Hunter Henry 98%. The plan already
read `#128 TE → #133 K → #148 DST`.

### The resolution floor

Added 2026-09-02, after the panel was caught presenting a ranking the curve
cannot support. **The curve resolves the top of the board and not the back of
it**, and until now nothing downstream knew that.

A slot mean averages ~14 seasons of one draft position, and at the back of the
board the season-to-season spread swamps any slot-to-slot trend. WR30 has
ranged from 36.4 to 251.3 half-PPR points — sd 64.6 on a mean of 127.6 — so a
single slot's mean carries an SE of 15-19 points. Regress points on slot across
WR30-39 and the estimate is **+1.50 pts/slot (SE 1.66, t +0.90)**: the point
estimate slopes the *wrong way* and the 95% interval on the WR30 → WR39 change
runs −16 to +43. There is no measurable decline there, which is exactly why
PAVA pools those ten slots into one block — and pooling is the estimator
working, not failing: 140 observations drop the block's SE to 4.8 points
against 15-19 for any slot alone.

What that costs in decisions, on the 2012-2025 history behind the 2026 board:

| comparison | Δ VORP | 95% | t | resolved |
| --- | --- | --- | --- | --- |
| WR30-39 (29.2) vs WR50-54 (0.0) | +32.2 | ±8.2 | 3.92 | **yes** |
| RB34-36 (34.6) vs RB42-46 (0.0) | +19.1 | ±10.6 | 1.81 | no |
| RB34-36 (34.6) vs RB37-41 (6.9) | +12.2 | ±11.2 | 1.09 | no |
| WR30-39 (29.2) vs WR40-42 (21.4) | +7.8 | ±9.4 | 0.83 | no |

Row two is the *entire* spread of available backs at a round-8 pick, best to
worst, and it does not clear. So `vorp.py` now ships `vorp_se` and `edge_se`
with every board row — the SE of a difference of two block means, exactly 0
when PAVA pooled both slots into one block — and the panel groups candidates it
cannot separate under one shared rank number, quotes the band, and says so in
words.

**A tie bounds what the panel may claim; it is not a licence to reorder.** That
distinction was learned the expensive way. `cost` (expected lineup points lost
by waiting) is the one number on a candidate row that is not a difference of
two isotonic blocks, so the first version ordered the whole 1.96-SE band by it.
That cost **24.5 mean lineup points** (528.2 → 503.7) and took seats beating
best-available from 10/10 to 4/10. Inside the band the point estimate is still
the best estimate available. Restricting the same idea to *exact* ties measures
as a coin flip — +0.0, −2.0, +0.3, +0.0 across four board snapshots — so `cost`
is displayed on every row and ordered on by nobody.

Two scope limits, stated rather than buried. The tie test applies in **bench
mode only**: there the score is a difference of curve blocks and the SEs are in
the right units, whereas `ev = now + later` is a plan value whose `later` term
moves *against* `now`, so banding it with the players' own slot SEs over-ties —
it called Ja'Marr Chase (ev 555.1) and Brock Bowers (516.0) indistinguishable
on the opening pick, in the one region where the curve demonstrably does
resolve. Banding `ev` honestly means propagating curve error through the plan
DP; until someone does, starter mode claims no tie beyond exact equality. And
`insurance`'s cutoff term carries uncertainty of its own that is not
propagated.

**Do not "fix" a flat block by adding an ECR tie-break inside it.** It is the
first thing anyone reaches for on seeing ten identical numbers, and it invents
signal the history rejects at t = 0.90.

### `insuranceWeights` was circular

Found 2026-09-02, in the same pass. Each position's insurance weight was
`|empty[p]| / max |empty|`, documented as "how much a lost starter costs,
normalised — which is the EMPTY floor again, so no new number is invented
here". The number was not invented; it was **circular**. `empty[p]` is the VORP
of the best undrafted player at p, and VORP is defined relative to replacement,
which `vorp.py` sets to *that same best undrafted player*. So `empty[p]` is ~0
by construction, its magnitude is residual curve-shape noise, and dividing
noise by noise collapsed the term: on a real round-8 board in this 10-team
league, insurance read **0.0 for every candidate at every position**, so the
bench ranking was market edge alone while the panel claimed it was both.

The replacement needs no new number either. Insurance is 0 where there is no
curve (a backup kicker is worth what the starter is worth) and 0 at a
**streamable** position — `STREAMABLE` means the free in-season option *is*
replacement level, so a pick spent on a backup quarterback buys what waivers
would have given you. Elsewhere it is 1, and `insuranceCutoffs` does the rest.

A related gate landed with it: **market edge counts only while a player beats
his position's free end-of-draft floor.** `edge` is a difference of two curve
slots, so it grows without bound down the steep tail of the QB curve; with the
candidate set capped this never showed, but on the full board a quarterback
nobody would roster (VORP −99) scored +56 of "market edge" and outranked the
best available back. A bargain on a worthless asset is worthless.

---

## Train/serve parity

The model is fitted in Python and scored in the browser, so the two
implementations are held to an exact standard rather than a plausible one.
`tests/parity.mjs` replays golden draft states — empty board, first pick, mid
draft, deep endgame — through `web/engine.js` under the node binary bundled
with Playwright, and compares against Python.

**Agreement is 7e-16 at full precision** (machine epsilon); the shipped
artifact rounds coefficients to 9 significant digits for diffable files, which
puts the practical bound at 1e-9. The RNG (`mulberry32`) is implemented in both
languages and asserted to produce identical streams, so a simulation is
reproducible across languages pick for pick — the same board state always shows
the same numbers.

That test earned its keep immediately: it caught a dropped XOR in the Python
RNG that produced a perfectly random-looking but different stream. It earned it
again on 2026-09-01: the per-manager rookie term was added to `train` and to
`engine.js` but missed in `predict.player_utilities`, and the only symptom
anywhere was a 3.8e-6 position-probability diff in this test.

Two guards sit either side of it, because parity alone cannot see everything:

* `assertContract` compares the FEATURE NAME LISTS in the payload against
  `engine.js`'s own constants, so a feature added in Python and forgotten in JS
  throws at page load rather than shifting every coefficient by one slot.
* `assertBoard` and `tests/contract.mjs` cover what `assertContract` cannot:
  board columns, and the WIDTH of each per-manager coefficient block. Those
  blocks were sliced open-endedly out of the coefficient vector (`gamma[off:]`)
  until 2026-09-01, so a coefficient appended after the last block would have
  been absorbed into it silently — and parity could not have caught that
  either, because Python and JS would have read the same wrong array.

The payload `schema` is **3** as of the second tendency round, which also
replaced the fixed `player.reach` / `player.rookie` arrays with an ordered
`player.channels` list carrying `{name, feature, beta}`. JS resolves each
channel's column by NAME from that list, so adding an eighth is a Python-side
table entry with no JS edit — and `tests/contract.mjs` asserts every channel is
exactly `n_managers` wide and names a feature that exists.

---

## What is optimistic, and cannot be fixed here

Stated plainly rather than in a footnote.

1. **The specification was chosen while seeing all six evaluated seasons.** The
   feature list, the factorisation, the candidate windows and the decision to
   model autodraft as a Markov chain were all selected with full-sample
   knowledge. Nested selection protects hyperparameters inside each fold; it
   does nothing for the specification. The parent repo has three direct
   measurements of this kind of selection bias on real test data; budget a
   couple of points of the reported gain as optimism.
2. **No reserved test set.** By choice — 906 picks is little enough that
   holding out three seasons would have left the tuning surface too thin. The
   correct label for these numbers is "walk-forward with in-sample
   specification", not "held out".
3. **No held-out manager.** Manager coefficients are fitted on ≤107 picks each
   and evaluated on the same 13 people. Given that they are worth ~0.3pp, this
   matters less than it would if they had worked.
4. **Six folds.** A sign-flip test bottoms out at p = 0.016 one-sided.
5. **Board vintage.** FantasyPros restates historical boards — that is how
   Robby Anderson appears as "Robbie Chosen" on the 2019 board. The market
   feature in training may be slightly sharper than what was visible on draft
   day. It inflates the model and the baselines alike.

---

## Reproducing

```bash
cd tendies
uv run python -m tendies build            # picks (needs ESPN cookies, see README)
uv run python -m tendies ecr-check        # audit the expert ranks; exits 1 on a failure
uv run python -m tendies traits           # per-manager traits + split-half reliability
uv run python -m tendies model            # fit, print coefficients
uv run python -m tendies evaluate --b4 --ablate   # headline + per-channel leave-one-out
uv run python -m tendies site --season 2026
uv run pytest tests/                      # includes Python/JS parity
```
