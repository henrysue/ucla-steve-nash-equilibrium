# Handoff: VGC Draft Strategist

State as of 2026-10-09, after the "basics first" simplification and the predict-then-answer
battle turn. 1373 tests pass
(`.venv\Scripts\python.exe -m pytest tests/ -q`). Platform rules: `GAMES.md`.
Pre-simplification code is not in git; the original design spec is
`vgc-draft-strategist.md` (it describes layers since removed: opponent model,
exploit check, game trees).

## What this is

An agent for the Altruagent `pokemon_vgc_doubles_draft` game, built on the
`altruagent` starter. Entry point: `agent.agent:create_agent` -> `VGCDraftStrategist`
in `agent/vgc/`. Three decisions, each with a deterministic fallback:

| Decision | Model call | Code around it | Fallback |
|---|---|---|---|
| Draft pick (9s) | picks one of the 12 best-fitting cards | `draft.score` per card: team plan, roles, coverage, matchup vs their picks (types, counters, playstyle), strength | top score |
| Team Preview (78s) | ONE call: match plan + bring/leads | lineup rules checked; a breaking answer is re-asked once, the retry is played as given | best legal four |
| Battle turn (40s, thinking off) | predicts each foe's action (protect / attack whom / Fake Out / switch / speed control / setup) with a confidence | 1. `rules` withholds/rejects blunders; 2. `predict.code_guess` first read (carries Protect and did not Protect last turn -> will Protect; else attacks the one of ours it hurts most), refined by the model; 3. `lookahead.ranked_lines` plays our best legal answer to that prediction | the answer to the code's guess |

The model reads the opponent; code enforces the rules and picks our move.

## Run it

```powershell
.\play.ps1 --check-tournament     # readiness; prints which model it will use
.\play.ps1 --tournament
```

Use `play.ps1`: it pins the venv. Config is in `.env` (see `.env.example`):
`ANTHROPIC_API_KEY`, optional `VGC_MODEL`, `VGC_LOG_DIR`, `VGC_META_CSV`.

## Layout (`agent/vgc/`)

| File | Role |
|---|---|
| `strategist.py` | The three decisions, the `_ask` retry loop, fallbacks, audit recording |
| `cards.py` | Exact sets from the draft board: every card (species, item, ability, nature, EVs, four moves) is remembered, saved to `data/card_sets.json` (`VGC_CARD_SETS`; empty = off) and put in front of the reference table, so foe moves/items/abilities/stats are the card's, not usage guesses |
| `predict.py` | Step 2: each foe's most likely action (code's guess, the model's prediction) as probabilities |
| `decisions.py` | The three prompts and strict schemas |
| `draft.py` | One score per draft card |
| `archetypes.py` | Team plans (`data/team_plans.json`), opponent playstyle reads, playstyle counters |
| `rules.py` | The one rules layer for a turn: withheld options + row checks |
| `moverules.py` | Per-match move history: Fake Out after first move, Protect chains, Blood Moon, opening Fake Out lock; which foes protected last turn (server's last move, else: we hit it and its HP did not drop) |
| `movelist.py` | Move data (`data/moves.json`) and the blunder checks (immunity, partner hits, outclassed moves, Stamina chips, resisted targets) |
| `lookahead.py` | L50 damage estimates, threat tables, `ranked_lines`/`best_line`: our best legal answer to the prediction (one-turn values pick 24 candidate pairs; `race.py` ranks them) |
| `race.py` | Two-turn damage race (turn 1 vs the prediction, then 2 follow-up turns of best attacks; KO-range-of-a-faster-foe counted half lost at the end) |
| `style.py` | Opponent style: basic (Protects whenever it can, never twice) vs strong (holds Protect when it looks obvious). Each Protect opportunity seen updates P(strong) by Bayes; the blend sets the Protect odds, the code guess and the model's `opponent_style` fact; saved per opponent agent in `data/opponent_styles.json` (`VGC_OPPONENT_STYLES`) |
| `observed.py` | What happened last turn (HP, faints, switches, reveals, PP) rebuilt from consecutive observations; each prediction checked hit/miss/unknown; field parsing |
| `protect.py` | Each foe's chance to Protect this turn |
| `abilities.py` | Target-ability warnings on each move option |
| `actions.py` | Template parsing, option fact sheets, payload building, guaranteed-legal floor |
| `meta.py` | Reference tables; picks the table matching the dealt species |
| `report.py` | `MatchLog`: per-match markdown + JSONL |
| `providers.py` | Claude / OpenAI with a hard per-request deadline |

Data: `vgc_reg_mc_draft_doubles.csv` (Champions Reg M-C, Megas) and
`vgc_2025_reg_i_draft_doubles.csv` (Reg I). The agent uses whichever covers the
dealt species (`meta.best_table`), so keep both. Logged matches through 2026-10-09
dealt Reg I species. `moves.json` is built by `scripts/build_movelist.py` from
Showdown's move data. `vgc_2025_reg_i_archetypes.json` is no longer read.

## Settings worth knowing

| Setting | Value | Why |
|---|---|---|
| Model | `claude-sonnet-5` | User's choice |
| Thinking | Off for battle turns (`effort="none"`), low for draft/preview | Thinking-on turns ran 13-40s and timed out; off: 6-10s |
| Public summary | `Move submitted.` by default; `--show-reasoning` (or `VGC_PUBLIC_REASONING=1`) sends the one-line reasoning instead (battle: `Read: <their predicted actions>. Play: <our move>. <note>`, max 280 chars) | Public text is visible to spectators and possibly the opponent; the model's private thinking is never sent |
| Answer scoring | `race.py` ranks the candidates: damage dealt minus taken over turn 1 + 2 follow-up turns (+50 per expected KO), in speed order with Fake Out, Protect, switches, Helping Hand, redirection, Tailwind/Trick Room/screens, speed drops, sleep/burn/paralysis, boosts, Gale Wings/Prankster. Candidates come from one-turn values: a likely KO = remaining HP + 50 (removing a Pokemon, not just its HP); Protect = predicted damage blocked x its chance of working (1/3 for a second in a row), a delayed KO counting a quarter of the bonus; attack = damage x chance it does not Protect (x0.5 into a switch, x chance we survive to act if slower and predicted KO'd); Fake Out adds the damage it prevents; Protect = predicted damage blocked; switch = damage moved onto a safer Pokemon; setup = valuable when every foe is predicted to Protect | Rock-paper-scissors: we play the counter to the read |
| Prediction source | the model's prediction when valid, the code's guess otherwise (logged as `prediction_source`) | |
| Lineup rules | >= 3 attackers, Fake Out user leads, a Tailwind/Trick Room setter brought, no lead pair whose spread move hits the other, <= 1 restricted/Mega | Each came from a logged loss |
| Opening turn | weaker-offense Fake Out lead is locked to Fake Out | Skipped vs Ghosts / Indeedee / Farigiraf |
| Attempts | 2; a retry only with half the budget left | Late retries never finished |

## Hard rules to preserve

1. **A model problem must never cost a legal move.** `_ask` catches `Exception`; every
   decision prepares its fallback first; `_last_resort` plays a guaranteed-legal move.
2. **`model_reasoning` is private.** The platform gets `Move submitted.`, or with `--show-reasoning`
   the one-line summary; never the model's thinking.
3. **The server is authoritative.** Tables are prior knowledge.
4. **Log with `altruagent.console.print_line`, not `print`.**
5. **Absence is normal.** Missing species/fields return `None`/`[]`.
6. **New blunder found? Make it a rule plus a test from the real board** (`movelist.row_problem`
   or `option_blocks`), not prompt advice.

## Environment trap

Two interpreters exist: `.venv` and a global Python 3.14. The global `anthropic` once
lacked `output_config` and crashed matches; `AnthropicProvider` now refuses such an
SDK at startup. If something works in the venv but not for the user, compare package
versions across both interpreters. `tests/conftest.py` blanks `VGC_LOG_DIR` so tests
never write to `matchlogs/`.

## Known limits

- Sets come from draft cards (platform spec, api.altruagent-game.com/skill/pokemon: each
  card is a fixed set; rosters list only card_id + species, so the board must be
  remembered). Seat 1 never sees the card taken at pick 1 in that match; the saved
  catalog covers it if it was on an earlier board. Keyed by species: the spec does not
  promise one card per species (card ids `vgc-<species>` suggest it); this match's
  board always wins over the saved file. With no card, sets fall back to revealed
  moves, the table's common moves, then a generic 90-power STAB.
- Damage: card EVs/nature (else estimates), items (incl. resist berries, consumption not
  tracked), Ruin abilities; Body Press uses the user's Defense, Foul Play the target's
  Attack, Psyshock/Psystrike/Secret Sword hit Defense. No weather, terrain or other
  abilities. Base stats from the table, else the server's `base_stats`.
- Field effects are read from the observation's `fields`/`side_conditions` keys (a key's
  presence = up; values are start turns, 0 for a lead's terrain). Psychic Terrain withholds
  priority attacks on grounded foes; Prankster status moves are withheld on Dark types.
- The race charges recoil (Wood Hammer, Flare Blitz...), Rocky Helmet on contact and Life
  Orb; follow-up turns never use Fake Out / First Impression.
- Which foe to hit (resisted, Stamina) is the race's call, not a hard rule: Stamina adds
  +1 Defense per landed hit in the race (weaker physical hits, stronger Body Press). Only a
  multi-hit move into Stamina is always rejected.
- Race: Eviolite (x1.5 both defences), Focus Sash (chance the holder is untouched), recoil.
  Later turns: our gains count 0.85 per turn (a KO now beats one next turn), our losses in
  full (delaying their hits with Protect earns nothing by itself).
- Protect read: a newly revealed Protect counts; otherwise HP is compared with the least
  damage our attacks should have done (a Leftovers rise after a blocked hit = Protect).
- Team Preview / draft matchups (`draft.versus`) use each side's real damaging moves
  (60+ power or variable) when the set is known, else STAB types; a 4x threat counts double.
- Search: a two-turn race in expected values (no branching, no local Showdown
  simulator). Turn 2-3 assume everyone just attacks; the opponent's switch-ins, our
  bench's later turns, abilities beyond Ruin/Gale Wings/Prankster, weather and terrain
  are not simulated. Unsimulated status moves (Taunt, Encore, Parting Shot...) get a
  flat 10. "Survives to act" uses card or
  estimated Speed with boosts and Choice Scarf, but not Tailwind/Trick Room.
- We best-respond to the prediction, so we are as predictable as our reads: a
  good opponent who models us can exploit that. The old Nash/mixing layer is in git
  history (`agent/vgc/nash.py`) if that becomes a problem.
- "Protected last turn" is inferred from HP when the server does not report the
  last move: only UNCHANGED HP counts (a rise is healing, not a Protect), and only if
  the Pokemon that attacked it is still on the field (one KO'd before moving never
  hit). A heal that exactly cancels a light hit can still fool it.
- Our timed effects (Tailwind 4 turns; Reflect, Light Screen, Aurora Veil, Trick Room 5)
  are tracked from our own moves and a re-use is withheld while up (Trick Room's would end
  it). Base durations only (no Light Clay), counted only if the setter was still out next
  decision. The opponent's effects, weather and terrain are not tracked.
- Team Preview rosters show the opponent with placeholders (`unknown_item`, no moves):
  those never overwrite a draft card; a roster entry only fills a card's gaps.
- A Protect while our other slot does nothing (lone active, or both protecting) is
  worth `STALL_SHARE` (0.25) of its value: it only delays the same hits. The observation
  has no field timers, so stalling out their Tailwind/Trick Room is not valued.
- An attack every foe resists (<= 0.5x) is withheld when a switch or a better attack
  exists, which is how a bad Choice lock gets switched out of.
- The runner hook `on_match_end` was added to `altruagent/runner.py` (starter code);
  re-add it if the starter is updated from upstream.
- `.env` pins `VGC_META_CSV` to Reg I; the per-match auto-switch overrides it.

## Audit log

With `VGC_LOG_DIR=./matchlogs`, each match writes `<stamp>-<game>-<session>-seat<n>.md`,
`.jsonl` and `.summary.json` (with the outcome). Per decision: payload, attempts,
seconds, fallback reason, matrix, equilibrium, rows the rules removed, `move_state`
(move history, Protect odds, our side effects' turns left) and `last_turn` (what happened
since the previous decision and whether each foe's predicted action was a hit, miss or
unknown). The summary totals the reads per predicted action with a hit rate. The raw
field values are logged too (`raw_field`), to confirm their shape. Draft picks log the
whole board (`board`, JSONL only) and the summary logs `ending` (the server's
final_result and the last board's HP). `scripts/export_card_pool.py` turns the card
catalog plus all logs into `data/card_pool.csv` (one row per species seen: exact set,
types, stats, roles, tier, how often we picked/faced it, our record with it). Files named
`*-draft-s-seat0.*` are old test artifacts.
