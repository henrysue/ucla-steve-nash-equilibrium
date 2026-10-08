# Handoff: VGC Draft Strategist

State as of 2026-10-08. 931 tests pass (`.venv\Scripts\python.exe -m pytest tests/ -q`).
The spec is `vgc-draft-strategist.md`; the platform rules are `GAMES.md`.

## What this is

An agent for the Altruagent `pokemon_vgc_doubles_draft` game (format `gen9vgc2025regi`),
built on the `altruagent` starter. Entry point: `agent.agent:create_agent`, which
returns `VGCDraftStrategist` from `agent/vgc/`. It plays three phases: draft picks,
Team Preview lineup, and battle turns.

The design splits the work. **The model only estimates payoffs**: it proposes a few
candidate lines for each side and a win-probability matrix. **Deterministic code does
the rest**: Nash equilibrium (`nash.py`), opponent modelling and the "cheese" check
(`opponent.py`), and sampling the mixed strategy. Do not move equilibrium or risk
arithmetic into the prompt.

## Run it

```powershell
.\play.ps1 --check-tournament     # readiness; prints which model it will use
.\play.ps1 --tournament
```

Use `play.ps1`: it pins the venv. Plain `python -m agent` hits the global Python.
Config is in `.env` (see `.env.example`): `ANTHROPIC_API_KEY`, optional `VGC_MODEL`,
`VGC_LOG_DIR`, `VGC_META_CSV`.

## Layout (`agent/vgc/`)

| File | Role |
|---|---|
| `strategist.py` | The agent. Phase routing, the `_ask` retry loop, fallbacks, audit recording |
| `decisions.py` | Prompts, JSON schemas, and **per-decision effort** |
| `providers.py` | `AnthropicProvider`, `OpenAIProvider`, `provider_from_env` |
| `meta.py` | Loads the reference CSV; restricted-legendary lookups; counter/partner cross-refs |
| `nash.py`, `opponent.py` | Equilibrium solve; opponent posterior and exploit gate |
| `actions.py`, `typechart.py` | Building/parsing platform actions; heuristic fallback moves |
| `report.py` | `MatchLog`: per-match markdown + JSONL audit trail |

Data: `data/vgc_2025_reg_i_draft_doubles.csv` (472 rows, the live table).
`data/vgc_reg_mc_draft_doubles.csv` is the old Reg MC table, unused except as a
test fixture for column-rename tolerance.

## Current settings (and why)

| Setting | Value | Reason |
|---|---|---|
| Model | `claude-sonnet-5` | User's choice; cost |
| Effort | `low` for draft, preview, and turns | **Measured:** doubles turn is 10-14s at low, ~40s at medium |
| Budgets | draft 9s, preview 78s, turn 40s | Platform: 15s / 90s / 55s from a shared bank |
| Attempts | 2, then deterministic fallback | An escaping exception forfeits the match |

Do not set draft effort above `low`: a 9s budget cannot fit a medium call.

## Hard rules to preserve

1. **A model problem must never cost a legal move.** `_ask` catches `Exception`
   deliberately and logs the type. Every phase has a heuristic fallback.
2. **`model_reasoning` is private.** It goes to the local log only, never into
   `reasoning_summary` (public to spectators). Tested.
3. **The server is authoritative.** The CSV is prior knowledge. Never hard-code
   rules the server decides, such as how many restricted legendaries are legal.
   Reg I normally allows one; prompts and fallbacks steer toward it but say so as an
   assumption.
4. **Log with `altruagent.console.print_line`, not `print`.** Concurrent match workers
   share a terminal; `print` interleaves lines. The starter's tests enforce this.
5. **Absence is normal.** Species missing from the CSV, or blank draft points (below
   the top 50), return `None` and the agent reasons without them.

## Environment trap

Two interpreters exist: `.venv` and the global Python 3.14 (which also hosts vllm and
gptqmodel). The global `anthropic` was 0.75.0 and crashed matches with
`TypeError: unexpected keyword argument 'output_config'`. It was upgraded to 1.12.1
with the user's approval (rollback: `pip install 'anthropic==0.75.0'`).
`AnthropicProvider` now refuses an SDK without `output_config` at startup and
`provider_from_env` falls through to OpenAI or heuristics. If something works in the
venv but not for the user, compare package versions across **both** interpreters first.
A pre-existing `protobuf` vs `gptqmodel` conflict is not ours; leave it.

## Known limits

- **Haiku 4.5 does not work.** It returns HTTP 400 because the provider sends
  adaptive thinking and `effort`. Supporting it needs a separate request path with
  neither parameter.
- **OpenAI path is thin.** `gpt-4o-mini` is a weak fit for payoff estimation, and
  Chat Completions exposes no reasoning, so the audit log's reasoning section is empty.
- **Reasoning is empty at low effort** even on Claude (0 chars measured). The matrix,
  equilibrium, and verdict are still logged.
- **Timeouts retry at one lower effort** (`FASTER_EFFORT`), but at `low` there is
  nothing lower, so a slow call that times out then falls back.
- **Live-match behaviour is unverified.** Everything is tested against fakes and
  single real calls, not a full tournament game. The first real match is the real test.

## Audit log

Set `VGC_LOG_DIR=./matchlogs`. Each match writes `<stamp>-<game>-<session>-seat<n>.md`
and `.jsonl`: model, attempts, seconds, fallback reason, payoff matrix, equilibrium,
opponent posterior, exploit verdict, and the action sent. Check `fallback_reason`,
`attempts`, and `seconds` first when a move looks wrong.

## Suggested next steps

1. Play one real match with `VGC_LOG_DIR` set; read the JSONL for fallbacks and timings.
2. If turns finish well under 40s, consider `medium` effort on Team Preview only
   (78s budget) to recover reasoning in the log.
3. Fill draft points for sub-top-50 species if the league prices them.
4. Delete `data/vgc_reg_mc_draft_doubles.csv` once the rename-tolerance test no
   longer needs it.
5. Nothing is committed: the whole `agent/vgc/` tree, `data/`, tests, and
   `play.ps1` are untracked in git.
