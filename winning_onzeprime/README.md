# winning_onzeprime

Draft toolkit to study and play [onzeprime.com](https://onzeprime.com) → **Três Grandes** (Benfica, FC Porto, Sporting) and chase the record-level XI.

## Key findings

- **All rules and ratings are client-side.** They live in one JS bundle (`js/onzeprime.js`). Its `core.js` part has no DOM code. So the full player pool is mapped from that file, without scraping games.
- **The engine is deterministic.** Each game is `mulberry32(seed)` plus your actions. `onzeprime/engine.py` is a port that matches the site's own engine bit for bit (`tests/test_parity_js.py`).
- **"90,5" is the theoretical maximum.** The site shows `avg.toFixed(1)`, and the best possible Três Grandes XI sums to 995, or 995/11 = 90.4545, which displays as 90,5. Only three formations can reach it: **4-4-2, 4-2-4 and 3-5-2**.
- **Odds:** in offline simulation, the `ceiling` policy reaches 995 in about 1 of every 3,000 games (3-5-2).

## Layout

| Path | What |
|---|---|
| `onzeprime/data.py` | Downloads and caches the bundle, then parses clubs and players |
| `onzeprime/engine.py` | Python port of the game rules (RNG, draws, slots, swaps, replay) |
| `onzeprime/optimizer.py` | Best XI per formation (assignment problem) and top players per role |
| `onzeprime/strategies.py` | Policies: `greedy`, `elite` (top-k bar per role), `ceiling` (keeps the reachable max high) |
| `onzeprime/simulate.py` | Monte Carlo runs, grid search, parallel record hunt |
| `onzeprime/training.py` | Training loop: survival metrics, early-abort simulation, knob search, validation, time plan |
| `onzeprime/live_bot.py` | Live play: `hunt` (autopilot) and `copilot` (you play, it advises) |
| `notebooks/onzeprime_lab.ipynb` | Exploration, tests, simulations and hunt |
| `notebooks/onzeprime_training.ipynb` | The training process, step by step, with time estimates |
| `tests/` | pytest suite, including parity against the real JS engine |

## Setup

```bash
cd winning_onzeprime
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest -q
```

## Run

```bash
# offline record hunt (thousands of games, no network after the first data download)
.venv/bin/python -c "from onzeprime import simulate as s; r=s.hunt('3-5-2','ceiling',{'target_sum':995},n_games=20000,workers=7); print(r['best_avg'], r['lineup'])"

# live autopilot: restart in-page whenever 995 becomes impossible, stop at the record
.venv/bin/python -m onzeprime.live_bot hunt --formation 3-5-2 --attempts 10000 --minutes 90 --headed --keep-open
# add --slow-mo 150 to slow every click down so it is easy to follow (~2 s per attempt instead of ~0.5 s)

# live copilot: YOU play in a visible browser, an on-page box tells you the pick / swap / restart
.venv/bin/python -m onzeprime.live_bot copilot --formation 3-5-2 --minutes 60
```

### Live bot safeguards

The autopilot blocks `/api/live`, `/api/ranking`, `/api/partilha`, `/api/visitas`, `/api/og` and analytics. Bot games don't add to the public "plantéis feitos" counter, and they **never reach the ranking**. Without a server seed, the game falls back to a local random seed. Restarts happen in the page (no reloads), so after the first page load a hunt sends no traffic to the site. The copilot changes nothing: it only reads the page and shows advice, so your games are normal games.

## Browser MCP (optional, interactive)

`.mcp.json` registers the official **Playwright MCP** (`@playwright/mcp`, tools such as `browser_navigate`, `browser_click`, `browser_snapshot`). It lets Claude Code drive the site step by step when started from this folder. It needs Node.js: `brew install node`. For hundreds of games, use the Python Playwright bot above. It is the same engine, scriptable and much faster than tool calls.
