"""Offline Monte-Carlo: play many games with a policy and collect results."""
from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from .data import Club, Player
from .engine import Game

# The ranking shows "90,5", but the site rounds with toFixed(1): the real record is
# 995/11 = 90.4545..., which is exactly the theoretical best Três Grandes XI.
RECORD_SUM = 995
RECORD = RECORD_SUM / 11


def play(clubs, by_club, policy: Callable, formation: str, seed: int, mode: str = "grandes") -> Game:
    g = Game(clubs, by_club, mode=mode, formation=formation, seed=seed)
    guard = 0
    while not g.finished and g.draw is not None:
        d = policy(g)
        if d.kind == "swap":
            g.swap()
        else:
            g.choose(d.player, d.slot_id)
        guard += 1
        if guard > 40:
            raise RuntimeError(f"policy stuck at seed {seed}")
    return g


def run_many(clubs, by_club, policy, formation: str, n: int = 1000, seed_start: int = 0,
             mode: str = "grandes", keep_lineups: bool = False) -> pd.DataFrame:
    rows = []
    for seed in range(seed_start, seed_start + n):
        g = play(clubs, by_club, policy, formation, seed, mode)
        row = {"seed": seed, "formation": formation, "policy": getattr(policy, "name", "?"),
               "sum": sum(p.ovr for p in g.xi.values()), "avg": round(g.average(), 4), "swaps_used": 2 - g.swaps}
        if keep_lineups:
            row["lineup"] = g.lineup()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, target_sum: int = RECORD_SUM) -> pd.Series:
    return pd.Series({
        "games": len(df), "mean": df.avg.mean(), "p50": df.avg.median(), "p95": df.avg.quantile(.95),
        "max": df.avg.max(), "P(record)": (df["sum"] >= target_sum).mean(),
        "P(>=990)": (df["sum"] >= 990).mean(),
    })


def grid_search(clubs, by_club, make_policy: Callable[..., Callable], formation: str,
                grid: Dict[str, Iterable], n: int = 500, seed_start: int = 10_000) -> pd.DataFrame:
    """Evaluate every parameter combination on the same seeds (common random numbers)."""
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        t = time.time()
        df = run_many(clubs, by_club, make_policy(**params), formation, n, seed_start)
        s = summarize(df)
        out.append({**params, **s.to_dict(), "secs": round(time.time() - t, 1)})
    return pd.DataFrame(out).sort_values("mean", ascending=False).reset_index(drop=True)


def _hunt_chunk(args):
    """Worker: rebuild data + policy in-process, play a block of seeds, return compact results."""
    from . import data, optimizer
    from .strategies import POLICIES

    formation, policy_name, params, seeds, mode = args
    clubs, by_club = data.load()
    pool = optimizer.pool_for_mode(by_club, mode)
    pol = POLICIES[policy_name]() if policy_name == "greedy" else POLICIES[policy_name](pool, formation, **params)
    rows, best = [], None
    for seed in seeds:
        g = play(clubs, by_club, pol, formation, seed, mode)
        tot = sum(p.ovr for p in g.xi.values())
        rows.append((seed, tot))
        if best is None or tot > best[1]:
            best = (seed, tot, g.lineup(), g.actions)
    return rows, best


def hunt(formation: str, policy_name: str = "ceiling", params: Optional[dict] = None, n_games: int = 20_000,
         seed_start: int = 0, workers: int = 4, chunk: int = 500, mode: str = "grandes",
         target_sum: int = RECORD_SUM, stop_on_hit: bool = True) -> dict:
    """Play up to n_games seeds in parallel, looking for a game that reaches target_sum.

    Returns all (seed, sum) results plus the best game's lineup and actions.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    params = params or {}
    jobs = [(formation, policy_name, params, range(a, min(a + chunk, seed_start + n_games)), mode)
            for a in range(seed_start, seed_start + n_games, chunk)]
    rows, best, t0 = [], None, time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_hunt_chunk, j) for j in jobs]
        for f in as_completed(futs):
            r, b = f.result()
            rows.extend(r)
            if best is None or b[1] > best[1]:
                best = b
            if stop_on_hit and best[1] >= target_sum:
                for other in futs:
                    other.cancel()
                break
    df = pd.DataFrame(rows, columns=["seed", "sum"]).sort_values("seed").reset_index(drop=True)
    df["avg"] = df["sum"] / 11
    return {"results": df, "games_played": len(df), "best_sum": best[1], "best_avg": best[1] / 11,
            "best_seed": best[0], "reached": best[1] >= target_sum, "lineup": best[2], "actions": best[3],
            "secs": round(time.time() - t0, 1)}
