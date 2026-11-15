"""Training loop for the record hunter.

The record (995) is a rare event (~1 in 4,000 games), so we can't optimise it
directly with a few thousand games. Instead:

1. Survival view: after every pick, is the 995 ceiling still reachable?
   S(k) = P(still alive after k picks). P(record) = S(11).
2. Fast simulation: a dead game can't become a record, so training games
   stop at the first death (early abort). That's several times faster.
3. Search: evaluate every candidate config on the SAME training seeds
   (common random numbers), so differences come from the policy and not from luck.
4. Validation: re-run the best configs on fresh held-out seeds, with a
   Wilson confidence interval for P(record).
5. Planning: geometric distribution -> games / time needed for 50% / 90% odds.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .engine import Game, TOTAL

TARGET = 995


# --------------------------------------------------------------------------- one tracked game
def play_tracked(clubs, by_club, policy, formation: str, seed: int, early_abort: bool = True,
                 target_sum: int = TARGET) -> dict:
    """Play one game. Returns alive_picks (picks made while 995 was still reachable), record flag, steps."""
    g = Game(clubs, by_club, "grandes", formation, seed)
    alive_picks, steps, died_at = 0, 0, None
    while not g.finished and g.draw is not None:
        d = policy(g)
        steps += 1
        if d.kind == "swap":
            g.swap()
            continue
        g.choose(d.player, d.slot_id)
        if died_at is None:
            if policy.ceiling_now(g) >= target_sum - 1e-9:
                alive_picks += 1
            else:
                died_at = g.picked()
                if early_abort:
                    break
    total = sum(p.ovr for p in g.xi.values())
    return {"seed": seed, "alive_picks": alive_picks, "record": g.finished and total >= target_sum,
            "picks": g.picked(), "swaps_used": 2 - g.swaps, "steps": steps}


def _chunk(args):
    from . import data, optimizer
    from .strategies import POLICIES

    formation, policy_name, params, seeds, early_abort = args
    clubs, by_club = data.load()
    pool = optimizer.pool_for_mode(by_club, "grandes")
    pol = POLICIES[policy_name](pool, formation, **params)
    t = time.perf_counter()
    rows = [play_tracked(clubs, by_club, pol, formation, s, early_abort) for s in seeds]
    return rows, time.perf_counter() - t


def evaluate(formation: str, policy_name: str, params: dict, seeds: Iterable[int], workers: int = 7,
             early_abort: bool = True, chunk: int = 250) -> pd.DataFrame:
    seeds = list(seeds)
    jobs = [(formation, policy_name, params, seeds[i:i + chunk], early_abort) for i in range(0, len(seeds), chunk)]
    rows, cpu = [], 0.0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, secs in ex.map(_chunk, jobs):
            rows.extend(r)
            cpu += secs
    df = pd.DataFrame(rows)
    df.attrs["cpu_secs"] = cpu
    return df


# --------------------------------------------------------------------------- metrics
def survival(df: pd.DataFrame) -> pd.Series:
    """S(k) for k = 0..11: share of games still able to reach 995 after k picks."""
    return pd.Series({k: (df.alive_picks >= k).mean() for k in range(TOTAL + 1)}, name="S(k)")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def summarize(df: pd.DataFrame) -> dict:
    s = survival(df)
    p, lo, hi = wilson(int(df.record.sum()), len(df))
    return {"games": len(df), "records": int(df.record.sum()), "P(record)": p, "ci_low": lo, "ci_high": hi,
            "S(4)": s[4], "S(7)": s[7], "S(9)": s[9], "mean_alive": df.alive_picks.mean(),
            "ms_per_game_cpu": 1000 * df.attrs.get("cpu_secs", float("nan")) / len(df)}


def games_needed(p: float, conf: float) -> float:
    """Attempts needed so that P(at least one record) >= conf (geometric distribution)."""
    if p <= 0:
        return float("inf")
    return math.log(1 - conf) / math.log(1 - p)


def time_plan(p: float, secs_per_attempt: float) -> pd.DataFrame:
    rows = []
    for conf in (0.5, 0.632, 0.9, 0.99):
        n = games_needed(p, conf)
        rows.append({"confidence": conf, "attempts": round(n), "hours": n * secs_per_attempt / 3600})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- search
def random_configs(space: Dict[str, list], n: int, seed: int = 0) -> List[dict]:
    rnd = random.Random(seed)
    full = [dict(zip(space, c)) for c in itertools.product(*space.values())]
    rnd.shuffle(full)
    return full[:n]


def search(formation: str, policy_name: str, configs: List[dict], train_seeds: Iterable[int],
           workers: int = 7, verbose: bool = True) -> pd.DataFrame:
    train_seeds = list(train_seeds)
    out = []
    for i, cfg in enumerate(configs, 1):
        t = time.time()
        df = evaluate(formation, policy_name, cfg, train_seeds, workers)
        row = {**cfg, **summarize(df), "wall_secs": round(time.time() - t, 1)}
        out.append(row)
        if verbose:
            print(f"[{i}/{len(configs)}] {cfg} -> S(9)={row['S(9)']:.4f} records={row['records']} ({row['wall_secs']}s)")
    return pd.DataFrame(out).sort_values(["S(9)", "mean_alive"], ascending=False).reset_index(drop=True)


def hazard(df: pd.DataFrame) -> pd.Series:
    """h(k) = P(the k-th pick kills the 995 chance | alive before it)."""
    s = survival(df)
    return pd.Series({k: 1 - s[k] / s[k - 1] if s[k - 1] > 0 else float("nan") for k in range(1, TOTAL + 1)}, name="h(k)")
