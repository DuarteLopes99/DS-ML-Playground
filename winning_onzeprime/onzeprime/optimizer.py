"""Theoretical ceiling: the best possible XI per formation, ignoring draw luck.

Solved exactly as an assignment problem (slots x persons). A person who
appears for several clubs counts once, with their best eligible rating for
that slot.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import Player
from .engine import FORMATIONS, MODES, Slot, fits


def pool_for_mode(by_club: Dict[str, List[Player]], mode: str = "grandes") -> List[Player]:
    ids = MODES[mode]["clubs"]
    return [p for cid, ps in by_club.items() if ids is None or cid in ids for p in ps]


def best_xi(pool: Sequence[Player], formation: str) -> dict:
    slots = FORMATIONS[formation]
    persons = sorted({p.id for p in pool})
    col = {pid: j for j, pid in enumerate(persons)}
    value = np.zeros((len(slots), len(persons)))
    who = [[None] * len(persons) for _ in slots]
    for p in pool:
        j = col[p.id]
        for i, s in enumerate(slots):
            if fits(p, s) and p.ovr > value[i, j]:
                value[i, j], who[i][j] = p.ovr, p
    rows, cols = linear_sum_assignment(-value)
    lineup = [{"slot": slots[i].id, "papel": slots[i].papel, "nome": who[i][j].nome, "clube": who[i][j].clube,
               "epoca": who[i][j].epoca, "ovr": who[i][j].ovr, "uid": who[i][j].uid}
              for i, j in zip(rows, cols)]
    order = {s.id: k for k, s in enumerate(slots)}
    lineup.sort(key=lambda r: order[r["slot"]])
    return {"formation": formation, "avg": float(value[rows, cols].mean()), "lineup": lineup}


def best_xi_all(pool: Sequence[Player]) -> List[dict]:
    return sorted((best_xi(pool, f) for f in FORMATIONS), key=lambda r: -r["avg"])


def role_table(pool: Sequence[Player], formation: str, top: int = 10) -> Dict[str, List[Player]]:
    """For each role (papel) of a formation, the top-N eligible players by overall."""
    out: Dict[str, List[Player]] = {}
    for s in FORMATIONS[formation]:
        if s.papel in out:
            continue
        el = sorted((p for p in pool if fits(p, s)), key=lambda p: -p.ovr)
        out[s.papel] = el[:top]
    return out


def slot_of_role(formation: str, papel: str) -> Slot:
    return next(s for s in FORMATIONS[formation] if s.papel == papel)


def value_matrix(pool: Sequence[Player], formation: str):
    """V[slot, person] = best rating of that person in that slot (0 = can't play there)."""
    slots = FORMATIONS[formation]
    persons = sorted({p.id for p in pool})
    col = {pid: j for j, pid in enumerate(persons)}
    V = np.zeros((len(slots), len(persons)))
    for p in pool:
        for i, s in enumerate(slots):
            if fits(p, s):
                V[i, col[p.id]] = max(V[i, col[p.id]], p.ovr)
    return V, persons


def optimal_candidates(pool: Sequence[Player], formation: str, target_sum: int = None) -> Dict[str, List[str]]:
    """For each slot, the people who appear in at least one XI reaching target_sum (default: the max).

    A person j is 'alive' at slot s if V[s,j] + best completion of the other 10 slots without j == target.
    """
    V, persons = value_matrix(pool, formation)
    r, c = linear_sum_assignment(-V)
    best = V[r, c].sum()
    target = best if target_sum is None else target_sum
    slots = FORMATIONS[formation]
    out = {}
    for i, s in enumerate(slots):
        alive = []
        other_rows = [k for k in range(len(slots)) if k != i]
        for j in np.nonzero(V[i])[0]:
            cols = [k for k in range(len(persons)) if k != j]
            sub = V[np.ix_(other_rows, cols)]
            rr, cc = linear_sum_assignment(-sub)
            if V[i, j] + sub[rr, cc].sum() >= target - 1e-9:
                alive.append(persons[j])
        out[s.id] = alive
    return out
