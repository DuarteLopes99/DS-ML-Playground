import math

import pytest

from onzeprime import training as tr
from onzeprime.engine import Game
from onzeprime.optimizer import best_xi, optimal_candidates
from onzeprime.strategies import CeilingPolicy, RecordHunterPolicy


def test_optimal_candidates_contain_best_xi(pool):
    for f in ("3-5-2", "4-4-2"):
        oc = optimal_candidates(pool, f)
        for row in best_xi(pool, f)["lineup"]:
            assert row["uid"].split(":", 1)[1] in oc[row["slot"]]


def test_3_5_2_has_seven_must_have_stars(pool):
    oc = optimal_candidates(pool, "3-5-2")
    singles = {s for s, v in oc.items() if len(v) == 1}
    assert singles == {"AE", "AD", "MDC"}
    assert len(set(oc["PL1"])) == 2 and len(set(oc["MC1"])) == 2   # 2 strikers + 2 MCs, both needed


def test_fast_completion_equals_full_assignment(game_data, pool):
    # The top-n column pruning must give the same ceiling as the full matrix.
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    clubs, by_club = game_data
    pol = CeilingPolicy(pool, "3-5-2")
    for seed in range(15):
        g = Game(clubs, by_club, "grandes", "3-5-2", seed)
        while not g.finished:
            free = [s.id for s in g.free_slots()]
            rows = [pol.slot_idx[s] for s in free]
            cols = [j for pid, j in pol.col.items() if pid not in g.persons]
            sub = pol.V[np.ix_(rows, cols)]
            r, c = linear_sum_assignment(-sub)
            assert pol.completion(free, g.persons) == pytest.approx(sub[r, c].sum())
            g.choose(g.draw.options[seed % len(g.draw.options)])


def test_hunter_never_kills_an_alive_game_when_it_has_an_alive_option(game_data, pool):
    clubs, by_club = game_data
    pol = RecordHunterPolicy(pool, "3-5-2")
    for seed in range(60):
        g = Game(clubs, by_club, "grandes", "3-5-2", seed)
        while not g.finished:
            was_alive = not pol.is_dead(g)
            d = pol(g)
            if d.kind == "swap":
                g.swap()
                continue
            # was there an alive option?
            picked = sum(p.ovr for p in g.xi.values())
            free = [s.id for s in g.free_slots()]
            any_alive = any(
                picked + p.ovr + pol.completion([f for f in free if f != s.id], g.persons | {p.id}) >= 995
                for p in g.draw.options for s in g.slots_for(p))
            g.choose(d.player, d.slot_id)
            if was_alive and any_alive:
                assert not pol.is_dead(g), seed


def test_wilson_and_geometric():
    p, lo, hi = tr.wilson(5, 20000)
    assert lo < p < hi and p == 5 / 20000
    assert tr.games_needed(0.5, 0.5) == pytest.approx(1.0)
    assert tr.games_needed(1 / 4000, 0.632) == pytest.approx(4000, rel=0.01)
    assert math.isinf(tr.games_needed(0, 0.9))


def test_play_tracked_early_abort(game_data, pool):
    clubs, by_club = game_data
    pol = RecordHunterPolicy(pool, "3-5-2")
    r = tr.play_tracked(clubs, by_club, pol, "3-5-2", 1, early_abort=True)
    assert r["picks"] == r["alive_picks"] + (0 if r["record"] else 1) or r["record"]
