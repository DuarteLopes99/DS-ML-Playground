import pytest

from onzeprime.engine import FORMATIONS, Game, fits, mulberry32, replay
from onzeprime.simulate import play
from onzeprime.strategies import CeilingPolicy, EliteTargetPolicy, GreedyPolicy


def test_rng_deterministic_and_in_range():
    a, b = mulberry32(42), mulberry32(42)
    xs = [a() for _ in range(1000)]
    assert xs == [b() for _ in range(1000)]
    assert all(0 <= x < 1 for x in xs)
    assert xs[:3] != [mulberry32(43)() for _ in range(3)]


def test_rng_known_values():
    # Reference values of the JS PD.rngSemente(1) (also asserted live in test_parity_js.py).
    r = mulberry32(1)
    assert [round(r(), 10) for _ in range(3)] == [0.6270739406, 0.0027357212, 0.52744704]


def test_mo_winger_cannot_play_mc_slot(game_data):
    _, by_club = game_data
    james = next(p for p in by_club["fcp"] if p.nome == "James Rodríguez")
    mc_433 = next(s for s in FORMATIONS["4-3-3"] if s.id == "MC1")
    assert not fits(james, mc_433)
    mo_4231 = next(s for s in FORMATIONS["4-2-3-1"] if s.id == "MO")
    assert fits(james, mo_4231)


@pytest.mark.parametrize("formation", list(FORMATIONS))
def test_game_invariants(game_data, formation):
    clubs, by_club = game_data
    for seed in range(20):
        g = play(clubs, by_club, GreedyPolicy(), formation, seed)
        assert g.finished and len(g.xi) == 11
        assert len({p.id for p in g.xi.values()}) == 11          # one person once
        slots = {s.id: s for s in FORMATIONS[formation]}
        assert all(fits(p, slots[sid]) for sid, p in g.xi.items())


def test_same_club_never_twice_in_a_row(game_data):
    clubs, by_club = game_data
    g = Game(clubs, by_club, "grandes", "4-3-3", seed=7)
    seen = [g.draw.club]
    while not g.finished:
        g.choose(g.draw.options[0])
        if g.draw:
            seen.append(g.draw.club)
    assert all(a != b for a, b in zip(seen, seen[1:]))


def test_six_options_and_swaps(game_data):
    clubs, by_club = game_data
    g = Game(clubs, by_club, "grandes", "3-5-2", seed=3)
    assert len(g.draw.options) == 6
    assert g.swap() and g.swap() and not g.swap()
    assert g.swaps == 0


def test_replay_reproduces_game(game_data, pool):
    clubs, by_club = game_data
    pol = CeilingPolicy(pool, "3-5-2", swap_margin=2)
    g = play(clubs, by_club, pol, "3-5-2", 123)
    r = replay(clubs, by_club, "grandes", "3-5-2", 123, g.actions)
    assert r is not None and r.lineup() == g.lineup()


def test_policies_respect_ceiling(game_data, pool):
    from onzeprime.optimizer import best_xi
    clubs, by_club = game_data
    top = best_xi(pool, "4-4-2")["avg"]
    for pol in (GreedyPolicy(), EliteTargetPolicy(pool, "4-4-2"), CeilingPolicy(pool, "4-4-2")):
        for seed in range(10):
            assert play(clubs, by_club, pol, "4-4-2", seed).average() <= top + 1e-9
