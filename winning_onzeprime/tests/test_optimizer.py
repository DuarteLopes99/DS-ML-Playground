from onzeprime.engine import FORMATIONS, fits
from onzeprime.optimizer import best_xi, best_xi_all, role_table


def test_best_xi_valid(pool):
    for f in FORMATIONS:
        r = best_xi(pool, f)
        assert len(r["lineup"]) == 11
        ids = [row["uid"].split(":", 1)[1] for row in r["lineup"]]
        assert len(set(ids)) == 11


def test_record_equals_theoretical_max(pool):
    # The ranking's "90,5" is 995/11 = 90.4545 rounded by toFixed(1).
    best = best_xi_all(pool)[0]
    assert round(best["avg"] * 11) == 995
    assert f"{best['avg']:.1f}" == "90.5"


def test_ceiling_beats_bruteforce_on_small_case(pool):
    # Sanity: best XI is at least the greedy one (fill slots by best available).
    for f in ("4-3-3", "3-5-2"):
        used, total = set(), 0
        for s in FORMATIONS[f]:
            p = max((p for p in pool if fits(p, s) and p.id not in used), key=lambda p: p.ovr)
            used.add(p.id)
            total += p.ovr
        assert best_xi(pool, f)["avg"] * 11 >= total - 1e-9


def test_role_table_sorted(pool):
    t = role_table(pool, "3-5-2", top=5)
    assert set(t) == {s.papel for s in FORMATIONS["3-5-2"]}
    for players in t.values():
        assert [p.ovr for p in players] == sorted((p.ovr for p in players), reverse=True)
