from onzeprime import data
from onzeprime.engine import FORMATIONS

VALID_POS = {"GR", "DD", "DC", "DE", "MDC", "MC", "MO", "ED", "EE", "PL"}


def test_three_big_clubs_present(game_data):
    clubs, by_club = game_data
    assert [c.id for c in clubs[:3]] == ["slb", "fcp", "scp"]
    for cid in data.GRANDES:
        assert len(by_club[cid]) > 50, cid


def test_player_fields_are_sane(game_data):
    _, by_club = game_data
    for plist in by_club.values():
        for p in plist:
            assert 40 <= p.ovr <= 99
            assert p.pos and set(p.pos) <= VALID_POS, p
            assert "/" in p.epoca


def test_player_count_matches_raw_bundle(game_data):
    # Every `{ id: ...}` line of PD.jogadores blocks must have been parsed.
    import re

    js = data.fetch_bundle()
    blocks = re.findall(r'PD\.jogadores\("[a-z0-9-]+",\s*\[(.*?)\n\]\);', js, re.S)
    raw = sum(1 for b in blocks for line in b.splitlines() if line.lstrip().startswith("{ id:"))
    assert raw == sum(len(v) for v in game_data[1].values())


def test_every_formation_has_11_slots():
    for f, slots in FORMATIONS.items():
        assert len(slots) == 11, f
        assert len({s.id for s in slots}) == 11, f
