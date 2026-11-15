"""Faithful Python port of Onze Prime's game rules (`js/core.js`).

Same seed + same actions => same draws as the website (checked against the
real JS engine in tests/test_parity_js.py). This lets us play hundreds of
thousands of games offline without touching the site's servers.

Rules in short (mode "grandes" = Benfica, FC Porto, Sporting):
  * Each round a club is drawn (never the same club twice in a row).
  * 6 players from that club are offered, weighted toward high overall,
    trying to cover the still-empty roles.
  * You place one in a compatible free slot. 11 rounds. 2 "trocas" (re-draws).
  * Score = average overall of the XI. The same person can only be picked once.
"""
from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .data import Club, Player

# --------------------------------------------------------------------------- RNG (mulberry32)
_M32 = 0xFFFFFFFF


def _i32(x: int) -> int:
    x &= _M32
    return x - 0x100000000 if x & 0x80000000 else x


def _imul(a: int, b: int) -> int:
    return _i32((a & _M32) * (b & _M32))


def mulberry32(seed: int) -> Callable[[], float]:
    """Bit-exact port of PD.rngSemente (JS int32 semantics)."""
    state = [seed & _M32]

    def rng() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & _M32
        t = state[0]
        t = _imul(_i32(t ^ (t >> 15)), _i32(t | 1))
        t = _i32(t) ^ _i32(t + _imul(_i32(t ^ ((t & _M32) >> 7)), _i32(t | 61)))
        return (((t ^ ((t & _M32) >> 14)) & _M32)) / 4294967296.0

    return rng


# --------------------------------------------------------------------------- formations
@dataclass(frozen=True)
class Slot:
    id: str
    papel: str                 # role label shown on the pitch
    aceita: Tuple[str, ...]    # player positions accepted


def _v(i, papel, aceita):
    return Slot(i, papel, tuple(aceita))


MC3, MC2 = ("MC", "MO"), ("MC", "MDC")
_GR = _v("GR", "GR", ["GR"])
_DEF4 = [_v("DE", "DE", ["DE"]), _v("DC1", "DC", ["DC"]), _v("DC2", "DC", ["DC"]), _v("DD", "DD", ["DD"])]
_DEF3 = [_v("DC1", "DC", ["DC"]), _v("DC2", "DC", ["DC"]), _v("DC3", "DC", ["DC"])]
_DEF5 = [_v("DE", "DE", ["DE"]), _v("DC1", "DC", ["DC"]), _v("DC2", "DC", ["DC"]), _v("DC3", "DC", ["DC"]), _v("DD", "DD", ["DD"])]

FORMATIONS: Dict[str, List[Slot]] = {
    "4-4-2": [_v("PL1", "PL", ["PL"]), _v("PL2", "PL", ["PL"]), _v("EE", "EE", ["EE"]), _v("MC1", "MC", MC2),
              _v("MC2", "MC", MC2), _v("ED", "ED", ["ED"])] + _DEF4 + [_GR],
    "4-3-3": [_v("EE", "EE", ["EE"]), _v("PL", "PL", ["PL"]), _v("ED", "ED", ["ED"]), _v("MC1", "MC", MC3),
              _v("MC2", "MC", MC3), _v("MDC", "MDC", ["MDC"])] + _DEF4 + [_GR],
    "4-2-3-1": [_v("PL", "PL", ["PL"]), _v("EE", "EE", ["EE"]), _v("MO", "MO", ["MO"]), _v("ED", "ED", ["ED"]),
                _v("MC1", "MC", MC2), _v("MC2", "MC", MC2)] + _DEF4 + [_GR],
    "4-2-4": [_v("EE", "EE", ["EE"]), _v("PL1", "PL", ["PL"]), _v("PL2", "PL", ["PL"]), _v("ED", "ED", ["ED"]),
              _v("MC1", "MC", MC2), _v("MC2", "MC", MC2)] + _DEF4 + [_GR],
    "3-4-3": [_v("EE", "EE", ["EE"]), _v("PL", "PL", ["PL"]), _v("ED", "ED", ["ED"]), _v("AE", "AE", ["DE", "EE"]),
              _v("MC1", "MC", MC2), _v("MC2", "MC", MC2), _v("AD", "AD", ["DD", "ED"])] + _DEF3 + [_GR],
    "3-5-2": [_v("PL1", "PL", ["PL"]), _v("PL2", "PL", ["PL"]), _v("AE", "AE", ["DE", "EE"]), _v("MC1", "MC", MC3),
              _v("MC2", "MC", MC3), _v("AD", "AD", ["DD", "ED"]), _v("MDC", "MDC", ["MDC"])] + _DEF3 + [_GR],
    "5-3-2": [_v("PL1", "PL", ["PL"]), _v("PL2", "PL", ["PL"]), _v("MC1", "MC", MC3), _v("MC2", "MC", MC3),
              _v("MDC", "MDC", ["MDC"])] + _DEF5 + [_GR],
    "5-4-1": [_v("PL", "PL", ["PL"]), _v("EE", "EE", ["EE"]), _v("MC1", "MC", MC2), _v("MC2", "MC", MC2),
              _v("ED", "ED", ["ED"])] + _DEF5 + [_GR],
}

MODES = {
    "grandes": {"clubs": ["slb", "fcp", "scp"], "repete": True},
    "liga": {"clubs": None, "repete": False},
}

OPTIONS_PER_ROUND = 6
SWAPS = 2
TOTAL = 11
RULES_VERSION = 2


def fits(player: Player, slot: Slot, rules: int = RULES_VERSION) -> bool:
    """Jogo.cabe: can this player occupy this slot?"""
    old = rules == 1
    pos = player.pos + player.pos_antes if (old and player.pos_antes) else player.pos
    for p in slot.aceita:
        if p not in pos:
            continue
        # Rules v2: an MO who is also a winger/striker can't play in a central-midfield (MC) slot.
        if not old and p == "MO" and slot.papel == "MC":
            if any(q in ("EE", "ED", "PL") for q in player.pos):
                continue
        return True
    return False


def player_weight(p: Player) -> float:
    """Stars show up more often: max(ovr-58, 4) ** 1.6."""
    return max(p.ovr - 58, 4) ** 1.6


def _weighted_pick(items, weight, rng):
    total = 0.0
    for it in items:
        total += weight(it)
    r = rng() * total
    for it in items:
        r -= weight(it)
        if r <= 0:
            return it
    return items[-1]


def _shuffle(lst, rng):
    a = list(lst)
    for i in range(len(a) - 1, 0, -1):
        j = int(rng() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def _sort_key_name(nome: str) -> str:
    return unicodedata.normalize("NFD", nome).encode("ascii", "ignore").decode().casefold()


# --------------------------------------------------------------------------- game
_FIT_CACHE: Dict[tuple, Dict[str, int]] = {}

@dataclass
class Draw:
    club: str
    options: List[Player]


class Game:
    """One Onze Prime match. Mirrors the JS `estado` object and `Jogo.*` functions."""

    def __init__(self, clubs: Sequence[Club], by_club: Dict[str, List[Player]], mode: str = "grandes",
                 formation: str = "4-3-3", seed: Optional[int] = None, rules: int = RULES_VERSION,
                 auto_draw: bool = True):
        self.clubs = list(clubs)
        self.club_by_id = {c.id: c for c in clubs}
        self.by_club = by_club
        self.mode = mode if mode in MODES else "grandes"
        self.formation = formation if formation in FORMATIONS else "4-3-3"
        self.slots = FORMATIONS[self.formation]
        self.rules = rules
        # Static fit table: player uid -> bitmask of slot indexes he can play (fits() is pure).
        key = (self.formation, rules, id(by_club))
        if key not in _FIT_CACHE:
            _FIT_CACHE[key] = {p.uid: sum(1 << i for i, sl in enumerate(self.slots) if fits(p, sl, rules))
                               for plist in by_club.values() for p in plist}
        self._fit = _FIT_CACHE[key]
        self.seed = seed if seed is not None else random.getrandbits(32)
        self.rng = mulberry32(self.seed)
        self.xi: Dict[str, Player] = {}
        self.persons: set = set()
        self.clubs_used: set = set()
        self.last_club: Optional[str] = None
        self.swaps = SWAPS
        self.draw: Optional[Draw] = None
        self.finished = False
        self.actions: List[list] = []
        if auto_draw:
            self._draw_club()

    # ---- queries
    def picked(self) -> int:
        return sum(1 for s in self.slots if s.id in self.xi)

    def round(self) -> int:
        return min(self.picked() + 1, TOTAL)

    def _free_mask(self) -> int:
        return sum(1 << i for i, s in enumerate(self.slots) if s.id not in self.xi)

    def free_slots(self) -> List[Slot]:
        return [s for s in self.slots if s.id not in self.xi]

    def slots_for(self, p: Player) -> List[Slot]:
        m = self._fit.get(p.uid)
        if m is None:  # player not in the table (e.g. built by hand): fall back to the rule
            return [s for s in self.free_slots() if fits(p, s, self.rules)]
        m &= self._free_mask()
        return [s for i, s in enumerate(self.slots) if m >> i & 1]

    def roles_for(self, p: Player) -> List[str]:
        seen, roles = set(), []
        for s in self.slots_for(p):
            if s.papel not in seen:
                seen.add(s.papel)
                roles.append(s.papel)
        return roles

    def eligible(self, club_id: str) -> List[Player]:
        free = self._free_mask()
        return [p for p in self.by_club.get(club_id, []) if p.id not in self.persons and self._fit[p.uid] & free]

    def average(self) -> float:
        vals = [self.xi[s.id].ovr for s in self.slots if s.id in self.xi]
        return sum(vals) / len(vals) if vals else 0.0

    def mode_clubs(self) -> List[Club]:
        ids = MODES[self.mode]["clubs"]
        return [c for c in self.clubs if self.by_club.get(c.id) and (ids is None or c.id in ids)]

    # ---- draws
    def _sample(self, club_id: str) -> List[Player]:
        rng = self.rng
        pool = self.eligible(club_id)
        chosen: List[Player] = []
        roles: Dict[str, bool] = {}
        for s in self.free_slots():
            roles[s.papel] = True
        roles_of = {p.uid: set(self.roles_for(p)) for p in pool}  # cache: same result as recomputing
        for role in _shuffle(list(roles.keys()), rng):
            if len(chosen) >= OPTIONS_PER_ROUND:
                break
            cands = [p for p in pool if role in roles_of[p.uid]]
            if cands:
                pick = _weighted_pick(cands, player_weight, rng)
                chosen.append(pick)
                pool = [p for p in pool if p is not pick]
        while len(chosen) < OPTIONS_PER_ROUND and pool:
            pick = _weighted_pick(pool, player_weight, rng)
            chosen.append(pick)
            pool = [p for p in pool if p is not pick]
        return sorted(chosen, key=lambda p: (-p.ovr, _sort_key_name(p.nome)))

    def _draw_club(self) -> Optional[Draw]:
        repete = MODES[self.mode]["repete"]
        in_mode = [c for c in self.mode_clubs() if self.eligible(c.id)]
        cands = [c for c in in_mode if (c.id != self.last_club if repete else c.id not in self.clubs_used)]
        if not cands:
            cands = in_mode
        if not cands:
            self.draw = None
            return None
        if repete:
            club = cands[int(self.rng() * len(cands))]
        else:
            club = _weighted_pick(cands, lambda c: c.peso or 1, self.rng)
        self.clubs_used.add(club.id)
        self.last_club = club.id
        self.draw = Draw(club.id, self._sample(club.id))
        return self.draw

    # ---- actions
    def can_swap(self) -> bool:
        return not self.finished and self.swaps > 0

    def swap(self) -> bool:
        if not self.can_swap():
            return False
        self.swaps -= 1
        self.actions.append(["t"])
        self._draw_club()
        return True

    def choose(self, player: Player, slot_id: Optional[str] = None) -> Optional[Slot]:
        if self.finished or not self.draw or player not in self.draw.options:
            return None
        slots = self.slots_for(player)
        slot = next((s for s in slots if s.id == slot_id), None) if slot_id else (slots[0] if slots else None)
        if not slot:
            return None
        self.xi[slot.id] = player
        self.persons.add(player.id)
        self.actions.append(["e", player.id, slot.id])
        if self.picked() >= TOTAL:
            self.finished = True
            self.draw = None
        else:
            self._draw_club()
        return slot

    def lineup(self) -> List[Tuple[str, str, str, int]]:
        return [(s.id, self.xi[s.id].nome, self.xi[s.id].clube, self.xi[s.id].ovr) for s in self.slots if s.id in self.xi]


def replay(clubs, by_club, mode, formation, seed, actions, rules=RULES_VERSION) -> Optional[Game]:
    """Jogo.repetir: rebuild a game from seed + actions (what the server does for the ranking)."""
    g = Game(clubs, by_club, mode, formation, seed, rules)
    for a in actions:
        if a[0] == "t":
            if not g.swap():
                return None
        elif a[0] == "e":
            p = next((o for o in (g.draw.options if g.draw else []) if o.id == a[1]), None)
            if not p or not g.choose(p, a[2]):
                return None
        else:
            return None
    return g if g.finished else None
