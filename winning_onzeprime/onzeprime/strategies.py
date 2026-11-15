"""Decision policies: given the current game state, pick a player+slot or use a swap.

A policy is any callable `policy(game) -> Decision`. The same policies drive both
the offline simulator and the live Playwright bot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data import Player
from .engine import FORMATIONS, TOTAL, Game, fits


@dataclass
class Decision:
    kind: str                       # "pick" | "swap"
    player: Optional[Player] = None
    slot_id: Optional[str] = None


def _first_slot_of_role(game: Game, player: Player, papel: str) -> str:
    # The website places a player in the first free slot of the chosen role; mirror that
    # so the live shadow state never drifts from the UI (DC1 vs DC2 is equivalent anyway).
    return next(s.id for s in game.slots_for(player) if s.papel == papel)


class GreedyPolicy:
    """Baseline: take the highest-rated option, never swap."""

    name = "greedy"

    def __call__(self, game: Game) -> Decision:
        best = max(game.draw.options, key=lambda p: p.ovr)
        return Decision("pick", best, game.slots_for(best)[0].id)


class EliteTargetPolicy:
    """Aim for the elite players of each role.

    1. Map the pool: for every role in the formation, the ratings of the top-k
       candidates (the "elite bar" = mean of the top `k` ratings for that role).
    2. Each option/role pair gets   gain = ovr - bar[role]   (how close to elite).
    3. If even the best gain is below `swap_below` and swaps remain, re-draw.
       Late in the game we lower our standards (`late_round`).
    4. Multi-role players go where they beat the bar by most, which keeps
       scarce roles open for the stars that only fit there.
    """

    name = "elite"

    def __init__(self, pool: Sequence[Player], formation: str, k: int = 3,
                 swap_below: float = -3.0, late_round: int = 9, target_bonus: float = 1.0,
                 target_uids: Optional[set] = None):
        self.formation = formation
        self.k, self.swap_below, self.late_round, self.target_bonus = k, swap_below, late_round, target_bonus
        self.target_uids = target_uids or set()
        self.bar: Dict[str, float] = {}
        for s in FORMATIONS[formation]:
            if s.papel in self.bar:
                continue
            ratings = sorted((p.ovr for p in pool if fits(p, s)), reverse=True)
            # Number of slots with this role (e.g. 2 x DC) -> look a bit deeper into the list.
            n_slots = sum(1 for t in FORMATIONS[formation] if t.papel == s.papel)
            top = ratings[: max(1, k + n_slots - 1)]
            self.bar[s.papel] = sum(top) / len(top) if top else 0.0

    def score(self, game: Game, p: Player, papel: str) -> float:
        bonus = self.target_bonus if p.uid in self.target_uids else 0.0
        # Prefer the role whose bar is highest relative to the player (=scarcest elite).
        return p.ovr - self.bar[papel] + bonus

    def __call__(self, game: Game) -> Decision:
        best, best_s = None, -1e9
        for p in game.draw.options:
            for papel in game.roles_for(p):
                s = self.score(game, p, papel)
                if s > best_s:
                    best, best_s = (p, papel), s
        late = game.round() >= self.late_round
        if game.can_swap() and not late and best_s < self.swap_below:
            return Decision("swap")
        p, papel = best
        return Decision("pick", p, _first_slot_of_role(game, p, papel))


class CeilingPolicy:
    """Keep the best still-achievable XI as high as possible (record hunter).

    ceiling(state) = sum of picked ratings + optimal completion of the free slots
    from the players not yet used (exact, via linear assignment).

    Each round, every (option, role) is scored by the ceiling it leaves. We pick
    the highest (ties -> higher overall). If every option lowers the ceiling by
    more than `swap_margin` points (sum, not average) and a swap is left, we swap.
    With `target_sum` set (995 = 90.45 avg in Três Grandes), a swap is also used
    whenever the best option would drop the ceiling below the target.
    """

    name = "ceiling"

    def __init__(self, pool: Sequence[Player], formation: str, swap_margin: float = 0.0,
                 target_sum: Optional[int] = None, late_round: int = 12):
        import numpy as np

        self.formation = formation
        self.slots = FORMATIONS[formation]
        self.slot_idx = {s.id: i for i, s in enumerate(self.slots)}
        self.persons = sorted({p.id for p in pool})
        self.col = {pid: j for j, pid in enumerate(self.persons)}
        V = np.zeros((len(self.slots), len(self.persons)))
        for p in pool:
            j = self.col[p.id]
            for i, s in enumerate(self.slots):
                if fits(p, s):
                    V[i, j] = max(V[i, j], p.ovr)
        self.V = V
        # Per slot: people sorted by rating (only those who can play there).
        self.order = [[int(j) for j in np.argsort(-V[i], kind="stable") if V[i, j] > 0] for i in range(len(self.slots))]
        self._cache = {}
        self.swap_margin, self.target_sum, self.late_round = swap_margin, target_sum, late_round

    def completion(self, free_slot_ids, used_person_ids) -> float:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        if not free_slot_ids:
            return 0.0
        rows = sorted(self.slot_idx[s] for s in free_slot_ids)
        used = frozenset(self.col[pid] for pid in used_person_ids if pid in self.col)
        key = (tuple(rows), used)
        if key in self._cache:
            return self._cache[key]
        # An optimal assignment only ever uses, for each row, one of that row's top-n unused people
        # (n = number of free slots), so the union of those lists is enough: exact and much smaller.
        n, cols = len(rows), set()
        for r in rows:
            k = 0
            for j in self.order[r]:
                if j not in used:
                    cols.add(j)
                    k += 1
                    if k == n:
                        break
        cols = sorted(cols)
        sub = self.V[np.ix_(rows, cols)]
        r, c = linear_sum_assignment(-sub)
        val = float(sub[r, c].sum())
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[key] = val
        return val

    def ceiling_now(self, game: Game) -> float:
        picked = sum(p.ovr for p in game.xi.values())
        return picked + self.completion([s.id for s in game.free_slots()], game.persons)

    def __call__(self, game: Game) -> Decision:
        picked = sum(p.ovr for p in game.xi.values())
        free = [s.id for s in game.free_slots()]
        best, best_key = None, None
        for p in game.draw.options:
            for papel in game.roles_for(p):
                sid = _first_slot_of_role(game, p, papel)
                rest = [f for f in free if f != sid]
                ub = picked + p.ovr + self.completion(rest, game.persons | {p.id})
                key = (ub, p.ovr)
                if best_key is None or key > best_key:
                    best, best_key = (p, sid), key
        if game.can_swap() and game.round() < self.late_round:
            now = self.ceiling_now(game)
            below_target = self.target_sum is not None and best_key[0] < self.target_sum <= now
            if below_target or now - best_key[0] > self.swap_margin:
                return Decision("swap")
        return Decision("pick", best[0], best[1])


class RecordHunterPolicy(CeilingPolicy):
    """Only cares about reaching `target_sum` (995). Every decision is explainable (`explain`).

    Decision process, each draw (6 options):
      1. Candidates: every (option, role it can still play). Multi-role players give several rows.
      2. Ceiling after the pick = points so far + rating + best completion of the other free slots
         from unused people (exact assignment). Alive <=> ceiling >= target.
      3. Alive rows get a score; the highest wins:
             w_scarce * 1/(alive people left for that slot)   lock irreplaceable stars
           + w_close  * [fills the last free slot of its role] fewer open roles -> more star draws
           + w_ovr    * ovr / 100                             tie-break
           + slot_bias[papel] + player_bias[nome] + club_bias[club]   <- YOUR BIAS
      4. Swap rules:
           * rescue: no alive row, game still alive, swaps left, round >= rescue_from_round
           * proactive (optional): alive rows exist but best score < proactive_swap_below and
             round <= proactive_until_round  (bet that a redraw offers something better)
      5. No alive row and no swap -> best ceiling row (dead game: live bot restarts here).
    """

    name = "hunter"

    def __init__(self, pool: Sequence[Player], formation: str, target_sum: int = 995, w_scarce: float = 1.0,
                 w_close: float = 0.0, w_ovr: float = 0.1, rescue_from_round: int = 1,
                 slot_bias: Optional[Dict[str, float]] = None, player_bias: Optional[Dict[str, float]] = None,
                 club_bias: Optional[Dict[str, float]] = None, proactive_swap_below: Optional[float] = None,
                 proactive_until_round: int = 3):
        super().__init__(pool, formation, target_sum=target_sum)
        from .optimizer import optimal_candidates

        self.alive_at = {sid: set(v) for sid, v in optimal_candidates(pool, formation, target_sum).items()}
        self.w_scarce, self.w_close, self.w_ovr = w_scarce, w_close, w_ovr
        self.rescue_from_round = rescue_from_round
        self.slot_bias, self.player_bias, self.club_bias = slot_bias or {}, player_bias or {}, club_bias or {}
        self.proactive_swap_below, self.proactive_until_round = proactive_swap_below, proactive_until_round

    def is_dead(self, game: Game) -> bool:
        return self.ceiling_now(game) < self.target_sum - 1e-9

    def explain(self, game: Game):
        """All candidate rows with every score component, plus the decision and the reason."""
        picked = sum(p.ovr for p in game.xi.values())
        free = [s.id for s in game.free_slots()]
        rows = []
        for p in game.draw.options:
            roles = game.roles_for(p)
            if not roles:
                rows.append({"player": p.nome, "club": p.clube, "ovr": p.ovr, "role": "-", "slot": None,
                             "ceiling_after": None, "alive": False, "score": None})
            for papel in roles:
                sid = _first_slot_of_role(game, p, papel)
                ub = picked + p.ovr + self.completion([f for f in free if f != sid], game.persons | {p.id})
                row = {"player": p.nome, "club": p.clube, "ovr": p.ovr, "role": papel, "slot": sid,
                       "ceiling_after": ub, "alive": ub >= self.target_sum - 1e-9, "_p": p}
                if row["alive"]:
                    left = len(self.alive_at[sid] - game.persons)
                    closes = sum(1 for s in game.free_slots() if s.papel == papel) == 1
                    row.update({"alive_left": left, "c_scarce": self.w_scarce / max(left, 1),
                                "c_close": self.w_close * closes, "c_ovr": self.w_ovr * p.ovr / 100,
                                "c_bias": self.slot_bias.get(papel, 0.0) + self.player_bias.get(p.nome, 0.0)
                                + self.club_bias.get(p.clube, 0.0)})
                    row["score"] = row["c_scarce"] + row["c_close"] + row["c_ovr"] + row["c_bias"]
                else:
                    row["score"] = None
                rows.append(row)
        now = picked + self.completion(free, game.persons)
        game_alive = now >= self.target_sum - 1e-9
        alive = [r for r in rows if r["alive"]]
        if alive:
            best = max(alive, key=lambda r: (r["score"], r["ovr"]))
            if (self.proactive_swap_below is not None and best["score"] < self.proactive_swap_below
                    and game.can_swap() and game.round() <= self.proactive_until_round):
                return rows, Decision("swap"), f"proactive swap: best score {best['score']:.3f} < {self.proactive_swap_below}"
            return rows, Decision("pick", best["_p"], best["slot"]), "best alive score"
        if game_alive and game.can_swap() and game.round() >= self.rescue_from_round:
            return rows, Decision("swap"), "rescue swap: no option keeps 995 reachable"
        cands = [r for r in rows if r["slot"]]
        fb = max(cands, key=lambda r: (r["ceiling_after"], r["ovr"]))
        why = "dead game: best remaining ceiling" if not game_alive else "no swaps left: 995 lost on this pick"
        return rows, Decision("pick", fb["_p"], fb["slot"]), why

    def __call__(self, game: Game) -> Decision:
        return self.explain(game)[1]


POLICIES = {"greedy": GreedyPolicy, "elite": EliteTargetPolicy, "ceiling": CeilingPolicy, "hunter": RecordHunterPolicy}
