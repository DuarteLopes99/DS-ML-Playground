"""Load Onze Prime's player database.

The whole game (rules + ratings) ships client-side in one JS bundle
(`js/onzeprime.js`). We download it once, cache it under `data/`, and parse the
`PD.clubes([...])` / `PD.jogadores("<club>", [...])` blocks into Python objects.

No scraping of live games is needed to map the player pool: every player the
game can ever offer is in this file.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_URL = "https://onzeprime.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUNDLE_CACHE = DATA_DIR / "onzeprime_bundle.js"
USER_AGENT = "Mozilla/5.0 (research script)"

# Marker that ends the DOM-free part of the bundle (core rules + data).
CORE_END_MARKER = "/* ---- js/analytics.js ---- */"

GRANDES = ("slb", "fcp", "scp")


@dataclass(frozen=True)
class Player:
    id: str                    # person id (same person across clubs shares it)
    nome: str
    pos: Tuple[str, ...]       # first is the main position
    epoca: str
    ovr: int
    nac: str
    clube: str
    curto: Optional[str] = None
    pos_antes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def uid(self) -> str:
        """Unique row id (a person can appear for several clubs)."""
        return f"{self.clube}:{self.id}"


@dataclass(frozen=True)
class Club:
    id: str
    nome: str
    sigla: str
    peso: int


# --------------------------------------------------------------------------- fetch
def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def bundle_url() -> str:
    """Find the versioned bundle URL from the home page (`js/onzeprime.js?v=...`)."""
    html = _get(BASE_URL + "/")
    m = re.search(r'src="(js/onzeprime\.js[^"]*)"', html)
    if not m:
        raise RuntimeError("Could not find js/onzeprime.js on the home page; the site layout changed.")
    return f"{BASE_URL}/{m.group(1)}"


def fetch_bundle(force: bool = False) -> str:
    """Return the JS bundle text, downloading it only if not cached (or force=True)."""
    if BUNDLE_CACHE.exists() and not force:
        return BUNDLE_CACHE.read_text(encoding="utf-8")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    js = _get(bundle_url())
    BUNDLE_CACHE.write_text(js, encoding="utf-8")
    return js


def core_js(js: Optional[str] = None) -> str:
    """The DOM-free part of the bundle (rules + data); runs in any JS engine."""
    js = js if js is not None else fetch_bundle()
    end = js.find(CORE_END_MARKER)
    if end == -1:
        raise RuntimeError("Bundle marker for analytics.js not found; update CORE_END_MARKER.")
    return js[:end]


# --------------------------------------------------------------------------- parse
_KEY = re.compile(r'([{,]\s*)([A-Za-z_]\w*)\s*:')


def _js_obj_to_dict(obj: str) -> dict:
    """Convert a one-line JS object literal with simple values into a dict."""
    return json.loads(_KEY.sub(r'\1"\2":', obj))


def parse_bundle(js: str) -> Tuple[List[Club], List[Player]]:
    clubs: List[Club] = []
    m = re.search(r"PD\.clubes\(\[(.*?)\]\);", js, re.S)
    if not m:
        raise RuntimeError("PD.clubes block not found")
    for line in re.findall(r"\{[^{}]*\}", m.group(1)):
        d = _js_obj_to_dict(line)
        clubs.append(Club(d["id"], d["nome"], d["sigla"], int(d.get("peso", 1))))

    players: List[Player] = []
    for block in re.finditer(r'PD\.jogadores\("([a-z0-9-]+)",\s*\[(.*?)\n\]\);', js, re.S):
        club_id, body = block.group(1), block.group(2)
        for raw in re.findall(r"^\s*(\{\s*id:.*\})\s*,?\s*(?://.*)?$", body, re.M):
            d = _js_obj_to_dict(raw)
            players.append(Player(
                id=d["id"], nome=d["nome"], pos=tuple(d["pos"]), epoca=str(d["epoca"]),
                ovr=int(d["ovr"]), nac=d.get("nac", ""), clube=club_id,
                curto=d.get("curto"), pos_antes=tuple(d.get("posAntes") or ()),
            ))
    return clubs, players


def load(force: bool = False) -> Tuple[List[Club], Dict[str, List[Player]]]:
    """Clubs (bundle order) and players grouped by club (file order, which the RNG depends on)."""
    clubs, players = parse_bundle(fetch_bundle(force=force))
    by_club: Dict[str, List[Player]] = {c.id: [] for c in clubs}
    for p in players:
        by_club.setdefault(p.clube, []).append(p)
    return clubs, by_club


def players_dataframe(force: bool = False, clubs_filter=None):
    """Flat pandas DataFrame, one row per player-season (optionally only some clubs)."""
    import pandas as pd

    clubs, by_club = load(force=force)
    names = {c.id: c.nome for c in clubs}
    rows = []
    for cid, plist in by_club.items():
        if clubs_filter and cid not in clubs_filter:
            continue
        for p in plist:
            rows.append({
                "uid": p.uid, "id": p.id, "nome": p.nome, "clube": cid, "clube_nome": names.get(cid, cid),
                "pos_main": p.pos[0], "pos": "/".join(p.pos), "n_pos": len(p.pos),
                "epoca": p.epoca, "ovr": p.ovr, "nac": p.nac,
            })
    return pd.DataFrame(rows)
