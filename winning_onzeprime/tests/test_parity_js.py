"""Python engine vs the real website engine (core.js run in a blank headless page, no network)."""
import json

import pytest

from onzeprime import data
from onzeprime.engine import Game, mulberry32

playwright = pytest.importorskip("playwright.sync_api")

JS_PLAY = """
([modo, formacao, semente, nSwapsAt]) => {
  const e = PD.Jogo.novo(modo, PD.rngSemente(semente), formacao, 2);
  const draws = [];
  let r = 0;
  while (!e.terminado && e.sorteio) {
    draws.push([e.sorteio.clube.id, e.sorteio.opcoes.map(o => o.id)]);
    if (nSwapsAt.includes(r) && PD.Jogo.podeTrocar(e)) { PD.Jogo.trocar(e); r++; continue; }
    // deterministic policy: highest-ovr option (options come sorted), last compatible slot
    const j = e.sorteio.opcoes[0];
    const vagas = PD.Jogo.vagasPara(e, j);
    PD.Jogo.escolher(e, j, vagas[vagas.length - 1].id);
    r++;
  }
  return { draws, media: PD.Jogo.media(e) };
}
"""


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page()
        pg.set_content("<html><body></body></html>")
        pg.add_script_tag(content=data.core_js())
        yield pg
        b.close()


def py_play(game_data, modo, formacao, seed, swaps_at):
    clubs, by_club = game_data
    g = Game(clubs, by_club, modo, formacao, seed)
    draws, r = [], 0
    while not g.finished and g.draw:
        draws.append([g.draw.club, [p.id for p in g.draw.options]])
        if r in swaps_at and g.can_swap():
            g.swap(); r += 1; continue
        j = g.draw.options[0]
        g.choose(j, g.slots_for(j)[-1].id)
        r += 1
    return {"draws": draws, "media": g.average()}


def test_rng_matches_js(page):
    js = page.evaluate("() => { const r = PD.rngSemente(1); return [r(), r(), r()]; }")
    r = mulberry32(1)
    assert [r(), r(), r()] == js
    big = page.evaluate("() => { const r = PD.rngSemente(4294967295); return Array.from({length: 50}, r); }")
    r = mulberry32(4294967295)
    assert [r() for _ in range(50)] == big


@pytest.mark.parametrize("formacao", ["4-3-3", "3-5-2", "4-4-2", "4-2-3-1", "5-4-1"])
def test_full_games_match_js(page, game_data, formacao):
    for seed in [0, 1, 2, 99, 12345, 2**31 + 7]:
        swaps_at = [1, 4] if seed % 2 else []
        js = page.evaluate(JS_PLAY, ["grandes", formacao, seed, swaps_at])
        py = py_play(game_data, "grandes", formacao, seed, swaps_at)
        assert py["draws"] == js["draws"], (formacao, seed)
        assert py["media"] == pytest.approx(js["media"])


def test_liga_mode_matches_js(page, game_data):
    for seed in [5, 6, 7]:
        js = page.evaluate(JS_PLAY, ["liga", "4-3-3", seed, []])
        assert py_play(game_data, "liga", "4-3-3", seed, [])["draws"] == js["draws"]
