"""Play Onze Prime "Três Grandes" in a real browser with Playwright.

Two modes:

* **autopilot** (`LiveBot`): the bot reads the 6 options, mirrors the game in a
  shadow `engine.Game`, asks a policy, and clicks. In `hunt` it abandons an attempt
  as soon as 995 becomes impossible and restarts in-page (no reload), until the record
  XI appears or the attempt/time budget runs out.
  Fair-play: /api/live, /api/ranking, /api/partilha, /api/visitas, /api/og and analytics
  are blocked, so bot games never touch the public counter or the ranking.

* **copilot** (`Copilot`): YOU play in a visible browser. The tool reads the page and
  shows a box with the recommended pick (and outlines it), says when to swap, and tells
  you as soon as the attempt is dead so you can restart. No clicks, no network changes:
  it's your normal game.

CLI:
    python -m onzeprime.live_bot hunt --formation 3-5-2 --attempts 200 --minutes 30
    python -m onzeprime.live_bot copilot --formation 3-5-2
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import data, optimizer
from .engine import Draw, Game
from .strategies import POLICIES, Decision

BASE_URL = data.BASE_URL
BLOCKED = re.compile(r"/api/(live|ranking|partilha|visitas|og)|posthog|/_vercel/insights")
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
TARGET = 995


def make_policy(policy_name: str, pool, formation: str, params: Optional[dict] = None):
    cls = POLICIES[policy_name]
    return cls() if policy_name == "greedy" else cls(pool, formation, **(params or {}))


class _Base:
    def __init__(self, formation: str, policy_name: str, policy_params: Optional[dict]):
        self.formation = formation
        self.clubs, self.by_club = data.load()
        self.pool = optimizer.pool_for_mode(self.by_club, "grandes")
        self.policy = make_policy(policy_name, self.pool, formation, policy_params)
        self.club_id_by_name = {c.nome.casefold(): c.id for c in self.clubs}

    def _match(self, club_id: str, nome: str, ovr: Optional[int] = None):
        cands = [pl for pl in self.by_club[club_id] if pl.nome == nome and (ovr is None or pl.ovr == ovr)]
        if len(cands) != 1:
            raise RuntimeError(f"Could not match {nome} ({ovr}) at {club_id}: {len(cands)} candidates")
        return cands[0]

    def read_options(self, page) -> Dict:
        if page.locator("#painel .res-cartao").count():
            return {"finished": True}
        cards = page.locator("#painel .opcao:not(.fantasma)")
        if not cards.count():
            return {"finished": False, "ready": False}
        club = page.locator("#sorteio-nome").text_content().strip()
        opts = [{"nome": cards.nth(i).locator(".op-nome").text_content().strip(),
                 "ovr": int(cards.nth(i).locator(".op-ovr").text_content().strip())} for i in range(cards.count())]
        swaps = int(page.locator("#painel .btn-troca b").text_content().strip())
        return {"finished": False, "ready": True, "club": club, "options": opts, "swaps": swaps}


# ============================================================================ autopilot
class LiveBot(_Base):
    def __init__(self, formation: str = "3-5-2", policy_name: str = "hunter", policy_params: Optional[dict] = None,
                 headed: bool = False, slow_mo: int = 0):
        super().__init__(formation, policy_name, policy_params)
        self.headed, self.slow_mo = headed, slow_mo
        self.log: List[dict] = []

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=not self.headed, slow_mo=self.slow_mo)
        self.ctx = self.browser.new_context(reduced_motion="reduce", locale="pt-PT", viewport={"width": 430, "height": 900})
        self.ctx.route(BLOCKED, lambda route: route.abort())
        self.page = self.ctx.new_page()
        self.page.set_default_timeout(15_000)
        self.page.goto(BASE_URL + "/", wait_until="domcontentloaded")
        return self

    def __exit__(self, *exc):
        self.ctx.close()
        self.browser.close()
        self._pw.stop()

    # ---- navigation (no reloads: restarting is pure in-page JS)
    def new_game(self):
        p = self.page
        if p.locator("#vista-inicio").is_hidden():
            p.click("#btn-marca")                      # back to the home screen
        p.click("#btn-grandes")
        p.wait_for_selector("#folha-nova:not([hidden])")
        p.locator("#nova-formacoes button.formacao", has_text=self.formation).first.click()
        p.click("#nova-jogar")
        p.wait_for_selector("#painel .opcao:not(.fantasma)")

    def _click_pick(self, player, slot_id: str, game: Game):
        p = self.page
        p.locator("#painel .opcao", has=p.locator(".op-nome", has_text=player.nome)).first.click()
        if len(game.roles_for(player)) > 1:  # the site asks where on the pitch
            papel = next(s.papel for s in game.slots if s.id == slot_id)
            p.locator(f'#vagas .vaga.alvo button.vaga-vazia[aria-label="Colocar {player.nome} a {papel}"]').first.click()

    # ---- one attempt
    def play_attempt(self, abort_when_dead: bool = True) -> dict:
        t0 = time.perf_counter()
        self.new_game()
        g = Game(self.clubs, self.by_club, "grandes", self.formation, seed=0, auto_draw=False)
        alive_picks, status = 0, "finished"
        can_check = hasattr(self.policy, "ceiling_now")
        while True:
            obs = self.read_options(self.page)
            if obs["finished"]:
                break
            if not obs["ready"]:
                self.page.wait_for_selector("#painel .opcao:not(.fantasma), #painel .res-cartao")
                continue
            club_id = self.club_id_by_name[obs["club"].casefold()]
            g.swaps = obs["swaps"]
            g.draw = Draw(club_id, [self._match(club_id, o["nome"], o["ovr"]) for o in obs["options"]])
            d: Decision = self.policy(g)
            if d.kind == "swap":
                self.page.click("#painel .btn-troca")
                g.actions.append(["t"])
                continue
            self._click_pick(d.player, d.slot_id, g)
            g.xi[d.slot_id] = d.player
            g.persons.add(d.player.id)
            g.actions.append(["e", d.player.id, d.slot_id])
            if can_check and status != "dead":
                if self.policy.ceiling_now(g) >= TARGET - 1e-9:
                    alive_picks += 1
                else:
                    status = "dead"
                    if abort_when_dead:
                        break
        total = sum(pl.ovr for pl in g.xi.values())
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "formation": self.formation,
               "policy": getattr(self.policy, "name", "?"), "status": status, "picks": len(g.xi),
               "alive_picks": alive_picks, "swaps_used": sum(1 for a in g.actions if a[0] == "t"),
               "sum": total, "record": len(g.xi) == 11 and total >= TARGET,
               "secs": round(time.perf_counter() - t0, 3),
               "lineup": "; ".join(f"{s}:{pl.nome}({pl.ovr})" for s, pl in g.xi.items())}
        if len(g.xi) == 11:
            rec["shown"] = self.page.locator("#painel .res-num").text_content().strip()
        self.log.append(rec)
        return rec

    # ---- hunt loop
    def _status(self, html: str):
        try:
            self.page.evaluate(_OVERLAY_JS, [html, None])
        except Exception:
            pass

    def hunt(self, max_attempts: int = 100, max_minutes: float = 30.0, verbose_every: int = 10,
             keep_open: bool = False) -> dict:
        t0 = time.time()
        for i in range(1, max_attempts + 1):
            rec = self.play_attempt(abort_when_dead=True)
            if self.headed:
                best = max(r["alive_picks"] for r in self.log)
                self._status(f"Attempt <b>{i}</b> · {time.time() - t0:.0f}s · last: alive {rec['alive_picks']}/11"
                             f" · best so far: <b>{best}/11</b>")
            if rec["record"]:
                RUNS_DIR.mkdir(exist_ok=True)
                shot = RUNS_DIR / f"record_{datetime.now():%Y%m%d_%H%M%S}.png"
                self.page.screenshot(path=str(shot), full_page=True)
                print(f"RECORD at attempt {i}: {rec['shown']} · {rec['lineup']} · screenshot {shot}")
                if self.headed:
                    self._status(f"<span style='color:#4ade80'>RECORD XI {rec['shown']}</span> at attempt {i}. "
                                 "Autopilot game: not eligible for the ranking.")
                if keep_open and self.headed:
                    print("Window left open. Close it to finish.")
                    self.page.wait_for_event("close", timeout=0)
                break
            if verbose_every and i % verbose_every == 0:
                print(f"{i:>5} attempts · {time.time() - t0:6.0f}s · best alive {max(r['alive_picks'] for r in self.log)}/11")
            if time.time() - t0 > max_minutes * 60:
                break
        return self.stats()

    def stats(self) -> dict:
        secs = [r["secs"] for r in self.log]
        return {"attempts": len(self.log), "records": sum(r["record"] for r in self.log),
                "mean_secs_per_attempt": statistics.mean(secs) if secs else None,
                "mean_picks_per_attempt": statistics.mean(r["picks"] for r in self.log) if secs else None,
                "secs_per_pick": sum(secs) / max(1, sum(r["picks"] for r in self.log)),
                "best_alive": max((r["alive_picks"] for r in self.log), default=0)}

    def save_log(self) -> Path:
        RUNS_DIR.mkdir(exist_ok=True)
        path = RUNS_DIR / f"live_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in self.log for k in r}))
            w.writeheader()
            w.writerows(self.log)
        return path


# ============================================================================ copilot
_OVERLAY_JS = """
([html, target]) => {
  let box = document.getElementById('op-copilot');
  if (!box) {
    box = document.createElement('div');
    box.id = 'op-copilot';
    box.style.cssText = 'position:fixed;left:8px;right:8px;bottom:8px;z-index:99999;padding:10px 12px;border-radius:10px;' +
      'font:600 14px/1.35 system-ui,sans-serif;background:#111;color:#fff;box-shadow:0 4px 18px rgba(0,0,0,.35);pointer-events:none';
    document.body.appendChild(box);
  }
  box.innerHTML = html;
  document.querySelectorAll('#painel .opcao').forEach(b => { b.style.outline = ''; b.style.outlineOffset = ''; });
  if (target) {
    document.querySelectorAll('#painel .opcao').forEach(b => {
      const n = b.querySelector('.op-nome');
      if (n && n.textContent.trim() === target) { b.style.outline = '4px solid #22c55e'; b.style.outlineOffset = '2px'; }
    });
  }
}
"""


class Copilot(_Base):
    """You click; the tool advises. Reads the pitch + options from the page every ~0.4 s."""

    def __init__(self, formation: str = "3-5-2", policy_name: str = "hunter", policy_params: Optional[dict] = None):
        super().__init__(formation, policy_name, policy_params)
        self.club_name_to_id = self.club_id_by_name

    def _read_xi(self, page, g: Game):
        """Pitch slots are rendered in formation order; filled ones carry title 'Nome · Clube 2003/04'."""
        titles = page.eval_on_selector_all("#vagas .vaga", "vs => vs.map(v => { const f = v.querySelector('.ficha'); return f ? f.title : null; })")
        g.xi.clear()
        g.persons.clear()
        for slot, title in zip(g.slots, titles):
            if not title:
                continue
            nome, rest = title.split(" · ", 1)
            club_name, epoca = rest.rsplit(" ", 1)
            pl = next(p for p in self.by_club[self.club_name_to_id[club_name.casefold()]] if p.nome == nome and p.epoca == epoca)
            g.xi[slot.id] = pl
            g.persons.add(pl.id)

    def advise(self, page) -> Optional[str]:
        if page.locator("#vista-jogo").is_hidden():
            return None
        form = page.evaluate("() => { try { return JSON.parse(localStorage.getItem('onzeprime.v1') || '{}').formacao } catch (e) { return null } }")
        if form and form != self.formation:
            return f"<b>Copilot</b> is set for {self.formation}, but this game is {form}. Start a {self.formation} game."
        obs = self.read_options(page)
        if obs.get("finished") or not obs.get("ready"):
            return None
        g = Game(self.clubs, self.by_club, "grandes", self.formation, seed=0, auto_draw=False)
        self._read_xi(page, g)
        g.swaps = obs["swaps"]
        club_id = self.club_name_to_id[obs["club"].casefold()]
        g.draw = Draw(club_id, [self._match(club_id, o["nome"], o["ovr"]) for o in obs["options"]])
        ceiling = self.policy.ceiling_now(g)
        if ceiling < TARGET - 1e-9:
            return (f"<span style='color:#f87171'>995 is no longer possible</span> (best now {ceiling:.0f} = {ceiling / 11:.2f}). "
                    "Tap the ONZE PRIME logo and start a new game.", None)
        d = self.policy(g)
        if d.kind == "swap":
            return ("Still alive, but none of these 6 keeps 995: <b>tap Trocar</b>.", None)
        papel = next(s.papel for s in g.slots if s.id == d.slot_id)
        return (f"Alive ✔ ({len(g.xi)}/11 placed). Pick <b>{d.player.nome}</b> ({d.player.ovr}) → <b>{papel}</b>", d.player.nome)

    def run(self, minutes: float = 60.0):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            ctx = browser.new_context(locale="pt-PT", viewport={"width": 430, "height": 900})
            page = ctx.new_page()
            page.goto(BASE_URL + "/")
            print(f"Copilot running for {minutes} min. Choose Três Grandes → {self.formation} in the browser window.")
            last, t_end = None, time.time() + minutes * 60
            while time.time() < t_end and not page.is_closed():
                try:
                    adv = self.advise(page)
                    if adv != last:
                        html, target = adv if isinstance(adv, tuple) else (adv or "", None)
                        if html:
                            page.evaluate(_OVERLAY_JS, [html, target])
                        last = adv
                except Exception as e:  # page mid-render, closed, etc.
                    if "closed" in str(e).lower():
                        break
                time.sleep(0.4)
            browser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("hunt", help="autopilot: abort dead attempts, restart, stop at the record")
    h.add_argument("--formation", default="3-5-2")
    h.add_argument("--policy", default="hunter", choices=list(POLICIES))
    h.add_argument("--params", default="{}", help='JSON, e.g. \'{"rescue_from_round": 3}\'')
    h.add_argument("--attempts", type=int, default=50)
    h.add_argument("--minutes", type=float, default=15)
    h.add_argument("--headed", action="store_true", help="show the browser window")
    h.add_argument("--slow-mo", type=int, default=0, help="ms delay per browser action (easier to watch)")
    h.add_argument("--keep-open", action="store_true", help="leave the window open when the record appears")
    c = sub.add_parser("copilot", help="you play; on-page recommendations")
    c.add_argument("--formation", default="3-5-2")
    c.add_argument("--params", default="{}")
    c.add_argument("--minutes", type=float, default=60)
    args = ap.parse_args()
    if args.mode == "hunt":
        with LiveBot(args.formation, args.policy, json.loads(args.params), headed=args.headed, slow_mo=args.slow_mo) as bot:
            st = bot.hunt(args.attempts, args.minutes, keep_open=args.keep_open)
            print(st)
            print("saved", bot.save_log())
    else:
        Copilot(args.formation, "hunter", json.loads(args.params)).run(args.minutes)


if __name__ == "__main__":
    main()
