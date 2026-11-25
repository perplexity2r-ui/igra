"""Браузерная арена: человек против обученной PPO-модели (Minecraft 1.16.5 PvP).

Один HTTP-порт, без WebSocket — так надёжнее через прокси предпросмотра:
  GET  /             — страница арены (web/play.html)
  GET  /api/state    — текущее состояние боя
  POST /api/action   — {"action": N} — сделать ход, вернуть новое состояние
  POST /api/reset    — начать новый бой
  GET  /api/health   — проверка живости

Запуск:
    .venv/bin/python human_server.py --port 8090 --model models/pvp_ppo_pro_500k.zip
"""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.pro_vec_env import ARMOR_TIERS, MELEE_POOL, TECHNIQUES, ProArena

ROOT = Path(__file__).resolve().parent
PLAY_HTML = ROOT / "web" / "play.html"


class ArenaSession:
    """Одна арена «игрок vs нейросеть»."""

    LEVELS = {"easy": 0.6, "normal": 0.85, "pro": 1.0}

    def __init__(self, model: PPO, seed: int, difficulty: float = 1.0, level: str = "normal"):
        self.model = model
        self.difficulty = difficulty
        self.level = level if level in self.LEVELS else "normal"
        self.seed = seed
        self.core = ProArena(1, seed=seed, difficulty=difficulty)
        self.core.pve = True
        self.core.set_enemy_skill(difficulty)
        self.core.reset()
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.over = False
        self.result = ""
        self.last_event = "Бой начался — вперёд!"
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(seed)
        self._tick = 0

    # --------------------------------------------------------------- helpers
    def _kit(self, B) -> dict:
        wi = int(B.weapon[0]) if B.force_weapon[0] < 0 else int(B.force_weapon[0])
        ai = int(B.armor[0]) if B.force_armor[0] < 0 else int(B.force_armor[0])
        return {
            "weapon": MELEE_POOL[wi].weapon_id,
            "damage": round(float(MELEE_POOL[wi].damage), 2),
            "cooldown": round(float(MELEE_POOL[wi].cooldown), 3),
            "armor": ARMOR_TIERS[ai].name,
            "reduction": round(float(ARMOR_TIERS[ai].reduction), 3),
            "crystals": int(B.crystals[0]), "obsidian": int(B.obsidian[0]),
            "anchors": int(B.anchors[0]), "minecarts": int(B.minecarts[0]),
            "gaps": int(B.gaps[0]), "splashes": int(B.splashes[0]),
            "pearls": int(B.pearls[0]), "buffs": int(B.buffs[0]),
            "ranged": int(B.ranged[0]), "blocks": int(B.blocks[0]),
            "cobwebs": int(B.cobwebs[0]), "lava": int(B.lava_bk[0]),
            "water": int(B.water_bk[0]),
            "shield": bool(B.has_shield[0]), "shield_up": bool(B.shield_up[0]),
            "totem": bool(B.has_totem[0]),
        }

    def snapshot(self) -> dict:
        L, E = self.core.L, self.core.E  # L = ИИ, E = человек
        def side(B, extra=None):
            d = {
                "hp": round(float(B.hp[0]), 2), "absorption": round(float(B.abs[0]), 2),
                "x": round(float(B.pos[0, 0]), 2), "z": round(float(B.pos[0, 1]), 2),
                "y": round(float(B.y[0]), 2),
                "on_fire": bool(B.fire[0] > 0), "in_web": bool(B.web[0] > 0),
                "in_lava": bool(B.lava[0] > 0), "slowed": bool(B.slow[0] > 0),
                "buff": bool(B.buff[0] > 0), "caged": bool(B.cage_until[0] > 0),
                "cooldown": round(float(B.cd[0]), 2),
            }
            d.update(self._kit(B))
            return d
        return {
            "type": "state",
            "over": self.over, "result": self.result,
            "you": side(E),
            "ai": side(L),
            "obstacles": [
                [round(float(self.core.obstacles[0, i, 0]), 2),
                 round(float(self.core.obstacles[0, i, 1]), 2),
                 round(float(self.core.obstacle_ttl[0, i]), 2)]
                for i in range(int(self.core.obstacle_n[0]))
            ],
            "distance": round(float(np.linalg.norm(E.pos[0] - L.pos[0])), 2),
            "events": [self.last_event],
            "score": {"you": self.wins, "ai": self.losses, "draws": self.draws},
            "level": self.level,
            "ai_max_hp": round(20.0 * self.LEVELS.get(self.level, 0.8), 1),
            "step": int(self.core.step_count[0]),
        }

    # -------------------------------------------------------------- game logic
    def new_fight(self) -> dict:
        with self.lock:
            self.core.reset()
            self.core._human_action = None
            self.core._ai_action = None
            self.over = False
            self.result = ""
            self._apply_level()
            self._tick = 0
            self.last_event = "Новый бой! Снаряжение выдано, противник готов."
            return self.snapshot()

    def set_level(self, level: str) -> dict:
        with self.lock:
            if level in self.LEVELS:
                self.level = level
            return self.new_fight()

    def _apply_level(self) -> None:
        """Ограничиваем запас HP у ИИ, чтобы новичку было реально выиграть."""
        frac = self.LEVELS.get(self.level, 0.8)
        cap = 20.0 * frac
        if frac < 1.0:
            self.core.L.hp[0] = min(float(self.core.L.hp[0]), cap)
            self.core.L.abs[0] = min(float(self.core.L.abs[0]), cap)

    def step(self, action: int) -> dict:
        with self.lock:
            if self.over:
                return self.new_fight()

            # ИИ выбирает действие по текущему состоянию, затем оба хода идут вместе
            obs = self.core._obs()
            ai_action = int(
                np.asarray(self.model.predict(obs, deterministic=False)[0]).reshape(-1)[0]
            )
            # гандикап: на лёгком уровне ИИ «тормозит» — действует через ход
            self._tick += 1
            if self.level == "easy" and self._tick % 2 == 0:
                ai_action = 0

            self.core.set_human_action(np.array([int(action)]))
            _, _, term, trunc, _, info = self.core.step(np.array([ai_action]))

            human_tech = int(info["human_tech"][0])
            ai_tech = int(info["tech"][0])
            if human_tech:
                self.last_event = f"Ты применил: {TECHNIQUES[human_tech]}"
            elif ai_tech:
                self.last_event = f"ИИ применил: {TECHNIQUES[ai_tech]}"

            if term[0] or trunc[0]:
                ai_won = bool(info["enemy_dead"][0])     # enemy seat = ты (человек)
                human_won = bool(info["learner_dead"][0])  # learner seat = ИИ
                self.over = True
                self.result = "win" if human_won else ("lose" if ai_won else "timeout")
                if human_won:
                    self.wins += 1
                    self.last_event = "🏆 Победа! Нейросеть повержена."
                elif ai_won:
                    self.losses += 1
                    self.last_event = "💀 Поражение. Нейросеть победила."
                else:
                    self.draws += 1
                    self.last_event = "⏱ Время вышло — ничья."
            return self.snapshot()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    session: ArenaSession

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")
        except Exception:
            return {}

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/play", "/index.html"):
            self._send(200, PLAY_HTML.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(self.session.snapshot())
        elif path == "/api/health":
            self._json({"ok": True, "step": int(self.session.core.step_count[0])})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        data = self._body()
        if path == "/api/action":
            self._json(self.session.step(int(data.get("action", 0))))
        elif path == "/api/reset":
            self._json(self.session.new_fight())
        elif path == "/api/level":
            self._json(self.session.set_level(str(data.get("level", "normal"))))
        else:
            self._send(404, b"not found", "text/plain")

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model", default="models/pvp_ppo_pro_500k.zip")
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--level", default="normal", choices=["easy", "normal", "pro"])
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    args = ap.parse_args()

    print(f"Загружаю модель {args.model} ...", flush=True)
    model = PPO.load(str(Path(args.model).with_suffix("")), device="cpu")

    Handler.session = ArenaSession(model, seed=args.seed, difficulty=args.difficulty,
                                   level=args.level)
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    httpd.daemon_threads = True
    print(f"Браузерная арена готова: http://0.0.0.0:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
