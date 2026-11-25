"""Gymnasium environment for Minecraft PvP (1.16.5) — Pro Meta Crystal, TNT-Cart & Anchor Mode.

Talks to `bot/bot.js` over WebSocket (`Box(30,)` observations, `Discrete(28)` actions).
"""
from __future__ import annotations

import json
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import websocket  # websocket-client


class MinecraftPvPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "ws://127.0.0.1:8765",
        retries: int = 45,
        delay: float = 2.0,
        ready_retries: int = 30,
    ):
        super().__init__()
        self.host = host
        self.retries = retries
        self.delay = delay
        self.ready_retries = ready_retries

        self.action_space = spaces.Discrete(28)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(30,), dtype=np.float32
        )
        self.ws: websocket.WebSocket | None = None
        self._prev_dist: float | None = None
        self._connect()

    def _connect(self) -> None:
        self.close()
        last_err: Exception | None = None
        for i in range(self.retries):
            try:
                self.ws = websocket.create_connection(self.host, timeout=60)
                print(f"Connected to bots at {self.host}")
                return
            except Exception as e:
                last_err = e
                print(f"Waiting for bots... ({i + 1}/{self.retries}) {e}")
                time.sleep(self.delay)
        raise ConnectionError(
            f"Could not connect to bots at {self.host} after {self.retries} tries. "
            f"Is 'node bot/bot.js' running? Last error: {last_err}"
        )

    def _send_and_recv(self, payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        for attempt in range(3):
            try:
                if self.ws is None:
                    self._connect()
                assert self.ws is not None
                self.ws.send(json.dumps(payload))
                raw = self.ws.recv()
                msg = json.loads(raw)
                raw_obs = np.array(msg["obs"], dtype=np.float32)
                if raw_obs.shape != self.observation_space.shape:
                    padded = np.zeros(self.observation_space.shape, dtype=np.float32)
                    n = min(raw_obs.size, padded.size)
                    padded[:n] = raw_obs[:n]
                    raw_obs = padded
                obs = np.clip(
                    np.nan_to_num(raw_obs, nan=0.0, posinf=2.0, neginf=-1.0),
                    self.observation_space.low,
                    self.observation_space.high,
                )
                return obs, msg
            except Exception as e:
                print(f"WebSocket communication error (attempt {attempt + 1}/3): {e}")
                time.sleep(1.0)
                self._connect()
        raise ConnectionError(f"Lost WebSocket connection to {self.host}")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info: dict[str, Any] = {}
        payload: dict[str, Any] = {"reset": True}
        if options and "kit" in options:
            payload["kit"] = int(options["kit"])

        for attempt in range(self.ready_retries):
            obs, msg = self._send_and_recv(payload)
            info = msg.get("info", {})
            if info.get("ready", True):
                break
            if attempt == 0 or (attempt + 1) % 5 == 0:
                print(f"Waiting for both bots to spawn in arena... ({attempt + 1}/{self.ready_retries})")
            time.sleep(2.0)

        self._prev_dist = float(obs[2]) * 20.0
        return obs, info

    def step(self, action: int | np.ndarray):
        act = int(np.asarray(action).item())
        obs, msg = self._send_and_recv({"action": act})
        info = msg.get("info", {})
        done = bool(msg.get("done", False))
        truncated = bool(info.get("timeout", False) or info.get("bot_disconnected", False))
        terminated = bool(done and not truncated)

        reward = -0.01
        damage_dealt = float(info.get("damage_dealt", 0.0))
        damage_taken = float(info.get("damage_taken", 0.0))
        healed = float(info.get("healed", 0.0))
        reward += damage_dealt * 2.5
        reward -= damage_taken * 1.4
        reward += healed * 0.4

        if info.get("shield_broken"):
            reward += 4.5
        if info.get("blocked_hit"):
            reward += 2.5
        if info.get("trapped_enemy"):
            reward += 3.0
        if info.get("tactical_cover"):
            reward += 2.0
        if info.get("feet_shield_mitigated"):
            reward += 4.5
        if info.get("explosive_combo"):
            reward += 5.0
        if info.get("totem_popped_enemy"):
            reward += 8.0
        if info.get("wasted_consumable"):
            reward -= 1.5

        curr_dist = float(obs[2]) * 20.0
        if self._prev_dist is not None and curr_dist > 3.0:
            reward += float(np.clip((self._prev_dist - curr_dist) * 0.05, -0.05, 0.05))
        self._prev_dist = curr_dist

        if info.get("enemy_dead"):
            reward += 100.0
        if info.get("learner_dead"):
            reward -= 100.0

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass
        finally:
            self.ws = None
