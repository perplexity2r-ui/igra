"""Gymnasium environment for Minecraft PvP (1.16.5).
Talks to bot/bot.js over WebSocket. Retries connection for 90 seconds
so training can start before bots are fully ready.
"""
import json
import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import websocket  # websocket-client


class MinecraftPvPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="ws://localhost:8765", retries=45, delay=2.0):
        super().__init__()
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(10,), dtype=np.float32
        )
        self.ws = None
        last_err = None
        for i in range(retries):
            try:
                self.ws = websocket.create_connection(host, timeout=60)
                print(f"Connected to bots at {host}")
                break
            except Exception as e:
                last_err = e
                print(f"Waiting for bots... ({i + 1}/{retries}) {e}")
                time.sleep(delay)
        if self.ws is None:
            raise ConnectionError(
                f"Could not connect to bots at {host} after {retries} tries. "
                f"Is 'node bot.js' running? Last error: {last_err}"
            )

    def _recv(self):
        msg = json.loads(self.ws.recv())
        return np.array(msg["obs"], dtype=np.float32), msg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ws.send(json.dumps({"reset": True}))
        obs, _ = self._recv()
        return obs, {}

    def step(self, action):
        self.ws.send(json.dumps({"action": int(action)}))
        obs, msg = self._recv()
        info = msg.get("info", {})
        done = bool(msg.get("done", False))

        reward = -0.01
        reward += info.get("damage_dealt", 0) * 2.0
        reward -= info.get("damage_taken", 0) * 1.0
        if info.get("enemy_dead"):
            reward += 100.0
        if info.get("learner_dead"):
            reward -= 100.0

        return obs, float(reward), done, False, info

    def close(self):
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
