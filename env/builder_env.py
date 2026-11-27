"""A compact PPO design arena for procedurally built Minecraft houses.

Each episode is one seven-decision build plan (footprint, palette, roof, windows,
entry, ornament, landscaping). The reward combines a handcrafted style-fit proxy
and architectural compatibility. `builder.render` maps plans to blocks, and
`builder.export_plan` writes a Minecraft Java function. This is deliberately a
separate task/model from the PvP environment.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

STAGES = ("footprint", "palette", "roof", "windows", "entry", "ornament", "landscape")
N_STAGES = len(STAGES)
N_OPTIONS = 8
DESIGN_CHOICES = (N_OPTIONS,) * N_STAGES
THEMES = ("cottage", "alpine", "desert", "fantasy")
THEME_NAMES = {
    "cottage": "Woodland Cottage",
    "alpine": "Alpine Lodge",
    "desert": "Desert Courtyard",
    "fantasy": "Enchanted Keep",
}
OPTION_NAMES = (
    ("cozy square", "alpine chalet", "courtyard villa", "timber tower",
     "L-shaped lodge", "open pavilion", "twin wings", "modern studio"),
    ("oak and cobble", "spruce and stone", "sandstone and acacia", "dark oak gothic",
     "birch and brick", "mossy manor", "enchanted teal", "red-sand adobe"),
    ("steep gable", "hip roof", "mansard", "flat terrace", "pagoda eaves",
     "shed roof", "stepped roof", "spire roof"),
    ("symmetric casements", "tall arch", "dormer windows", "wraparound glass",
     "lattice panes", "bay windows", "stained glass", "minimal slits"),
    ("central porch", "stone arch", "veranda", "side porch", "grand steps",
     "recessed entry", "double entry", "raised balcony"),
    ("chimney", "dormer", "cupola", "flower boxes", "corner turret",
     "lantern beams", "carved trim", "water feature"),
    ("cottage garden", "pine grove", "courtyard fountain", "enchanted grove",
     "stone path", "pond", "orchard", "formal garden"),
)

# Per-option affinity for cottage, alpine, desert, and fantasy themes. These are
# reward targets, not pretrained/PvP weights.
AFFINITY = np.array([
    [[.98,.56,.38,.58],[.56,.99,.24,.48],[.48,.40,.98,.52],[.38,.68,.24,.99],
     [.84,.89,.50,.52],[.66,.56,.88,.87],[.88,.70,.50,.72],[.56,.38,.76,.55]],
    [[.99,.55,.24,.47],[.58,.99,.30,.60],[.32,.25,.99,.40],[.42,.50,.24,.99],
     [.70,.48,.78,.72],[.84,.75,.42,.56],[.50,.56,.68,.99],[.62,.40,.96,.52]],
    [[.99,.99,.35,.65],[.80,.84,.79,.70],[.64,.56,.48,.99],[.47,.36,.88,.70],
     [.26,.38,.70,.91],[.62,.65,.70,.60],[.56,.56,.78,.95],[.42,.69,.24,.99]],
    [[.98,.91,.84,.86],[.76,.82,.60,.99],[.66,.99,.36,.80],[.58,.42,.99,.68],
     [.58,.68,.56,.99],[.99,.80,.75,.77],[.76,.56,.54,.99],[.50,.42,.70,.55]],
    [[.99,.94,.66,.69],[.78,.74,.86,.99],[.94,.78,.88,.69],[.78,.89,.62,.72],
     [.88,.72,.90,.95],[.62,.48,.72,.62],[.82,.75,.88,.75],[.48,.62,.70,.99]],
    [[.99,.92,.34,.62],[.74,.99,.36,.82],[.58,.50,.74,.99],[.99,.50,.68,.62],
     [.54,.76,.48,.99],[.82,.90,.80,.91],[.84,.72,.86,.96],[.45,.42,.98,.68]],
    [[.99,.62,.60,.55],[.62,.99,.35,.68],[.56,.50,.99,.72],[.72,.60,.73,.99],
     [.86,.89,.81,.85],[.75,.62,.98,.70],[.95,.58,.76,.66],[.80,.72,.92,.90]],
], dtype=np.float32)

# Explicit architectural affinities supplement the broad theme matching.
FOOTPRINT_ROOF = np.array([
    [.98,.82,.55,.52,.38,.68,.72,.45],
    [.94,.86,.72,.42,.38,.62,.70,.50],
    [.66,.96,.70,.91,.48,.55,.62,.40],
    [.58,.55,.72,.66,.55,.62,.70,.99],
    [.82,.84,.68,.55,.38,.62,.65,.50],
    [.58,.76,.44,.86,.99,.52,.91,.72],
    [.87,.80,.60,.48,.42,.58,.78,.54],
    [.50,.62,.58,.99,.32,.86,.58,.50],
], dtype=np.float32)
FOOTPRINT_WINDOWS = np.array([
    [.99,.82,.68,.60,.72,.92,.78,.58],
    [.90,.75,.78,.58,.62,.91,.77,.54],
    [.78,.80,.65,.99,.74,.78,.68,.56],
    [.64,.99,.72,.48,.66,.62,.76,.65],
    [.84,.78,.80,.66,.71,.84,.70,.58],
    [.82,.74,.68,.88,.99,.82,.72,.62],
    [.92,.80,.70,.68,.72,.90,.82,.58],
    [.92,.84,.62,.78,.48,.78,.58,.90],
], dtype=np.float32)
FOOTPRINT_ENTRY = np.array([
    [.99,.82,.88,.80,.86,.75,.82,.60],
    [.92,.85,.96,.84,.88,.68,.82,.62],
    [.82,.99,.86,.72,.94,.75,.86,.62],
    [.70,.84,.64,.72,.78,.74,.76,.99],
    [.94,.87,.92,.78,.90,.70,.88,.65],
    [.82,.76,.96,.68,.84,.74,.72,.88],
    [.92,.84,.86,.90,.94,.72,.98,.70],
    [.72,.76,.84,.65,.86,.90,.75,.62],
], dtype=np.float32)

PAIR_STAGES = ((0, 2, FOOTPRINT_ROOF), (0, 3, FOOTPRINT_WINDOWS),
               (0, 4, FOOTPRINT_ENTRY))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-8 else 0.0


def design_quality(plan: np.ndarray | list[int], theme: int) -> float:
    """Normalized heuristic style score in [0, 1] for seven design decisions."""
    choices = np.asarray(plan, dtype=np.int32)
    if choices.shape != (N_STAGES,) or np.any(choices < 0) or np.any(choices >= N_OPTIONS):
        raise ValueError(f"plan must contain {N_STAGES} option indices in 0..{N_OPTIONS - 1}")
    if not 0 <= int(theme) < len(THEMES):
        raise ValueError(f"theme must be between 0 and {len(THEMES) - 1}")
    theme = int(theme)
    local = [float(AFFINITY[s, choices[s], theme]) for s in range(N_STAGES)]
    pairs = [float(matrix[choices[i], choices[j]]) for i, j, matrix in PAIR_STAGES]
    # Style coherence across palette, roof, facade openings, and ornament.
    style_pairs = ((1, 2), (1, 3), (1, 5), (2, 5), (4, 6), (5, 6))
    for i, j in style_pairs:
        pairs.append(_cosine(AFFINITY[i, choices[i]], AFFINITY[j, choices[j]]))
    theme_fit = float(np.mean(local))
    coherence = float(np.mean(pairs))
    # A little preference for balanced ornament and landscape rather than empty lots
    # or over-decorated facades. Every option retains a positive score floor.
    ornament = int(choices[5])
    landscape = int(choices[6])
    density = 0.95 if (ornament in (0, 1, 3, 5, 6) and landscape != 7) else 0.82
    quality = 0.58 * theme_fit + 0.37 * coherence + 0.05 * density
    return float(np.clip(quality, 0.0, 1.0))


def _partial_pair_scores(plan: list[int], theme: int) -> list[float]:
    completed = len(plan)
    scores: list[float] = []
    for i, j, matrix in PAIR_STAGES:
        if j < completed:
            scores.append(float(matrix[plan[i], plan[j]]))
    for i, j in ((1, 2), (1, 3), (1, 5), (2, 5), (4, 6), (5, 6)):
        if j < completed:
            scores.append(_cosine(AFFINITY[i, plan[i]], AFFINITY[j, plan[j]]))
    return scores


class BuilderDesignEnv(gym.Env):
    """Seven-step PPO environment whose policy selects a coherent building plan."""

    metadata = {"render_modes": []}
    OBS_DIM = N_STAGES + len(THEMES) + N_STAGES * N_OPTIONS + 4 + 3

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(N_OPTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, (self.OBS_DIM,), np.float32)
        self.stage = 0
        self.theme = 0
        self.rotation = 0
        self.lot_scale = 0
        self.season = 0
        self.plan: list[int] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.stage = 0
        if options and "theme" in options:
            requested_theme = options["theme"]
            if isinstance(requested_theme, str):
                if requested_theme not in THEMES:
                    raise ValueError(f"Unknown theme {requested_theme!r}; choose from {THEMES}")
                self.theme = THEMES.index(requested_theme)
            else:
                self.theme = int(requested_theme)
                if not 0 <= self.theme < len(THEMES):
                    raise ValueError(f"theme must be between 0 and {len(THEMES) - 1}")
        else:
            self.theme = int(self.np_random.integers(len(THEMES)))
        self.rotation = int(self.np_random.integers(4))
        self.lot_scale = int(self.np_random.integers(3))
        self.season = int(self.np_random.integers(4))
        self.plan = []
        return self._observation(), {"theme": self.theme, "rotation": self.rotation}

    def _observation(self) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        if self.stage < N_STAGES:
            obs[self.stage] = 1.0
        theme0 = N_STAGES
        obs[theme0 + self.theme] = 1.0
        plan0 = theme0 + len(THEMES)
        for s, action in enumerate(self.plan):
            obs[plan0 + s * N_OPTIONS + action] = 1.0
        variation0 = plan0 + N_STAGES * N_OPTIONS
        obs[variation0 + self.rotation] = 1.0
        obs[variation0 + 4 + self.lot_scale] = 1.0
        return obs

    def step(self, action: int):
        choice = int(np.asarray(action).item())
        if not 0 <= choice < N_OPTIONS:
            raise ValueError(f"Builder action must be 0..{N_OPTIONS - 1}, got {choice}")
        stage = self.stage
        self.plan.append(choice)
        self.stage += 1

        affinity = float(AFFINITY[stage, choice, self.theme])
        pairs = _partial_pair_scores(self.plan, self.theme)
        pair_reward = float(np.mean(pairs)) if pairs else 0.72
        # Dense, bounded reward guides early choices; terminal score rewards a coherent whole.
        reward = 0.22 * affinity + 0.12 * pair_reward
        terminated = self.stage >= N_STAGES
        info: dict[str, Any] = {"stage": STAGES[stage], "choice": choice}
        if terminated:
            quality = design_quality(self.plan, self.theme)
            reward += 3.0 * quality
            info.update({
                "quality": quality,
                "theme": self.theme,
                "theme_name": THEME_NAMES[THEMES[self.theme]],
                "plan": list(self.plan),
                "plan_names": [OPTION_NAMES[i][self.plan[i]] for i in range(N_STAGES)],
                "rotation": self.rotation,
                "lot_scale": self.lot_scale,
                "season": self.season,
            })
        return self._observation(), float(reward), terminated, False, info

    def describe_plan(self) -> dict[str, Any]:
        if len(self.plan) != N_STAGES:
            raise RuntimeError("A plan is available only after all seven design choices")
        return {
            "theme": self.theme,
            "theme_name": THEME_NAMES[THEMES[self.theme]],
            "rotation": self.rotation,
            "lot_scale": self.lot_scale,
            "season": self.season,
            "choices": list(self.plan),
            "names": [OPTION_NAMES[i][self.plan[i]] for i in range(N_STAGES)],
            "quality": design_quality(self.plan, self.theme),
        }


# Stable import name used by the training/sampling tools.
BuilderEnv = BuilderDesignEnv
