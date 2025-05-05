from __future__ import annotations

import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
from typing import Any, SupportsFloat


class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.observation_space = Discrete(2)
        self.action_space = Discrete(2)
        self.state = 0
        self.horizon = 10
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        self.state = 0
        self.steps = 0
        return int(self.state), {}

    def step(self, action: int) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (expected 0 or 1)")
        self.state = action
        reward = float(action)
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.horizon
        return int(self.state), reward, terminated, truncated, {}


    def get_reward_per_action(self) -> np.ndarray:
        return np.array([[0, 1], [0, 1]], dtype=float)

    def get_transition_matrix(self) -> np.ndarray:
        T = np.zeros((2, 2, 2))
        T[:, 0, 0] = 1.0
        T[:, 1, 1] = 1.0
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._maybe_corrupt(obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._maybe_corrupt(obs), reward, terminated, truncated, info

    def _maybe_corrupt(self, obs: int) -> int:
        if self.rng.random() < self.noise:
            return 1 - obs  # flip between 0 and 1
        return obs
