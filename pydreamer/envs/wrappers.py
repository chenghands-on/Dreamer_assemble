import time
from logging import exception
from typing import Callable

import gym
import gym.spaces
import numpy as np
import datetime
import uuid


class DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = ...  # TODO

    def observation(self, obs):
        if isinstance(obs, dict):
            return obs  # Already a dictionary
        if len(obs.shape) == 1:
            return {'vecobs': obs}  # Vector env
        else:
            return {'image': obs}  # Image env


class TimeLimitWrapper(gym.Wrapper):

    def __init__(self, env, time_limit):
        super().__init__(env)
        self.time_limit = time_limit

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # type: ignore
        self.step_ += 1
        # print(self.step_)
        if self.step_ >= self.time_limit:
            done = True
            info['time_limit'] = True
        return obs, reward, done, info

    def reset(self):
        self.step_ = 0
        return self.env.reset()  # type: ignore


class ActionRewardResetWrapper(gym.Wrapper):

    def __init__(self, env, no_terminal):
        super().__init__(env)
        self.env = env
        self.no_terminal = no_terminal
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        self.action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(action, int):
            action_vec = np.zeros(self.action_size)
            action_vec[action] = 1.0
        else:
            assert isinstance(action, np.ndarray) and action.shape == (self.action_size,), "Wrong one-hot action shape"
            action_vec = action
        obs['action'] = action_vec
        obs['reward'] = np.array(reward)
        obs['terminal'] = np.array(False if self.no_terminal or 'TimeLimit.truncated' in info or info.get('time_limit') else done)
        obs['reset'] = np.array(False)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs['action'] = np.zeros(self.action_size)
        obs['reward'] = np.array(0.0)
        obs['terminal'] = np.array(False)
        obs['reset'] = np.array(True)
        return obs


class CollectWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.episode = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode.append(obs.copy())
        if done:
            episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
            info['episode'] = episode
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.episode = [obs.copy()]
        return obs


class OneHotActionWrapper(gym.Wrapper):
    """Allow to use one-hot action on a discrete action environment."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Note: we don't want to change env.action_space to Box(0., 1., (n,)) here,
        # because then e.g. RandomPolicy starts generating continuous actions.

    def step(self, action):
        if not isinstance(action, int):
            action = action.argmax()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


class RestartOnExceptionWrapper(gym.Wrapper):

    def __init__(self, constructor: Callable):
        self.constructor = constructor
        env = constructor()
        super().__init__(env)
        self.env = env
        self.last_obs = None

    def step(self, action):
        try:
            obs, reward, done, info = self.env.step(action)
            self.last_obs = obs
            return obs, reward, done, info
        except:
            exception('Error in env.step() - terminating episode.')
            # Dummy observation to terminate episode. time_limit=True to not count as terminal
            return self.last_obs, 0.0, True, dict(time_limit=True)

    def reset(self):
        while True:
            try:
                obs = self.env.reset()
                self.last_obs = obs
                return obs
            except:
                exception('Error in env.reset() - recreating env.')
                try:
                    self.env.close()
                except:
                    pass
                try:
                    self.env = self.constructor()
                except:
                    pass
            time.sleep(1)


# Wrappers from V3
class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()

    def action_space(self):
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.discrete = True
        return space

    def step(self, action):
        # index = np.argmax(action).astype(int)
        # reference = np.zeros_like(action)
        # reference[index] = 1
        if not isinstance(action, int):
            action = action.argmax()
        return self.env.step(action)
        # if not np.allclose(reference, action):
        #     raise ValueError(f"Invalid one-hot action:\n{action}")
        # return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation_space(self):
        spaces = self.env.observation_space.spaces
        if "reward" not in spaces:
            spaces["reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "reward" not in obs:
            obs["reward"] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "reward" not in obs:
            obs["reward"] = 0.0
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        # return self.env.step(action[self._key])
        return self.env.step(action)


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()