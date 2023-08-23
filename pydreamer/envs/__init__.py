# Ignore annoying warnings from imported envs
import warnings
warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")  # gym

import gym
import numpy as np

from . import wrappers


def create_env(env_id: str, no_terminal: bool, env_time_limit: int, env_action_repeat: int, worker_id: int,conf,info_only=False):
    wm_type=conf.wm_type
    # if wm_type=="v2":
    if env_id.startswith('MiniGrid-'):
        from .minigrid import MiniGrid
        env = MiniGrid(env_id)

    elif env_id.startswith('Atari-'):
        from .atari import Atari_v2
        env = Atari_v2(env_id.split('-')[1].lower(), action_repeat=env_action_repeat)
        # print('yes')
        
    elif env_id.startswith('AtariGray-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat, grayscale=True)

    elif env_id.startswith('MiniWorld-'):
        import gym_miniworld.wrappers as wrap
        env = gym.make(env_id)
        env = wrap.DictWrapper(env)
        env = wrap.MapWrapper(env)
        env = wrap.AgentPosWrapper(env)
        if env_id.startswith('MiniWorld-ScavengerHunt'):
            env = wrap.GoalPosWrapper(env)
            env = wrap.GoalVisibleWrapper(env)
            env = wrap.GoalVisAgeWrapper(env)

    elif env_id.startswith('DmLab-'):
        from .dmlab import DmLab
        env = DmLab(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat)
        env = wrappers.DictWrapper(env)
    
    elif env_id.startswith('DMM-'):
        from .dmm import DMMEnv
        env = DMMEnv(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat, worker_id=worker_id)
        env = wrappers.DictWrapper(env)

    elif env_id.startswith('MineRL'):
        from .minerl import MineRL
        constr = lambda: MineRL(env_id, action_repeat=env_action_repeat)
        env = wrappers.RestartOnExceptionWrapper(constr)

    elif env_id.startswith('DMC-'):
        from .dmc import DMC_v2
        env = DMC_v2(env_id.split('-', maxsplit=1)[1].lower(), action_repeat=env_action_repeat)
    
    elif env_id.startswith('Embodied-'):
        from .embodied import EmbodiedEnv
        task = env_id.split('-', maxsplit=1)[1].lower()
        env = EmbodiedEnv(task, action_repeat=env_action_repeat, time_limit=env_time_limit)
        env_time_limit = 0  # This is handled by embodied.Env

    else:
        env = gym.make(env_id)
        env = wrappers.DictWrapper(env)

    if hasattr(env.action_space, 'n'):
        env = wrappers.OneHotActionWrapper(env)
    if env_time_limit > 0:
        env = wrappers.TimeLimitWrapper(env, env_time_limit)
    env = wrappers.ActionRewardResetWrapper(env, no_terminal)
    env = wrappers.CollectWrapper(env)
    if info_only:
        return env.observation_space,env.action_space
    else:
        return env
    # elif wm_type=="v3":
    #     suite, task = env_id.split("-", 1)
    #     suite=suite.lower()
    #     task=task.lower()
    #     if suite == "dmc":
    #         from .dmc import DMC_v3

    #         env = DMC_v3(task, env_action_repeat, conf.size)
    #         env = wrappers.NormalizeActions(env)
    #     elif suite == "atari":
    #         from .atari import Atari_v3

    #         env = Atari_v3(
    #             task,
    #             env_action_repeat,
    #             conf.size,
    #             gray=conf.grayscale,
    #             noops=conf.noops,
    #             lives=conf.lives,
    #             sticky=conf.stickey,
    #             actions=conf.actions,
    #             resize=conf.resize,
    #         )
    #         env = wrappers.OneHotAction(env)
    #     elif suite == "dmlab":
    #         import dmlab as dmlab

    #         env = dmlab.DeepMindLabyrinth(
    #             task, mode if "train" in mode else "test", conf.action_repeat
    #         )
    #         env = wrappers.OneHotAction(env)
    #     elif suite == "MemoryMaze":
    #         from memorymaze import MemoryMaze

    #         env = MemoryMaze(task)
    #         env = wrappers.OneHotAction(env)
    #     elif suite == "crafter":
    #         import crafter as crafter

    #         env = crafter.Crafter(task, conf.size)
    #         env = wrappers.OneHotAction(env)
    #     elif suite == "minecraft":
    #         import minecraft as minecraft

    #         env = minecraft.make_env(task, size=conf.size, break_speed=conf.break_speed)
    #         env = wrappers.OneHotAction(env)
    #     else:
    #         raise NotImplementedError(suite)
    #     env = wrappers.TimeLimit(env, conf.time_limit)
    #     env = wrappers.SelectAction(env, key="action")
    #     env = wrappers.UUID(env)
    #     env = wrappers.RewardObs(env)
    #     env = wrappers.ActionRewardResetWrapper(env, no_terminal)
    #     env = wrappers.CollectWrapper(env)
    #     if info_only:
    #         return env.observation_space,env.action_space
    #     else:
    #         return env


