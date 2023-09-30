import os
import sys
import gym
import numpy as np
import torch


def setup_env(params, id=0):
    """
    Setup unity environment.

    author(s): Arnold Unterauer, Moritz T., Matthias F., Sina S.
    :param params: params
    :type params: DotMap
    :return: unity environment
    """
    path = None
    env = None

    if not params.unity_env:
        return gym.make(params.env)

    if params.env_mode == "single":
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from gym_unity.envs import UnityToGymWrapper
        # overcome annoying warning
        gym.logger.set_level(40)

        if params.os == "Linux":
            path = os.path.join(os.getcwd(), "envs", params.env, params.env_version, "crawler.x86_64")
        elif params.os == "Windows":
            path = os.path.join(os.getcwd(), "envs", params.env, params.env_version)
        elif params.os == "Darwin":
            path = os.path.join(os.getcwd(), "envs", params.env, "macos", params.env_version, params.env_version)
        else:
            print("\u001b[31m The platform {} is currently not supported.\u001b[0m".format(params.os))
            exit(1)

        channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(path, seed=42, worker_id=id, side_channels=[channel])
        channel.set_configuration_parameters(time_scale=params.time_scale)
        env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

    if params.env_mode == "multi":
        if sys.version_info[0] == 3 and sys.version_info[1] > 6:
            print("\u001b[31m Using multi environment requires python version 3.6 or lower.\u001b[0m")
            exit(1)

        from unityagents import UnityEnvironment
        if params.os == "Linux":
            path = os.path.join(os.getcwd(), "envs", params.env, "Crawler_Linux_NoVis", "Crawler.x86")
        elif params.os == "Windows":
            path = os.path.join(os.getcwd(), "envs", params.env, "Crawler_Windows_x86_64", "Crawler.exe")
        elif params.os == "Darwin":
            path = os.path.join(os.getcwd(), "envs", params.env, "Crawler.app")
        else:
            print("\u001b[31m The platform {} is currently not supported.\u001b[0m".format(params.os))
            exit(1)
        env = UnityEnvironment(file_name=path, worker_id=42)

    return env


def setup_spaces(env, params):
    """
    Configure action and observation spaces.

    author(s): Arnold Unterauer, Moritz T.
    :param env: environment
    :param params: params
    :type params: DotMap
    :return: params
    :rtype: DotMap
    """
    if params.env_mode == "single" or not params.unity_env:
        # observation shape
        if isinstance(env.observation_space, gym.spaces.Tuple):
            params.observation_shape = env.observation_space[0].shape[0]
        elif isinstance(env.observation_space, gym.spaces.Box):
            params.observation_shape = env.observation_space.shape[0]
        else:
            print("\u001b[31m Unknown env observation space type\u001b[0m")
            exit(1)

        # action shape
        if isinstance(env.action_space, gym.spaces.Box):
            params.continuous = True
            params.nr_actions = env.action_space.shape[0]
            params.action_space = env.action_space
            params.action_range = np.array([abs(env.action_space.high[i]) + abs(env.action_space.low[i]) for i in range(params.nr_actions)])
            params.action_mid = np.array([env.action_space.low[i] + params.action_range[i] / 2 for i in range(params.nr_actions)])
        elif isinstance(env.action_space, gym.spaces.Discrete):
            params.continuous = False
            params.nr_actions = env.action_space.n
            print("\u001b[31m Discrete environments are no longer supported\u001b[0m")
            exit(1)
        else:
            print("\u001b[31m Unknown env action space type\u001b[0m")
            exit(1)

    if params.env_mode == "multi" and params.unity_env:
        # get the environment brain
        params.brain_name = env.brain_names[0]
        brain = env.brains[params.brain_name]
        env_info = env.reset(train_mode=True)[params.brain_name]

        # observation shape
        params.observation_shape = env_info.vector_observations.shape[1]

        # action shape
        params.nr_actions = brain.vector_action_space_size


def map_action(action, params):
    """
    Maps action to environment action space.
    Centers action to the mid of the action space and stretch action into action space.

    author(s): Arnold Unterauer
    :param action: action(s), usually in [-1, 1]
    :type action: tensor([ ])
    :param params: params
    :type params: DotMap

    :return: actions mapped to action space
    :rtype: list
    """
    action = torch.clamp(action, -1, 1)
    env_actions = action * params.action_range / 2 + params.action_mid
    return env_actions.detach().numpy().astype("float32")


def sanitize_reward(params, reward):
    """
    Frees reward from nan values.

    author(s): Arnold Unterauer
    :param params: params
    :param reward: step reward
    :return: sanitized reward
    """
    reward_clean = []
    for r in reward:
        if np.isnan(r):
            reward_clean.append(- params.PENALTY)
        else:
            reward_clean.append(r)
    return reward_clean


def reset_env(params, env):
    """
    Resets environment.

    author(s): Arnold Unterauer
    :param params: params
    :param env: environment
    :return: start state of the environment
    """
    if not params.unity_env:
        return [env.reset()]

    if params.env_mode == "single":
        return env.reset()

    if params.env_mode == "multi":
        env_info = env.reset(train_mode=True)[params.brain_name]
        return env_info.vector_observations


def step_env(params, multi_done, env, action):
    """
    Take a step in the environment depending on the environment mode.

    author(s): Arnold Unterauer, Moritz T.
    :param params: params
    :param multi_done: signals if episode is done
    :type multi_done: list
    :param env: environment
    :param action: action in the environment
    :return: (next_state, reward, done, env_info, runner_done, multi_done) from environment as lists
    """
    if not params.unity_env:
        next_state, reward, runner_done, info = env.step(action)
        next_state = [np.concatenate(next_state)]
        return next_state, reward, [runner_done], info, runner_done, []

    if params.env_mode == "single":
        next_state, reward, runner_done, info = env.step(action)
        if np.isnan(reward):
            reward = -params.PENALTY
        return next_state, [reward], [runner_done], info, runner_done, []

    if params.env_mode == "multi":
        env_info = env.step(action)[params.brain_name]
        reward = env_info.rewards
        next_state = env_info.vector_observations
        done = env_info.local_done
        runner_done = False
        multi_done = [1 if done[i] else multi_done[i] for i in range(len(done))]
        if np.all(multi_done):
            runner_done = True
        if np.any(np.isnan(reward)):
            reward = sanitize_reward(params, reward)
        return next_state, reward, done, env_info, runner_done, multi_done
