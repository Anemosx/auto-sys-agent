import json
import torch
import numpy as np
import sys
import platform
from dotmap import DotMap
from torch import nn
import os

from src.Misc.preset import use_preset


def to_tensor(x, device="cpu"):
    """
    Convert array / np.array to tensor.

    author(s): Moritz T.
    :param x: array / np.array
    :param device: device the tensor is assigned to

    :return: x as tensor
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float().to(device)


def clip_grad_norm_(module, max_grad_norm):
    """
    Clip gradients of network / module.

    author(s): Moritz T.
    :param module: network / module to clip
    :param max_grad_norm: max norm of the gradients
    """
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


def intermediate_weight_save(params, log_dir, nr_episode, agent):
    """
    Occasionally save agent weights.

    author(s): Moritz T.
    :param params: params
    :param log_dir: directory path to weights store location
    :param nr_episode: number of the current episode
    :param agent: learning agent (a2c or ppo)
    """
    if params.local_logs and nr_episode > 0 and nr_episode % params.agent_save_frequency == 0 and params.interim_saves:
        agent.save_weights(os.path.join(log_dir, ""), nr_episode)
        print("Agent saved at episode {}".format(nr_episode))


def load_params():
    """
    Load params from params.json.

    author(s): Arnold Unterauer, Moritz T., Matthias F.
    :return: params
    """
    try:
        with open(sys.argv[1]) as f:
            params_json = json.load(f)
        with open('params.json', 'w') as outfile:
            json.dump(params_json, outfile)
    except IndexError:
        print("No path to params specified. Use default")

    # open params.json as DotMap
    params_path = os.path.join(os.getcwd(), "params.json")
    with open(params_path, 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)
    return params, params_json


def load_saved_params(params):
    """
    Load saved params from directory.

    author(s): Arnold Unterauer
    :param params: params
    """
    params_path = os.path.join(os.getcwd(), "logs", params.save_dir, "params.json")
    with open(params_path, 'r') as f:
        params_json = json.load(f)
    params_loaded = DotMap(params_json)
    params.update(params_loaded)


def adjust_params(params):
    """
    Setup params according to settings.

    author(s): Arnold Unterauer, Moritz T.
    :param params: params
    :type params: DotMap
    """
    # load params from checkpoint
    if params.load_params:
        load_saved_params(params)
    # load algorithm presets
    if params.use_preset:
        use_preset(params)
    # for displaying entropy in the plot visualization
    params.entropyWeights = []
    params.episode = 0
    if params.device == "auto":
        params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params.os = platform.system()
    # change settings to enable slurm execution
    if params.slurm_build:
        params.plot = 0
        params.carbon_tracker = 0
        params.render = 0
        params.env_version = "linux-server"
        params.os = "Linux"
    if params.memory_mode == "episode":
        params.reset_memory = True
    if params.horizon <= 0:
        params.horizon = params.memory_batch_size
    # set entropy of loss to 0
    if not params.use_entropy:
        params.entropy_weight = 0.0
        params.entropy_weight_min = 0.0
    # disable warmup
    if not params.warmup:
        params.warmup_epi = 0
    # if we need discounted or normalized return
    if params.a2c_discount_rewards:
        params.calc_episode = True


def calculate_returns(rewards, dones, gamma):
    """
    Calculate discounted and normalized rewards of undiscounted rewards. (deprecated)

    author(s): Moritz T.
    :param rewards: undiscounted rewards
    :type rewards: list
    :param dones: dones in the memory / batch
    :param gamma: discount factor
    :return: discounted and normalized rewards
    """
    r = 0
    discounted_returns = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + r * gamma * (1 - done)
        discounted_returns.append(r)
    discounted_returns = discounted_returns[::-1]
    normalized_returns = (discounted_returns - np.mean(discounted_returns))
    normalized_returns /= (np.std(discounted_returns) + np.finfo(np.float32).eps.item())
    return discounted_returns, normalized_returns


def calculate_gae(params, agent, states, next_states, rewards, dones):
    """
    Calculate Generalized Advantage Estimation. (deprecated)

    author(s): Arnold Unterauer
    :param params: params
    :param agent: agent to evaluate states
    :param states: states observed
    :param next_states: next states observed
    :param rewards: undiscounted rewards
    :param dones: dones in the memory / batch
    :return: Generalized Advantage Estimation
    """
    state_values = agent.critic(to_tensor(states))
    state_values = state_values.squeeze().tolist()
    next_state_values = state_values[1:]
    last_value = agent.critic(to_tensor(next_states[-1]))
    next_state_values.append(last_value.item())
    dones = np.array(dones)
    batch_size = len(states)
    advantages = np.zeros(batch_size + 1)

    for t in reversed(range(batch_size)):
        delta = rewards[t] + (params.learner_gamma * next_state_values[t] * dones[t]) - state_values[t]
        advantages[t] = delta + (params.learner_gamma * params.gae_tau * advantages[t + 1] * dones[t])
    return advantages
