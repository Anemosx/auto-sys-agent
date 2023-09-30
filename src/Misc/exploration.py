import random
import numpy as np
import torch


def eps_greedy(params, epsilon=1.0):
    """
    Use epsilon greedy approach to sample action.

    author(s): Arnold Unterauer, Moritz T.
    :param params: params
    :param epsilon: current epsilon
    :return: action
    """
    action = None
    if torch.rand(1) <= epsilon:
        # random network output from -1 to 1
        action = [random.uniform(-1, 1) for _ in range(params.nr_actions)]
    return torch.tensor(action)


def adjust_explore_value(params, exploration_value, optimizers):
    """
    Adjust exploration values.

    author(s): Arnold Unterauer, Moritz T.
    :param params: params
    :param exploration_value: current exploration value
    :param optimizers: Optimizer of the agent networks [actor, critic]
    :type optimizers: list
    :return: new exploration value
    """
    if params.use_exploration:
        if params.exploration_mode == "epsilon_greedy":
            params.exploration_value = max(exploration_value - params.exploration_decay, params.exploration_min)
        if params.exploration_mode == "alpha":
            # use sigmoid to calculate new learning rates
            x = -5 * params.current_episode/params.training_episodes
            s = (1/(1+np.exp(-x))) * params.exploration_value * 2
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['lr'] = max(s, params.exploration_min)
    if params.use_entropy:
        params.entropy_weight = max(params.entropy_weight - params.entropy_decay, params.entropy_weight_min)
    return params.exploration_value


def explore(params, exploration_value):
    """
    Use exploration to sample new action.

    author(s): Arnold Unterauer, Moritz T.
    :param params: params
    :param exploration_value: current exploration value
    :return: action if selected, None otherwise
    """
    if params.exploration_mode == "random":
        return eps_greedy(params, 1.0)
    if params.exploration_mode == "epsilon_greedy":
        return eps_greedy(params, exploration_value)
    return None
