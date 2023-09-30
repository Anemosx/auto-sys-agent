import torch
from torch.distributions import Normal

from src.Misc.exploration import explore
from src.Misc.util import to_tensor


class Agent:
    """
    Learning agent (a2c or ppo).

    author(s): Arnold Unterauer, Moritz T.
    """
    def __init__(self, params):
        self.params = params
        self.action_space = self.params.action_space
        self.exploration_value = self.params.exploration_value
        self.learner_gamma = self.params.learner_gamma
        self.device = torch.device(self.params.device)
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.entropy = 0
        self.repeat_actions = params.repeat_actions
        self.end_repetition_after_episode = params.end_repetition_after_episode
        self.action_repeat_max = params.action_repeat
        self.action_repeat = 0
        self.saved_actions = []
        pass

    def save_weights(self, path, weights_episode):
        """
        Save weights of agents.

        author(s): Arnold Unterauer
        :param weights_episode: save weights at episode n
        :param path: directory path to weights store location
        """
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict()
                    }, path + "{}-weights-actor.pth".format(weights_episode))

        torch.save({'critic_state_dict': self.critic.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
                    }, path + "{}-weights-critic.pth".format(weights_episode))

    def load_weights(self, path, weights_episode):
        """
        Load weights of agents.

        author(s): Arnold Unterauer
        :param weights_episode: load weights from episode n
        :param path: directory path to weights load location
        """
        actor_checkpoint = torch.load(path + "{}-weights-actor.pth".format(weights_episode), map_location=self.params.device)
        self.actor.load_state_dict(actor_checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['actor_optimizer_state_dict'])

        critic_checkpoint = torch.load(path + "{}-weights-critic.pth".format(weights_episode), map_location=self.params.device)
        self.critic.load_state_dict(critic_checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['critic_optimizer_state_dict'])

        self.actor.eval()
        self.critic.eval()

    def act(self, state):
        """
        Depending on the state / observation, select the next action to perform in the environment.

        author(s): Arnold Unterauer, Moritz T.
        :param state: state / observation
        :type state: tensor([ ])
        :return: action, log probability of action, value of state
        """
        # gain action and standard deviation from actor
        action_mean, std = self.actor(to_tensor(state, self.params.device))
        # predict state value with critic
        state_value = self.critic(to_tensor(state, self.params.device))
        state_value = state_value.squeeze(1)
        # sample action using Normal distribution
        dist = Normal(action_mean, std)
        if self.params.use_exploration:
            action = explore(self.params, self.exploration_value)
        else:
            action = dist.sample()
        log_p = dist.log_prob(action)
        return action.detach().cpu().data.numpy(), log_p.detach().cpu().data.numpy(), state_value.cpu().detach().data.numpy()

    def evaluate(self, states, actions):
        """
        Evaluate states and actions on the current agent networks.

        author(s): Arnold Unterauer, Moritz T.
        :param states: states / observations
        :type states: tensor([[ ], ..., [ ]])
        :param actions: taken actions
        :type actions: tensor([[ ], ..., [ ]])
        :return: log probabilities of taken actions, state values, distribution entropy
        """
        # predict state values with critic
        state_values = self.critic(states)
        # gain action and standard deviation from current actor
        action_mean, std = self.actor(states)
        dist = Normal(action_mean, std)
        # calculate current log probabilities of taken actions
        log_p = dist.log_prob(actions)
        # get entropy of the Normal distribution
        dist_entropy = dist.entropy()
        return log_p, state_values, dist_entropy

    def update(self, memory):
        pass
