import torch

from src.Agents.agent import Agent
from src.Misc.exploration import adjust_explore_value
from src.Misc.network import PPOActor, PPOCritic
from torch import nn


class A2C(Agent):
    """
    Agent using Advantage Actor Critic. This version of A2C is adapted as closely to our PPO as possible.

    author(s): Moritz T.
    """

    def __init__(self, params):
        """
        Initialize A2C Agent with separate actor and critic networks.

        author(s): Moritz T.
        :param params: params
        """
        super(A2C, self).__init__(params)
        self.actor = PPOActor(params)
        self.critic = PPOCritic(params)
        self.MseLoss = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.actor_alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params.critic_alpha)
        self.a2c_clip_grad = params.a2c_clip_grad
        if params.a2c_learner_gamma_decay:
            # this will increase the learner gamma from 0.8 gradually to 0.99 based on the current episode
            # change the factor to a desired number to change the amount of increase
            self.actor_scheduler = torch.optim.lr_scheduler \
                .LambdaLR(self.actor_optimizer, lr_lambda=lambda it: min(0.8 + self.params.episode * 0.0002, 0.99))
            self.critic_scheduler = torch.optim.lr_scheduler \
                .LambdaLR(self.critic_optimizer, lr_lambda=lambda it: min(0.8 + self.params.episode * 0.0002, 0.99))

        if not params.use_entropy:
            self.params.entropy_weight = 0

    def update(self, memory):
        """
        Learning method for the A2C Agent.
        Calculate losses and perform backpropagation.

        author(s): Moritz T.
        :param memory: memory, experience storage
        :return: mean of total, critic and actor loss
        """

        # unpack everything from memory
        states, actions, log_p_old, returns, advantages = memory.sample_last()

        # evaluate taken actions and states
        log_p, state_values, dist_entropy = self.evaluate(states, actions)

        # calculate value/critic and policy/actor losses
        loss_entropy = self.params.entropy_weight * dist_entropy
        loss_value = 0.5 * self.MseLoss(returns, state_values)
        loss_policy = - (log_p * advantages).mean() - loss_entropy.mean()
        loss = loss_policy + loss_value

        # optimization step
        self.critic_optimizer.zero_grad()
        loss_value.requires_grad_(True).backward()
        if self.params.clip_critic:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.params.ppo_clip_grad)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        loss_policy.requires_grad_(True).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.params.ppo_clip_grad)
        self.actor_optimizer.step()

        if self.params.ppo_weight_decay:
            self.actor_scheduler.step(None)
            self.critic_scheduler.step(None)

        memory.clear()

        # adjust exploration values
        self.exploration_value = adjust_explore_value(self.params, self.exploration_value,
                                                  [self.actor_optimizer, self.critic_optimizer])

        return loss_policy.item(), loss_value.item(), loss.item()
