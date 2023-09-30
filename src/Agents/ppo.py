import numpy as np
import torch
from torch import nn

from src.Agents.agent import Agent
from src.Misc.exploration import adjust_explore_value
from src.Misc.network import PPOActor, PPOCritic


class PPO(Agent):
    """
    Agent using Proximal Policy Optimization.

    author(s): Arnold Unterauer
    """
    def __init__(self, params):
        """
        Initialize PPO Agent with separate actor and critic networks.

        author(s): Arnold Unterauer
        :param params: params
        """
        super(PPO, self).__init__(params)
        self.MseLoss = nn.MSELoss()
        self.actor = PPOActor(self.params).to(self.device)
        self.critic = PPOCritic(self.params).to(self.device)
        if params.ppo_weight_decay:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params.actor_alpha, weight_decay=4e-2)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.critic_alpha, weight_decay=4e-2)
            self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda it: max(0.995 ** it, 0.01))
            self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda it: max(0.995 ** it, 0.01))
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params.actor_alpha)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.critic_alpha)

    def update(self, memory):
        """
        Learning method for the PPO Agent.
        Calculate losses and perform backpropagation.

        author(s): Arnold Unterauer
        :param memory: memory, experience storage
        :return: mean of total, critic and actor loss
        """
        loss_out = []
        loss_critic_out = []
        loss_actor_out = []
        for _ in range(self.params.ppo_nr_train_epochs):
            # unpack everything from memory
            if self.params.memory_mode == "random":
                states, actions, log_p_old, returns, advantages = memory.sample_random()
            else:
                states, actions, log_p_old, returns, advantages = memory.sample_last()

            # evaluate taken actions and states
            log_p, state_values, dist_entropy = self.evaluate(states, actions)

            # using log probs instead of render probs allows us to use addition instead of multiplication
            ratios = torch.exp(log_p - log_p_old)

            # calculate surrogate objectives
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.params.ppo_clip, 1 + self.params.ppo_clip) * advantages

            # calculate value/critic and policy/actor losses
            loss_entropy = self.params.entropy_weight * dist_entropy
            loss_value = 0.5 * self.MseLoss(returns, state_values)
            loss_policy = - torch.min(surr1, surr2).mean() - loss_entropy.mean()
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

            # for logging purposes
            loss_actor_out.append(loss_policy.item())
            loss_critic_out.append(loss_value.item())
            loss_out.append(loss.item())

        if self.params.reset_memory:
            memory.clear()

        # adjust exploration values
        self.exploration_value = adjust_explore_value(self.params, self.exploration_value,
                                                      [self.actor_optimizer, self.critic_optimizer])

        return np.array(loss_out).mean(), np.array(loss_critic_out).mean(), np.array(loss_actor_out).mean()
