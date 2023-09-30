import torch
import torch.nn.functional as F
import numpy as np

from src.Agents.agent import Agent
from src.Misc.exploration import adjust_explore_value
from src.Misc.network import A2CActor, A2CCritic
from src.Misc.util import clip_grad_norm_


class A2C(Agent):
    """
    Agent using Advantage Actor Critic.

    author(s): Moritz T.
    """
    def __init__(self, params):
        """
        Initialize A2C Agent with separate actor and critic networks.

        author(s): Moritz T.
        :param params: params
        """
        super(A2C, self).__init__(params)
        self.actor = A2CActor(params)
        self.critic = A2CCritic(params)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.actor_alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params.critic_alpha)
        self.a2c_clip_grad = params.a2c_clip_grad
        if params.a2c_learner_gamma_decay:
            # this will increase the learner gamma from 0.8 gradually to 0.99 based on the current episode
            # change the factor to a desired number to change the amount of increase
            self.actor_scheduler = torch.optim.lr_scheduler\
                .LambdaLR(self.actor_optimizer, lr_lambda=lambda it: min(0.8 + self.params.episode * 0.0002, 0.99))
            self.critic_scheduler = torch.optim.lr_scheduler\
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

        # actor
        if self.params.a2c_advantage_logp:
            advantage = (-log_p * advantages.detach()).mean()
        else:
            advantage = advantages.detach().mean()

        actor_loss = advantage - dist_entropy.mean() * self.params.entropy_weight
        self.actor_optimizer.zero_grad()
        actor_loss.requires_grad_(True).backward()
        clip_grad_norm_(self.actor_optimizer, self.a2c_clip_grad)
        self.actor_optimizer.step()


        #critic
        critic_loss = 0.5 * F.mse_loss(returns, state_values) #critic loss is weighted with 0.5 -> seems to be best according to literature
        self.critic_optimizer.zero_grad()
        critic_loss.requires_grad_(True).backward()
        clip_grad_norm_(self.critic_optimizer, self.a2c_clip_grad)
        self.critic_optimizer.step()

        # compute new parameter values due to decays
        self.exploration_value = adjust_explore_value(self.params, self.exploration_value,
                                                      [self.actor_optimizer, self.critic_optimizer])
        if not self.params.a2c_use_softplus:
            self.params.dist_std = max(self.params.dist_std_min, self.params.dist_std - self.params.dist_std_decay)
            self.actor.update_std()

        if self.params.a2c_learner_gamma_decay:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        #clear memory after every update --> A2C has no experience buffer
        memory.clear()

        loss = critic_loss + actor_loss

        return np.array(loss.detach().numpy()).mean(),\
               np.array(critic_loss.detach().numpy()).mean(),\
               np.array(actor_loss.detach().numpy()).mean()
