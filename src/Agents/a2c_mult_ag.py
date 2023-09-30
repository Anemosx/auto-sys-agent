import torch

from src.Agents.agent import Agent
from src.Misc.exploration import adjust_explore_value
from src.Misc.network import PPOActor, PPOCritic

from copy import deepcopy
from torch import nn


class A2C_Multiple_Agents(Agent):
    """
    Agent using Advantage Actor Critic. This agent is designed to be run parallel. It is derived from the a2c_ppo agent.
        There are different modes available:
        1. One mode which is similar to A3C. All agents use the same net and update it from time to time. This mode has
        an alternative version where the agents have their own nets and update their nets and the master_net frequently.
        Once in a while they synchronize with the master_net.
        2. The second mode has an "evolutionary" approach: Each agent has their own net and once in a while synchronizes
        to the agent which has the best return values since the last update.

    Note: This class has nothing to do with the env mode "multi". It is designed for env mode "single" and creates
    multiple agents and envs on that env mode.

    author(s): Moritz T.
    """
    def __init__(self, params, actor_master, critic_master, actor_master_optimizer, critic_master_optimizer, id):
        """
        Initialize A2C Agent with separate actor and critic networks.

        author(s): Moritz T.
        :param params: params
        """
        super(A2C_Multiple_Agents, self).__init__(params)
        self.id = id
        self.MseLoss = nn.MSELoss()
        if params.a2c_mult_ag_use_custom_master_update_frequency or params.evolution_mode:
            self.actor = PPOActor(params)
            self.critic = PPOCritic(params)
        else:
            self.actor = actor_master
            self.critic = critic_master
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.actor_alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params.critic_alpha)
        self.actor_master = actor_master
        self.critic_master = critic_master
        self.actor_master_optimizer = actor_master_optimizer
        self.critic_master_optimizer = critic_master_optimizer
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

    def update_net(self, memory, actor_optimizer, critic_optimizer, clear_memory = True):

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
        critic_optimizer.zero_grad()
        loss_value.requires_grad_(True).backward()
        if self.params.clip_critic:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.params.ppo_clip_grad)
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        loss_policy.requires_grad_(True).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.params.ppo_clip_grad)
        actor_optimizer.step()

        if self.params.ppo_weight_decay:
            self.actor_scheduler.step(None)
            self.critic_scheduler.step(None)

        # clear memory after every update --> A2C has no experience buffer
        if clear_memory:
            memory.clear()

        # adjust exploration values
        self.exploration_value = adjust_explore_value(self.params, self.exploration_value,
                                                      [self.actor_optimizer, self.critic_optimizer])

        return loss_policy.item(), loss_value.item(), loss.item()

    def adapt_net_to_master(self):
        self.actor = deepcopy(self.actor_master)
        self.critic = deepcopy(self.critic_master)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params.critic_alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.critic_alpha)

    def get_net(self):
        return self.actor, self.critic

    def adapt_to_net(self, actor, critic):
        self.actor = deepcopy(actor)
        self.critic = deepcopy(critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params.critic_alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.critic_alpha)

    def update(self, memory):
        """
        Learning method for the A2C Agent.
        Calculate losses and perform backpropagation.

        author(s): Moritz T.

        :param memory: memory, experience storage
        :return: mean of total, critic and actor loss
        """

        if self.params.episode % self.params.a2c_mult_ag_master_update_frequency == 0 \
                and self.params.a2c_mult_ag_use_custom_master_update_frequency:
            self.update_net(memory, self.actor_master_optimizer, self.critic_master_optimizer, False)

        return self.update_net(memory, self.actor_optimizer, self.critic_optimizer)
