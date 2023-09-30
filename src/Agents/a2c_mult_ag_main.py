import os
import traceback
import gym
import numpy as np
import sys

from src.Misc.environment_util import setup_env, setup_spaces, reset_env, step_env
from src.Misc.logger import Logger
from src.Misc.plotting import Plotter
from src.Agents.a2c_mult_ag import A2C_Multiple_Agents as A2C_Multiple_Agents
from src.Misc.memory import Memory
from src.Misc.util import to_tensor, intermediate_weight_save
import torch
from src.Misc.network import PPOActor, PPOCritic

"""
This document is based on the original main file. 
Therefore, it was originally made by all who made the main file
The adaption is made by the persons stated here.

author(s): Moritz T.
"""

def main(params, params_json):
    """
    Setup of the experiment from params.json. This main is adapted to work with a2c_multi_ag (for more info see
    description in a2c_multi_ag)

    Note: This class has nothing to do with the env mode "multi". It is designed for env mode "single" and creates
    multiple agents and envs on that env mode.

    author(s): Moritz T.
    """

    if params.a2c_mult_ag_use_custom_master_update_frequency and params.train_mode == "max_steps":
        raise Exception("Custom update frequency and max_steps do not work together! "
                        "Turn of custom update frequency or use train_mode 'episode'")


    # setup environments for individual agents
    if params.unity_env:
        envs = [setup_env(params, i) for i in range(params.nr_agents)]
    else:
        envs = [gym.make(params.env) for _ in range(params.nr_agents)]

    # setup observation and action shapes
    setup_spaces(envs[0], params)

    # create logging tools and directory as specified in the params
    log = Logger(params, params_json)
    plot = Plotter(params)

    # create runners to run episodes
    runners = [Runner(envs[i], log, plot, params, i) for i in range(params.nr_agents)]

    # perform experiment
    try:
        run_experiment(params, envs, runners)
    except:
        for env in envs:
            env.close()
            traceback.print_exc()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


class Runner:
    """
    Runner to run episodes.

    author(s): Moritz T.
    """

    def __init__(self, env, log, plot, params, id=0):
        self.id = id
        self.params = params
        self.env = env
        self.state = None
        self.multi_done = []
        self.done = True
        self.steps = 0
        self.loss = 0
        self.loss_critic = 0
        self.loss_actor = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.log = log
        self.plot = plot
        self.first_run = True

    def reset(self):
        """
        Resets environment.

        author(s): Arnold Unterauer, Moritz T.
        """
        self.episode_reward = 0
        self.done = False
        self.state = reset_env(self.params, self.env)
        self.multi_done = np.zeros(len(self.state), dtype=int)
        self.log.on_episode_start()

    def run(self, agent, memory):
        """
        Perform runs in the environment.

        author(s): Arnold Unterauer, Moritz T.
        :param agent: learning agent (a2c or ppo)
        :param memory: memory, experience storage
        """
        self.steps = 0

        # before the very first run the environment needs to be reset
        # afterwards it will be reset after done=True
        if self.first_run or self.params.env_mode == "multi":
            self.reset()
            self.first_run = False

        while self.steps < self.params.horizon or self.params.train_mode == "episode":
            # perform one step
            self.run_step(agent, memory)
            self.steps += 1
            if self.done:
                self.end_episode(agent)
                self.reset()
                if self.params.episode >= self.params.training_episodes or \
                        self.params.train_mode == "episode" or self.params.train_mode == "episode_horizon":
                    break
        if (self.params.episode >= self.params.warmup_epi and self.params.episode % self.params.update_period == 0) \
                or (self.params.train_mode == "max_steps" and self.params.algorithm == "a2c_mult_ag"):
            # append last next state value to memory and process episode memory
            memory.memory.append((to_tensor(self.state, self.params.device), None, None, None,
                                  to_tensor(agent.act(self.state)[2], self.params.device), None))
            memory.process_memory()
            # train agent
            self.loss, self.loss_critic, self.loss_actor = agent.update(memory)
        if self.params.env_mode == "multi" and not self.done and self.params.episode < self.params.training_episodes:
            self.end_episode(agent)

    def run_step(self, agent, memory):
        """
        Perform one step in the environment and saves experience into memory.

        author(s): Arnold Unterauer, Moritz T.
        :param agent: learning agent (a2c or ppo)
        :param memory: memory, experience storage
        """
        if self.params.render and self.params.nr_episode % self.params.render_frequency == 0:
            self.env.render()
        action, log_p, value = agent.act(to_tensor(self.state))
        next_state, reward, done, info, self.done, self.multi_done = step_env(self.params, self.multi_done, self.env, action)
        memory.add((self.state, action, reward, log_p, value, done))
        self.state = next_state
        self.episode_reward += np.mean(reward)

    def end_episode(self, agent):
        """
        Log and plot at the end of the episode.

        author(s): Moritz T.
        :param agent: learning agent (a2c or ppo)
        """
        if self.params.algorithm == "a2c_mult_ag":
            if self.id == self.params.nr_agents - 1:
                self.params.episode += 1
        else:
            self.params.episode += 1

        self.episode_rewards.append(self.episode_reward)
        self.plot.occasionally_plot(self.params.episode, self.episode_rewards)
        self.log.on_episode_finish(self.episode_reward, self.loss_critic, self.loss_actor, self.loss,
                                   agent.exploration_value, self.params.episode, self.steps, self.episode_rewards,
                                   agent)
        intermediate_weight_save(self.params, self.log.log_dir, self.params.episode, agent)


def run_experiment(params, envs, runners):
    """
    Initialize agents and perform experiment.

    author(s): Moritz T.
    :param params: params
    :param envs: environments
    :param runners: Runners to perform the actual episodes
    """

    # setup agent
    # each agent can have own memory if desired by user
    memories = [Memory(params) for _ in range(params.nr_agents)]
    # alternatively all agents can use the same memory
    memory = Memory(params)
    # uses the A2C Network for each individual agent
    actor = PPOActor(params)
    critic = PPOCritic(params)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=params.actor_alpha)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=params.critic_alpha)
    # create multiple a2c agents
    agents = [A2C_Multiple_Agents(params, actor, critic, actor_optimizer, critic_optimizer, i) for i in
              range(params.nr_agents)]

    # load agent weights not supported for a2c_mult_ag yet

    # if params.load_weights:
    #     path = os.path.join(os.getcwd(), "logs", params.weights_dir, "")
    #     agent.load_weights(path, params.weights_episode)

    # initialize log on experiment start
    for runner in runners:
        runner.log.on_experiment_start()

    # run episodes
    while True:
        for i, runner in enumerate(runners):
            # check which memory mode is wished
            if params.a2c_mult_ag_shared_memory:
                a2c_mult_ag_memory = memory
            else:
                a2c_mult_ag_memory = memories[i]
            #run the experiment
            runner.run(agents[i], a2c_mult_ag_memory)
        # After each run it is checked if the agents nets should be updated
        if params.episode % params.a2c_mult_ag_master_adaption_frequency == 0 and params.episode > params.a2c_mult_ag_delayed_start_at:
            # in evolution mode the net of the best agent is chosen to be the new net for all other agents
            if params.evolution_mode:
                best_agent = [0, 0]
                for i, runner in enumerate(runners):
                    mean_reward = np.mean(runner.episode_rewards[-params.a2c_mult_ag_master_adaption_frequency:])
                    if mean_reward > best_agent[1]:
                        best_agent[0] = i
                        best_agent[1] = mean_reward
                best_actor, best_critic = agents[best_agent[0]].get_net()
            for agent in agents:
                # adapt nets of agents
                if params.evolution_mode:
                    agent.adapt_to_net(best_actor, best_critic)
                elif params.a2c_mult_ag_use_custom_master_update_frequency:
                    # there is also the option to adapt the nets to a master net which
                    # gets updates from all the agents over time
                    agent.adapt_net_to_master()
            if params.evolution_mode:
                print(f"\nAdapted agents nets to agent {best_agent[0]} "
                      f"which had highest mean reward of {best_agent[1]} "
                      f"in the last {params.a2c_mult_ag_master_adaption_frequency} episodes")
            elif params.a2c_mult_ag_use_custom_master_update_frequency:
                print("Adapted agents nets to master net")
        if params.episode >= params.training_episodes:
            break

    # closing log on experiment end
    for i, runner in enumerate(runners):
        runner.log.on_experiment_finish(runner.episode_rewards, agents[i])

    # plot rewards for first runner only
    if params.plot:
        runners[0].plot.plot_experiment(params.training_episodes, runner.episode_rewards)

    # shutdown environment
    for env in envs:
        env.close()


if __name__ == '__main__':
    main()
