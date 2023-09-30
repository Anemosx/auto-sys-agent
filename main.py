import os
import traceback
import numpy as np
import sys

from src.Misc.logger import Logger
from src.Misc.plotting import Plotter
from src.Agents.ppo import PPO as PPO_Agent
from src.Agents.a2c import A2C as A2C_Agent
from src.Agents.a2c_ppo import A2C as A2C_PPO_Agent
from src.Misc.memory import Memory
from src.Misc.environment_util import setup_env, setup_spaces, reset_env, step_env
from src.Misc.util import to_tensor, adjust_params, intermediate_weight_save, load_params
from src.Agents.a2c_mult_ag_main import main as a2c_mult_ag_main


def main():
    """
    Setup of the experiment from params.json.

    author(s): Arnold Unterauer, Moritz T.
    """
    # load params
    params, params_json = load_params()

    # adjust params to work correctly
    adjust_params(params)

    if params.algorithm == "a2c_mult_ag":
        # a2c_mult_ag spawns several runners and environments
        a2c_mult_ag_main(params, params_json)
    else:
        # setup environment
        env = setup_env(params)

        # setup observation and action shapes
        setup_spaces(env, params)

        # create logging tools and directory as specified in the params
        log = Logger(params, params_json)
        plot = Plotter(params)

        # create runner to run episodes
        runner = Runner(env, log, plot, params)

        # perform experiment
        try:
            run_experiment(params, env, runner)
        except:
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
    def __init__(self, env, log, plot, params):
        self.params = params
        self.env = env
        self.state = None
        self.done = True
        self.multi_done = []
        self.steps = 0
        self.log_steps = 0
        self.loss = 0
        self.loss_critic = 0
        self.loss_actor = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.log = log
        self.plot = plot

    def reset(self):
        """
        Resets environment.

        author(s): Arnold Unterauer, Moritz T.
        """
        self.episode_reward = 0
        self.log_steps = 0
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
        while self.steps < self.params.horizon or self.params.train_mode == "episode":
            # perform one step
            self.run_step(agent, memory)
            self.steps += 1
            self.log_steps += 1
            if self.done:
                self.end_episode(agent)
                self.reset()
                if self.params.episode >= self.params.training_episodes or \
                        self.params.train_mode == "episode" or self.params.train_mode == "episode_horizon":
                    break
        if (self.params.episode >= self.params.warmup_epi and self.params.episode % self.params.update_period == 0)\
                or (self.params.algortihm == "a2c" and self.params.train_mode == "max_steps"):
            # append last next state value to memory and process episode memory
            memory.memory.append((to_tensor(self.state, self.params.device), None, None, None,
                                  to_tensor(agent.act(self.state)[2], self.params.device), None))
            memory.process_memory()
            # train agent
            self.loss, self.loss_critic, self.loss_actor = agent.update(memory)

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
        self.params.episode += 1
        self.episode_rewards.append(self.episode_reward)
        self.plot.occasionally_plot(self.params.episode, self.episode_rewards)
        self.log.on_episode_finish(self.episode_reward, self.loss_critic, self.loss_actor, self.loss,
                                   agent.exploration_value, self.params.episode, self.log_steps, self.episode_rewards)
        intermediate_weight_save(self.params, self.log.log_dir, self.params.episode, agent)


def run_experiment(params, env, runner):
    """
    Initialize agents and perform experiment.

    author(s): Arnold Unterauer, Moritz T.
    :param params: params
    :param env: environment
    :param runner: Runner to perform the actual episodes
    """
    # setup agent
    if params.algorithm == "a2c":
        agent = A2C_Agent(params)
    elif params.algorithm == "ppo":
        agent = PPO_Agent(params)
    elif params.algorithm == "a2c_ppo":
        agent = A2C_PPO_Agent(params)
    else:
        agent = A2C_Agent(params)

    # load agent weights
    if params.load_weights:
        path = os.path.join(os.getcwd(), "logs", params.save_dir, "")
        agent.load_weights(path, params.weights_episode)

    # setup memory for experience
    memory = Memory(params)

    # initialize log on experiment start
    runner.log.on_experiment_start()

    # before the very first run the environment needs to be reset
    # afterwards it will be reset after done=True
    runner.reset()
    # run episodes
    while True:
        runner.run(agent, memory)
        if params.episode >= params.training_episodes:
            break

    # closing log on experiment end
    runner.log.on_experiment_finish(runner.episode_rewards, agent)

    # plot rewards
    if params.plot:
        runner.plot.plot_experiment(params.training_episodes, runner.episode_rewards)

    # shutdown environment
    env.close()


if __name__ == '__main__':
    main()
