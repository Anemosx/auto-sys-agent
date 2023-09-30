from datetime import datetime
import os
import json
import numpy as np
from dotmap import DotMap
from carbontracker.tracker import CarbonTracker
import neptune.new as neptune
import time


class Logger:
    """
    Logger to store metadata and output status updates.
    Logs available in Neptune: https://neptune.ai/.

    author(s): Arnold Unterauer
    """
    def __init__(self, params, params_json) -> None:
        self.params = params
        self.params_json = params_json
        self.exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
        self.log_dir = None
        self.carbontracker = None
        self.nep_logger = None
        self.exp_start_time = None
        self.epi_start_time = None
        self.setup_local_logs()
        self.setup_carbontracker()
        self.setup_neptune()

    def setup_local_logs(self):
        """
        Setup directory for storing weights and param files.

        author(s): Arnold Unterauer
        """
        if self.params.local_logs:
            self.log_dir = os.path.join(os.getcwd(), 'logs', '{}'.format(self.exp_time))
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            with open(os.path.join(self.log_dir, 'params.json'), 'w') as outfile:
                json.dump(self.params_json, outfile)
            with open(os.path.join(self.log_dir, 'params.txt'), 'w') as outfile:
                json.dump(self.params_json, outfile)

    def setup_carbontracker(self):
        """
        Initialize carbon tracker to track energy consumption.
        Use https://github.com/leondz/carbontracker/tree/no_del to prevent errors.

        author(s): Arnold Unterauer
        """
        if self.params.carbon_tracker:
            self.carbontracker = CarbonTracker(epochs=self.params.training_episodes,
                                               monitor_epochs=self.params.training_episodes,
                                               update_interval=1,
                                               devices_by_pid=True,
                                               verbose=0)

    def setup_neptune(self):
        """
        Initialize Neptune Logger from neptune_auth.json
        https://neptune.ai/

        author(s): Arnold Unterauer
        """
        if self.params.neptune_logger:
            self.params.exp_time = self.exp_time
            with open('neptune_auth.json', 'r') as f:
                neptune_auth = json.load(f)
                neptune_auth = DotMap(neptune_auth)
            self.nep_logger = neptune.init(
                project=neptune_auth.project,
                api_token=neptune_auth.api_token,
                tags=[self.params.algorithm, self.params.env],
                source_files=["params.json"])
            self.nep_logger["parameters"] = self.params

    def on_experiment_finish(self, returns, agent):
        """
        Stops Logger, print status and saves weights of agents.

        author(s): Arnold Unterauer
        :param returns: undiscounted rewards from all episodes
        :param agent: learning agent (a2c or ppo)
        """
        # stop logger, tracker
        if self.params.nep_logger:
            self.nep_logger.stop()
        if self.params.carbon_tracker:
            self.carbontracker.stop()

        exp_end_time = datetime.now()
        print("\nTraining completed at {}".format(exp_end_time.strftime('%Y-%m-%d %H:%M:%S')))
        time_dur = exp_end_time - self.exp_start_time
        time_dur_parsed = "{}h {}m {}s".format(time_dur.days * 24 + time_dur.seconds // 3600,
                                                     (time_dur.seconds % 3600) // 60, time_dur.seconds % 60)
        print("Training Duration: ", time_dur_parsed)

        # Write Stuff
        if self.params.local_logs:
            agent.save_weights(os.path.join(self.log_dir, ""), self.params.episode)
            with open(os.path.join(self.log_dir, 'params.json'), 'r') as outfile:
                params_json = json.load(outfile)
                # will be logged in neptune anyway
                params_json["trainingDuration"] = time_dur_parsed
                params_json["returnOfLastEpisode"] = returns[-1]
            with open(os.path.join(self.log_dir, 'params.json'), 'w') as outfile:
                json.dump(params_json, outfile)
            with open(os.path.join(self.log_dir, 'params.txt'), 'w') as outfile:
                json.dump(params_json, outfile)

    def on_experiment_start(self):
        """
        Use timestamp for unique identifier of the experiment.

        author(s): Arnold Unterauer
        """
        self.exp_start_time = datetime.now()
        print("\nTraining started at {}\n".format(self.exp_start_time.strftime('%Y-%m-%d %H:%M:%S')))

    def on_episode_start(self):
        """
        Tracks energy and time consumption.

        author(s): Arnold Unterauer
        """
        if self.carbontracker:
            self.carbontracker.epoch_start()
        self.epi_start_time = time.time()

    def on_episode_finish(self, undiscounted_return, loss_critic, loss_actor, loss, exploration_value, nr_episode, time_step, episode_rewards, agent=None):
        """
        Logs all metadata from one episode in neptune and displays status output.

        author(s): Arnold Unterauer, Moritz T.
        :param undiscounted_return: undiscounted reward of the episode
        :param loss_critic: loss of critic
        :param loss_actor: loss of actor
        :param loss: total loss
        :param exploration_value: current exploration value
        :param nr_episode: number of the current episode
        :param time_step: number of steps taken in the environment
        :param episode_rewards: undiscounted reward of all episodes
        :type episode_rewards: list
        """

        # prints the values of entropy and action variance over time --> Thus making the decay over time visible
        if self.params.print_decay:
            print("Entropy Weight: ", self.params.entropy_weight)
            print("Action Variance: ", self.params.dist_std)

        if self.params.algorithm == "a2c_mult_ag":
            print(f"Agent {agent.id}: ")

        print("\repisode:{:>5} |\t mean_return: {:.2f} |\t undis_return: {:.1f} | loss: {:.3f} ".format(nr_episode, np.mean(episode_rewards[-100:]), undiscounted_return, loss), end="")
        if nr_episode > 0 and nr_episode % self.params.printing_frequency == 0:
            print("\repisode:{:>5} |\t mean_return: {:.2f} |\t undis_return: {:.1f} | loss: {:.3f} ".format(nr_episode, np.mean(episode_rewards[-100:]), undiscounted_return, loss))
        epi_end_time = time.time()
        time_epoch = round(epi_end_time - self.epi_start_time, 2)
        if self.carbontracker:
            # use https://github.com/leondz/carbontracker/tree/no_del
            self.carbontracker.epoch_end(True)
        if self.nep_logger:
            self.nep_logger["return"].log(undiscounted_return)
            self.nep_logger["time/s"].log(time_epoch)
            self.nep_logger["time_step"].log(time_step)
            self.nep_logger["loss_critic"].log(loss_critic)
            self.nep_logger["loss_actor"].log(loss_actor)
            self.nep_logger["loss"].log(loss)
            if self.params.use_exploration:
                self.nep_logger["explore_v"].log(exploration_value)
            if self.carbontracker:
                self.nep_logger["energy/kWh"].log(self.carbontracker.tracker.total_energy_per_epoch()[nr_episode - 1])
