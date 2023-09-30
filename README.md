# A2C and PPO on Crawler environment

![crawler_run](https://github.com/Anemosx/auto-sys-agent/blob/main/crawler.gif)

Autonomous Systems

### Quickstart
1. `conda create -n autonome_39 python=3.9` (analog für MultiEnv Support: `conda create -n autonome_36 python=3.6`)
2. `conda activate autonome_39`
3. `git clone git@github.com:Anemosx/auto-sys-agent.git`
4. `cd auto-sys-agent`
5. `pip install -r requirements.txt` (For Multiagent-Training autonom_36 env: `pip install -r requirements_36.txt`)
6. Add 'neptune_auth.json' und Informationen von https://neptune.ai
7. `python main.py`

### Quickstart - slurm
1. Connect to CIP via ssh
2. `conda create -n autonome_39 python=3.9`
3. `conda activate autonome_39`
4. `git clone git@github.com:Anemosx/auto-sys-agent.git`
5. `cd auto-sys-agent`
6. `chmod -R 755 envs`
7. `nano params.json`
   
    change `slurm_build` to `1`
    
   If you want to use neptune change `neptune_logger` to `1`
8. `pip install -r requirements.txt`
9. This step is only necessary if you want to use neptune
   If you want to use neptune follow the instructions under chapter "neptune.ai" below first
        `nano neptune_auth.json`
   
    insert neptune token into field instead of "..."
10. `cd..`
11. `nano job.sh`:

     #!/bin/bash
     cd auto-sys-agent
     python main.py

12. `chmod +x job.sh`
13. `sbatch --partition=All job.sh`

### Algorithms
- A2C: Equipped with an actor-critic network, we use an advantage function to enable good learning behavior. \
As of now the recommended agent for A2C and the Crawler environment is a2c_ppo or a2c_mult_ag.
- PPO: While using an actor-critic network, we make use of other tricks: \
With On-Policy Learning, Trust Region Methods and Clipped Surrogate Objectives we improve our learning and hope to achieve better results. \
https://arxiv.org/pdf/1707.06347.pdf

### Environments
The focus of our work is on the unity environment Crawler. \
This can be created by the user. \
Here, an agent with four continuously controlled legs must learn to walk and touch a cube.
Another supported environment is "Pendulum-v1". Note that in order to use this environment you must set the param "unity-env" to false.

### Presets
Presets provide you with predefined params that work well for selected environments.
E.g. if your algorithm is ppo and you want to use a preset for the "Crawler" environment set the "use-preset" param to "true".
Presets are existing for a2c and ppo. For both the "Crawler" and "Pendulum-v1" has an existing Preset. \
Additionally, there is a preset for a2c_ppo and a2c_mult_ag for the "Crawler" environment.

### Neptune.ai
Before using the Neptune logger, a json file "**neptune_auth.json**" must be created in the root directory. This must contain the following: \
`{
"project" : "autonomous-systems/autonomous-systems",
"api_token" : "INSERT PERSONAL API TOKEN HERE"
}`

### SLURM
For the experiments to run on the SLURM follow these instructions: \
https://github.com/Anemosx/auto-sys-agent/xxx

### Carbontracker
To analyze the energy consumption of our project, we decided to use the Carbontracker tool.
It can be downloaded directly from pip with `pip install carbontracker`. \
However, it requires some customization to make it work (see https://github.com/leondz/carbontracker/tree/no_del). \
Furthermore, the Carbontracker can only be used on certain systems.

### Multi Crawler
In order to run the crawler with multiple agents at the same time, we unfortunately need an older version, which can be downloaded below. \
Save the `Crawler_Linux_NoVis`, `Crawler_Windows_x86_64` or `Crawler.app` directory in `./envs/Crawler/` \
Also, this only works with `Python version 3.6`.

Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip \
Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip \
macOS: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip

### Parameters

|                 **Parameter**                  |                      **Functionality**                      |                                                                                                                                       **Information**                                                                                                                                        |                **Options**                 |
|:----------------------------------------------:|:-----------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------:|
|                  slurm_build                   |                     setup slurm params                      |                                                                                                                                              --                                                                                                                                              |                    0, 1                    |
|                 neptune_logger                 |                    enable neptune logger                    |                                                                                                                                              --                                                                                                                                              |                    0, 1                    |
|                 carbon_tracker                 |                    enable carbon tracker                    |                                                                                                                       enables energy tracking (see **carbontracker**)                                                                                                                        |                    0, 1                    |
|                      plot                      |                enable plots during training                 |                                                                                                                  use "plotting_frequency" and "plot_filter" to adjust plots                                                                                                                  |                    0, 1                    |
|                   local_logs                   |                      save data on disk                      |                                                                                                                              to save weights and params use _1_                                                                                                                              |                    0, 1                    |
|                 interim_saves                  |                        save weights                         |                                                                                                                saves weights for agents every "agent_save_frequency" episode                                                                                                                 |                    0, 1                    |
|                     render                     |                      enable rendering                       |                                                                                                         only working with gym environments, render every "render_frequency" episode                                                                                                          |                    0, 1                    |
|               printing_frequency               |                  define printing frequency                  |                                                                                                                             set printing frequency for new line                                                                                                                              |                     ℕ                      |
|                  print_decay                   | enable if current entropy_weight/dist_std should be printed |                                            Print the current entropy_weight and dist_std. This is useful when wanting to see the influence of a decay over time. Only works with a2c and dist_std printing is only useful if no softplus is used.                                            |                    0, 1                    |
|                     device                     |                      set torch device                       |                                                                                                                                              --                                                                                                                                              |           "auto", "cpu", "cuda"            |
|                   algorithm                    |                    set algorithm to use                     |                                                                                                                            set agent algorithm to train/evaluate                                                                                                                             |    "a2c", "ppo", "a2c_ppo", "a2c_mult_ag"  |
|                      env                       |                    set environment name                     |                                                                                                            put name of the env here e.g. _Crawler_, gym names are also supported                                                                                                             |               "Crawler", ...               |
|                   unity_env                    |                          unity env                          |                                                                                         if the environment is unity based use _true_, otherwise _false_ (loads env from ./envs/_env_/_env_version_)                                                                                          |                true, false                 |
|                    env_mode                    |                 different modes for crawler                 |                                                                                     to use crawler with one instance use _single_, for crawler with multiple agents use _multi_ (see **multi crawler**)                                                                                      |             "single", "multi"              |
|                  env_version                   |                 different unity env version                 |                                                                                                                 to render unity environment use _render_, otherwise _server_                                                                                                                 |     "render", "server", "linux-server"     |
|                   time_scale                   |         change how fast time passes in the simulation       |                                                           Defines the multiplier for the deltatime in the simulation. If set to a higher value, time will pass faster in the simulation but the physics may perform unpredictably                                                            |                     ℝ                      |
|                   use_preset                   |                     use working preset                      |                                                                                                                              loads preset params for algorithm                                                                                                                               |                true, false                 |
|                  hidden_units                  |                    define hidden layers                     |                                                                                                     define hidden layers of the agents (for both actor and critic) e.g. [512, 512, 512]                                                                                                      |                  [N, ..]                   |
|                   activation                   |                 use function as activation                  |                                                                                                                     use this function for the agent activation functions                                                                                                                     |       "ReLU", "ELU", "Mish", "Tanh"        |
|                 learner_gamma                  |                       discount factor                       |                                                                                                           set discount factor for reward and advantages (note use value below 1.0)                                                                                                           |                     ℝ                      |
|                  actor_alpha                   |                     actor learning rate                     |                                                                                                                                              --                                                                                                                                              |                     ℝ                      |
|                  critic_alpha                  |                    critic learning rate                     |                                                                                                                                              --                                                                                                                                              |                     ℝ                      |
|                    dist_std                    |                        set hard std                         |                                                                                      for a2c we use hard standard deviation, starting from "dist_std" decaying with "dist_std_decay" to "dist_std_min"                                                                                       |                     ℝ                      |
|                   train_mode                   |                      set training mode                      |                                                                          to update every episode use _episode_, every "horizon" steps use _max_steps_, every "horizon" steps or at episode finish _episode_horizon_                                                                          | "episode", "max_steps",  "episode_horizon" |
|                  memory_mode                   |                       set memory mode                       |                                                                          for random sampling use _random_, for episode sampling use _episode_ (note use "horizon"=0 and "memory_batch_size"=0 for complete episode)                                                                          |            "random", "episode"             |
|                memory_capacity                 |                     capacity of memory                      |                                                                                                                                              --                                                                                                                                              |                     ℕ                      |
|               memory_batch_size                |                 set batch size for training                 |                                                                                                                                              --                                                                                                                                              |                     ℕ                      |
|                  reset_memory                  |                reset memory after one update                |                                                                                                                                              --                                                                                                                                              |                true, false                 |
|                    horizon                     |              set number of steps until update               |                                                                                                                                              --                                                                                                                                              |                     ℕ                      |
|                 update_period                  |             perform update every update_period              |                                                                                                                                              --                                                                                                                                              |                     ℕ                      |
|                     warmup                     |                      use warm up phase                      |                                                                                                                           define warmup length using "warmup_epi"                                                                                                                            |                true, false                 |
|                a2c_use_softplus                |                    use softplus for a2c                     |                                                                                                                       use softplus function for a2c network to get std                                                                                                                       |                true, false                 |
|               a2c_advantage_logp               |   enable advantage weighting with log probability for a2c   |                                                                                                                                              --                                                                                                                                              |                true, false                 |
|                 a2c_clip_grad                  |                    clip a2c network grad                    |                                                                                                                                              --                                                                                                                                              |                     ℝ                      |
|              a2c_discount_rewards              |               use discounted rewards for a2c                |                                                                                                                                              --                                                                                                                                              |                true, false                 |
|                   nr_agents                    |            set number of agents for a2c_mult_ag             |                                                                                                                                              --                                                                                                                                              |                     ℕ                      |
|           a2c_mult_ag_shared_memory            |            enabled agents share the same memory             |                                                                                                                                              --                                                                                                                                              |                true, false                 |
| a2c_mult_ag_use_custom_master_update_frequency |      enable a custom update frequency with master_net       | enable a custom update frequency for agents. Note: if this is turned on each agent has its own separate net and adapts in the below stated frequency to the master net. Otherwise each agent uses the same net and updates the same net. Does not work together with train_mode "max_steps". |                true, false                 |
|      a2c_mult_ag_master_update_frequency       |      set the custom update frequency of the master_net      |                                                                       define how often the shared master net is updated. This is unrelated too how often the agent itself will be updated (for this use update period)                                                                       |                     ℕ                      |
|     a2c_mult_ag_master_adaption_frequency      |    set the synchronization frequency with the master_net    |                                                                                define in episodes how often all the agents will be synchronized with the master net or best agent (if evolution mode is true)                                                                                |                     ℕ                      |
|          a2c_mult_ag_delayed_start_at          |         start synchronizing at a specified episode          |                                                                                                   synchronizing with master/best agent is first started when a certain episode is reached                                                                                                    |                     ℕ                      |
|                 evolution_mode                 |                   turn on evolution mode                    |                                                                                    in evolution mode each agent has its own net and synchronizes at above stated frequency with the best performing agent                                                                                    |                true, false                 |
|              ppo_nr_train_epochs               |               number of ppo update iterations               |                                                                                                                      number of update iterations for the ppo algorithm                                                                                                                       |                     ℕ                      |
|                    ppo_clip                    |                     clip ratios of ppo                      |                                                                                                                        set clip range for ratios of the ppo algorithm                                                                                                                        |                     ℝ                      |
|                 ppo_clip_grad                  |                    clip ppo network grad                    |                                                                                                                                       clip ppo network                                                                                                                                       |                     ℝ                      |
|                  clip_critic                   |                      clip ppo network                       |                                                                                                                                   also clip critic network                                                                                                                                   |                true, false                 |
|                ppo_weight_decay                |               use weight decay for optimizers               |                                                                                                                                              --                                                                                                                                              |                true, false                 |
|                  use_entropy                   |                  enable entropy influence                   |                                                                                  for exploration use entropy term in algorithm, from "entropy_weight" decaying with "entropy_decay" to "entropy_weight_min"                                                                                  |                true, false                 |
|                use_exploration                 |                       for random runs                       |                                                                                                        to use random policy use "exploration_mode"=_"random"_ , otherwise depreciated                                                                                                        |                true, false                 |
|                   advantage                    |                     use advantage type                      |                                                                                     select the advantage to use: supported are _Generalized Advantage Estimation_, _Temporal Difference_ and _REINFORCE_                                                                                     |          "gae", "td", "reinforce"          |
|                   gae_lambda                   |                       lambda for GAE                        |                                                                                                                                              --                                                                                                                                              |                     ℝ                      |
|                    PENALTY                     |                    penalty on NaN reward                    |                                                                                                                       define penalty for agents encounting NaN rewards                                                                                                                       |                     ℝ                      |
|               training_episodes                |                 number of episodes to train                 |                                                                                                                                              --                                                                                                                                              |                     ℕ                      |
|                  load_weights                  |                   load pretrained weights                   |                                                                                                           load pretrained weights from "weights_dir" at episode "weights_episode"                                                                                                            |                   String                   |

![crawler](https://github.com/Anemosx/auto-sys-agent/blob/main/crawler_walking.png?raw=true)

### Contributions

**Arnold Unterauer**
- PPO
- agent
- agent models
- main: experiment from start to finish
- memory
- advantages
- logger
- plotter
- multi-environment
- connection of environment aka env utils
- different exploration approaches
- helper functions
- project, code structuring
- refactoring
- comments, documentation, readme
- code explanation
- research on a lot of things
- manual hyperparameter tuning
- intensive testing, training and experimentation
- parts of organisation and coordination

**Ludwig T.**
- Initialisierung für die Slurm-Umgebung
- Implementierung des Hyperparameter Tuning mit Optuna (wurde wegen des enormen Ressourcenverbrauchs und geringer Effizienz verworfen)
- Unity Environment Anpassung (SimpleCrawler)
- Testing
- manuelles Hyperparamter Tuning

**Manuele S.**
- manuelles Hyperparamter Tuning and testing
- Unity Environment Anpassung (SimpleCrawler)
- research of a2c
- Testing

**Moritz T.**
- Coding and research of a2c (a2c, a2c_ppo, a2c_mult_ag) and of many other code parts (see authors in code)
- code fundament at the start
- code maintenance and bug fixing
- experimentation with new agent and network paradigms
- agent
- agent models
- main: experiment from start to finish
- a2c_mult_ag_main: Run several a2c agents at once and implement new update approaches (master_net, evolutionary)
- memory
- different exploration approaches
- helper functions
- project, code structuring
- refactoring
- presets
- comments, documentation, readme
- manual hyperparameter tuning
- intensive testing, training and experimentation
- parts of organisation and coordination

**Matthias F.**
- Nachhaltigkeitsrechnung
- Implementierung Test Algorithmen (Haben es leider nicht in den "Haupt-Code" geschafft)
- Dokumentation
- Testing, manuelles Hyperparamter Tuning
- Research to Algorithms

**Sina S.**
- KickOff Präsentation
- Implementierung des Hyperparameter Tuning mit Optuna (wurde wegen des enormen Ressourcenverbrauchs und geringer Effizienz verworfen)
- Unity Environment Anpassung (Feste Start und Zielpunkte des "Simple Crawlers")
- Dokumentation
- Setup/Anpassung für MacOS
- Testing, manuelles Hyperparamter Tuning
