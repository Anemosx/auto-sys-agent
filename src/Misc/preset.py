def use_preset(params):
    """
    Load presets for stable algorithm run.

    author(s): Arnold Unterauer, Moritz T.
    :param params: params
    """
    if params.algorithm == "ppo":
        if params.env == "Pendulum-v1":
            params.hidden_units = [64, 64, 64]
            params.activation = "ReLU"
            params.learner_gamma = 0.95
            params.memory_gamma = 0.95
            params.actor_alpha = 0.0004
            params.critic_alpha = 0.004
            params.train_mode = "episode"
            params.memory_mode = "random"
            params.memory_capacity = 10000
            params.memory_batch_size = 2048
            params.horizon = 0
            params.reset_memory = False
            params.warmup = False
            params.warmup_epi = 0
            params.ppo_nr_train_epochs = 32
            params.ppo_clip = 0.2
            params.ppo_clip_grad = 0.5
            params.clip_critic = False
            params.use_entropy = True
            params.entropy_weight = 0.01
            params.entropy_weight_min = 0.01
            params.use_exploration = False
            params.advantage = "gae"
            params.gae_lambda = 0.95

        if params.env == "Crawler":
            params.hidden_units = [512, 512, 512]
            params.activation = "ReLU"
            params.learner_gamma = 0.995
            params.memory_gamma = 0.995
            params.actor_alpha = 0.0001
            params.critic_alpha = 0.0001
            params.train_mode = "max_steps"
            params.memory_mode = "random"
            params.memory_capacity = 25000
            params.memory_batch_size = 2048
            params.reset_memory = False
            params.horizon = 4096
            params.update_period = 1
            params.warmup = False
            params.warmup_epi = 0
            params.ppo_nr_train_epochs = 3
            params.ppo_clip = 0.2
            params.ppo_clip_grad = 0.5
            params.clip_critic = False
            params.ppo_weight_decay = False
            params.use_entropy = False
            params.use_exploration = False
            params.advantage = "gae"
            params.gae_lambda = 0.95

    if params.algorithm == "a2c":

        if params.env == "Pendulum-v1":
            params.hidden_units = [64, 64]
            params.activation = "Mish"
            params.learner_gamma = 0.9
            params.memory_gamma = 0.9
            params.actor_alpha = 0.0004
            params.critic_alpha = 0.004
            params.dist_std = 1.0
            params.dist_std_min = 1.0
            params.dist_std_decay = 0.0
            params.train_mode = "max_steps"
            params.use_exploration = False
            params.use_entropy = False
            params.entropy_weight = 0.2
            params.entropy_weight_min = 0.1
            params.entropy_decay = 0.004
            params.a2c_use_softplus = False
            params.a2c_clip_grad = 0.5
            params.a2c_discount_rewards = False
            params.a2c_learner_gamma_decay = False
            params.memory_mode = "max_steps"
            params.memory_capacity = 10000
            params.memory_batch_size = 32
            params.horizon = 0
            params.update_period = 1
            params.a2c_advantage_logp = True
            params.advantage = "td"
            params.united_loss_count = False

        if params.env == "Crawler":
            params.hidden_units = [128, 128]
            params.activation = "Mish"
            params.learner_gamma = 0.8
            params.memory_gamma = 0.8
            params.actor_alpha = 0.0004
            params.critic_alpha = 0.0004
            params.dist_std = 1.0
            params.dist_std_min = 0.5
            params.dist_std_decay = 0.02
            params.train_mode = "episode"
            params.use_exploration = False
            params.use_entropy = True
            params.entropy_weight = 0.2
            params.entropy_weight_min = 0.1
            params.entropy_decay = 0.004
            params.a2c_use_softplus = False
            params.a2c_clip_grad = 0.5
            params.a2c_discount_rewards = False
            params.a2c_learner_gamma_decay = False
            params.memory_mode = "episode"
            params.memory_capacity = 10000
            params.memory_batch_size = 128
            params.update_period = 5
            params.a2c_advantage_logp = True
            params.advantage = "td"
            params.united_loss_count = False

    if params.algorithm == "a2c_ppo":

        # if params.env == "Pendulum-v1":

        if params.env == "Crawler":
            params.hidden_units = [512, 512, 512]
            params.activation = "ReLU"
            params.learner_gamma = 0.995
            params.memory_gamma = 0.995
            params.actor_alpha = 0.0003
            params.critic_alpha = 0.0003
            params.train_mode = "episode"

            params.memory_mode = "random"
            params.memory_capacity = 2000000
            params.memory_batch_size = 2048
            params.horizon = 0

            params.update_period = 50

            params.use_entropy = False
            params.entropy_weight = 0.0001
            params.entropy_weight_min = 0.0001
            params.entropy_decay = 0.0001

            params.use_exploration = False

            params.advantage = "gae"
            params.gae_lambda = 0.95
            params.training_episodes = 100000

            params.load_weights = False
            params.load_params = False

    if params.algorithm == "a2c_mult_ag":

        # if params.env == "Pendulum-v1":

        if params.env == "Crawler":
            params.hidden_units = [512, 512, 512]
            params.activation = "Mish"
            params.learner_gamma = 0.995
            params.memory_gamma = 0.995
            params.actor_alpha = 0.0003
            params.critic_alpha = 0.0003
            params.train_mode = "episode"

            params.memory_mode = "random"
            params.memory_capacity = 2000000
            params.memory_batch_size = 2048
            params.horizon = 0

            params.update_period = 25

            params.nr_agents = 3
            params.a2c_mult_ag_shared_memory = False
            params.a2c_mult_ag_use_custom_master_update_frequency = False
            params.a2c_mult_ag_master_update_frequency = 1
            params.a2c_mult_ag_master_adaption_frequency = 100
            params.a2c_mult_ag_delayed_start_at = 0
            params.evolution_mode = True

            params.use_entropy = False
            params.entropy_weight = 0.0001
            params.entropy_weight_min = 0.0001
            params.entropy_decay = 0.0001

            params.use_exploration = False

            params.advantage = "gae"
            params.gae_lambda = 0.95
            params.training_episodes = 100000

            params.load_weights = False
            params.load_params = False

