import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


def mish(input):
    """
    Mish function

    author(s): Moritz T.
    :param input: input value coming from e.g. a neural network
    """
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    """
    Mish Class as activation for neural networks.

    author(s): Moritz T.
    """

    def __init__(self): super().__init__()

    def forward(self, input): return mish(input)


def init_model(params, module):
    """
    Setup neural network for agents.

    author(s): Arnold Unterauer, Moritz T.
    :param module: type of network (actor or critic)
    :type module: String
    :param params: params
    :return: network
    """
    layers = collections.OrderedDict()
    for i in range(len(params.hidden_units)):
        if i == 0:
            layers['0'] = nn.Linear(params.observation_shape, params.hidden_units[i])
        else:
            layers['{}'.format(i * 2)] = nn.Linear(params.hidden_units[i - 1], params.hidden_units[i])

        if params.activation == "Tanh":
            layers['{}'.format(i * 2 + 1)] = nn.Tanh()
        elif params.activation == "Mish":
            if params.env_mode == "multi":
                # in torch.nn for python 3.6 Mish is not yet available
                layers['{}'.format(i * 2 + 1)] = Mish()
            else:
                layers['{}'.format(i * 2 + 1)] = nn.Mish(True)
        elif params.activation == "ReLU":
            layers['{}'.format(i * 2 + 1)] = nn.ReLU()
        elif params.activation == "ELU":
            layers['{}'.format(i * 2 + 1)] = nn.ELU()
        else:
            layers['{}'.format(i * 2 + 1)] = nn.ReLU()
    model = nn.Sequential(layers).to(params.device)

    if module == "actor":
        action_head = nn.Linear(params.hidden_units[len(params.hidden_units) - 1], params.nr_actions).to(params.device)
        std_head = nn.Linear(params.hidden_units[len(params.hidden_units) - 1], params.nr_actions).to(params.device)
        return model, action_head, std_head
    if module == "critic":
        value_head = nn.Linear(params.hidden_units[len(params.hidden_units) - 1], 1).to(params.device)
        return model, value_head
    return model


class A2CActor(nn.Module):
    """
    Network for A2C Actor.

    author(s): Moritz T.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model, self.action_head, self.stds_head = init_model(params, "actor")
        self.stds = nn.Parameter(torch.full((self.params.nr_actions,), params.dist_std)).to(params.device)
        self.std_min = params.dist_std_min
        self.std_decay = params.dist_std_decay
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Evaluate observation and determine action, std.

        author(s): Moritz T.
        :param x: state / observation
        :type x: tensor([ ])
        :return: predicted action, standard deviation
        """
        x = self.model(x)
        means = self.action_head(x)

        if self.params.a2c_use_softplus:
            stds = F.softplus(self.stds_head(x))
        else:
            stds = self.stds

        # if stds gets to small an error gets raised
        stds = stds + 1e-10

        return self.tanh(means), stds

    def update_std(self):
        """
        Update standard deviation.

        author(s): Moritz T.
        """
        self.stds = nn.Parameter(torch.full((self.params.nr_actions,), self.params.dist_std)).to(self.params.device)


class A2CCritic(nn.Module):
    """
    Network for A2C Critic.

    author(s): Moritz T.
    """

    def __init__(self, params):
        super().__init__()
        self.model, self.value_head = init_model(params, "critic")

    def forward(self, x):
        """
        Evaluate observation to get state value.

        author(s): Moritz T.
        :param x: state / observation
        :type x: tensor([ ])
        :return: state value
        """
        x = self.model(x)
        return self.value_head(x)


class PPOActor(nn.Module):
    """
    Network for PPO Actor.

    author(s): Arnold Unterauer
    """

    def __init__(self, params):
        super(PPOActor, self).__init__()
        self.model, self.action_head, self.std_head = init_model(params, "actor")
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Evaluate observation and determine action, std.

        author(s): Arnold Unterauer
        :param x: state / observation
        :type x: tensor([ ])
        :return: predicted action, standard deviation
        """
        x = self.model(x)
        action = self.action_head(x)
        std = self.std_head(x)
        return self.tanh(action), self.softplus(std) + 1e-10


class PPOCritic(nn.Module):
    """
    Network for PPO Critic.

    author(s): Arnold Unterauer
    """

    def __init__(self, params):
        super(PPOCritic, self).__init__()
        self.model, self.value_head = init_model(params, "critic")

    def forward(self, x):
        """
        Evaluate observation to get state value.

        author(s): Arnold Unterauer
        :param x: state / observation
        :type x: tensor([ ])
        :return: state value
        """
        x = self.model(x)
        return self.value_head(x)
