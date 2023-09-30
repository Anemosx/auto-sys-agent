import random
import numpy as np
import torch

from src.Misc.util import to_tensor


class Memory:
    """
    Memory, experience storage.

    author(s): Arnold Unterauer, Moritz T.
    """
    def __init__(self, params):
        self.memory = []
        self.processed_memory = []
        self.params = params
        self.capacity = params.memory_capacity
        self.gamma = params.memory_gamma

    def clear(self):
        """
        Resets memory.

        author(s): Arnold Unterauer, Moritz T.
        """
        self.memory = []
        self.processed_memory = []

    def add(self, element: tuple):
        """
        Add experience to memory. For easier processing we add them as tensors.

        author(s): Arnold Unterauer, Moritz T.
        :param element: experience tuple
        :type element: tuple in form (state, action, reward, log_p, value, done)
        """
        state = to_tensor(element[0], self.params.device)
        action = to_tensor(element[1], self.params.device)
        reward = to_tensor(element[2], self.params.device)
        log_p = to_tensor(element[3], self.params.device)
        value = to_tensor(element[4], self.params.device)
        inverse_done = 1 - np.array([1 if t else 0 for t in element[5]])
        inverse_done = to_tensor(inverse_done, self.params.device)

        self.memory.append((state, action, reward, log_p, value, inverse_done))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample_last(self):
        """
        Sample the last n entries of the memory.

        author(s): Arnold Unterauer, Moritz T.
        :return: last experience batch
        """
        if 0 < self.params.memory_batch_size < len(self.processed_memory[0]) and not self.params.train_mode == "episode":
            indices = list(range(len(self.processed_memory[0]) - self.params.memory_batch_size, len(self.processed_memory[0])))
        else:
            indices = list(range(0, len(self.processed_memory[0])))
        return self.create_batch(indices)

    def sample_random(self):
        """
        Sample random n entries of the memory.

        author(s): Arnold Unterauer
        :return: random experience batch
        """
        sample_number = min(len(self.processed_memory[0]), self.params.memory_batch_size)
        indices = random.sample(range(0, len(self.processed_memory[0])), sample_number)
        return self.create_batch(indices)

    def process_memory(self):
        """
        Process accumulated experience memory of episode.
        Calculates discounted_returns and advantages (TD, GAE or REINFORCE) for the current memory.
        Stores processed_memory in form (states, actions, log_ps, discounted_returns, advantages)

        author(s): Arnold Unterauer, Moritz T.
        """
        prepare_memory = []
        advantage = to_tensor(np.zeros((len(self.memory[0][0]), 1)), self.params.device)
        discounted_return = to_tensor(np.zeros((len(self.memory[0][0]))), self.params.device)
        for i in reversed(range(len(self.memory) - 1)):
            # unpack entry from memory
            state, action, reward, log_p, value, inverse_done = self.memory[i]
            next_value = self.memory[i+1][-2]

            # calculate discounted return
            discounted_return = reward + self.params.learner_gamma * discounted_return * inverse_done

            # calculate advantage
            if self.params.advantage == "reinforce":
                # calculate reinforce advantage
                advantage = discounted_return.unsqueeze(1)
            if self.params.advantage == "td":
                # calculate td advantage
                target_value = reward + self.params.learner_gamma * inverse_done * next_value
                advantage = (target_value - value).unsqueeze(1)
            if self.params.advantage == "gae":
                # calculate gae advantage
                delta = reward + self.params.learner_gamma * inverse_done * next_value - value
                advantage = advantage * self.params.gae_lambda * self.params.learner_gamma * inverse_done[:, None] + delta[:, None]

            prepare_memory.append((state, action, log_p, discounted_return, advantage))

        prepare_memory.reverse()
        # unpack processed memory and normalize advantages
        states, actions, log_ps, discounted_returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*prepare_memory))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        discounted_returns = discounted_returns.unsqueeze(1)

        if len(self.processed_memory) > 0:
            states = torch.cat((self.processed_memory[0], states), 0)
            actions = torch.cat((self.processed_memory[1], actions), 0)
            log_ps = torch.cat((self.processed_memory[2], log_ps), 0)
            discounted_returns = torch.cat((self.processed_memory[3], discounted_returns), 0)
            advantages = torch.cat((self.processed_memory[4], advantages), 0)

            if len(states) > self.capacity:
                del_amount = len(states) - self.capacity
                states = states[del_amount:]
                actions = actions[del_amount:]
                log_ps = log_ps[del_amount:]
                discounted_returns = discounted_returns[del_amount:]
                advantages = advantages[del_amount:]

        self.memory = []
        self.processed_memory = [states, actions, log_ps, discounted_returns, advantages]

    def create_batch(self, indices):
        """
        Creates batch from processed_memory using indices list.

        author(s): Arnold Unterauer, Moritz T.
        :param indices: list of indices for memory sampling
        :return: states, actions, log_ps, discounted_returns, advantages
        :rtype: 5 x [tensor([[], ..., []])]
        """
        states = self.processed_memory[0][indices]
        actions = self.processed_memory[1][indices]
        log_ps = self.processed_memory[2][indices]
        discounted_returns = self.processed_memory[3][indices]
        advantages = self.processed_memory[4][indices]

        return states, actions, log_ps, discounted_returns, advantages
