from typing import List, Tuple, Union

from numpy import ndarray
from numpy import zeros as n_zeros
from torch import Tensor, cat, tensor
from torch import float32 as t_float32
from numpy import float32 as n_float32
from numpy.random import choice as n_choice

from classes.Globals import Globals


class ReplayBuffer:
    __CAPACITY: int
    __ALPHA: float

    __priorities: ndarray
    __buffer: List[Union[None, Tuple[Tensor, Union[None, int], float, Tensor, bool]]]
    __index: int

    def __init__(self, capacity: int, alpha: float):
        self.__CAPACITY = capacity
        self.__ALPHA = alpha

        self.__buffer = []
        self.__index = 0
        self.__priorities = n_zeros((capacity,), dtype=n_float32)

    def __len__(self) -> int:
        return len(self.__buffer)

    def update_priorities(self, indices, priorities) -> None:
        for index, priority in zip(indices, priorities):
            self.__priorities[index] = priority

    def __store(self, memory: Tuple[Tensor, Union[None, int], float, Tensor, bool]) -> None:
        if self.__len__() < self.__CAPACITY:
            self.__buffer = [*self.__buffer, memory]
            return
        self.__buffer[self.__index] = memory

    def store_memory(self, state: Tensor, action: Union[int, None], reward: float, next_state: Tensor,
                     is_done: bool) -> None:
        if self.__len__() == 100:
            b = True
        # if buffer is not empty get max priority
        maximum_priority: n_float32 = self.__priorities.max() if self.__buffer else n_float32(1.0)
        # store memory and corresponding priority

        self.__store(memory=(state, action, reward, next_state, is_done))
        self.__priorities[self.__index] = maximum_priority
        # update index
        self.__index = (self.__index + 1) % self.__CAPACITY

    def get_sample(self, amount_of_memories: int, beta: float):
        # compute probabilities
        priorities: ndarray = self.__priorities if self.__len__() == self.__CAPACITY else \
            self.__priorities[:self.__index]
        probabilities: ndarray = (priorities ** self.__ALPHA) / (priorities ** self.__ALPHA).sum()

        # retrieve samples from buffer
        memories_indices: ndarray = n_choice(self.__len__(), amount_of_memories, p=probabilities)
        memories_sample: List = [self.__buffer[idx] for idx in memories_indices]

        # compute weights
        weights: Union[ndarray, Tensor] = (self.__len__() * probabilities[memories_indices] ** (-beta))
        weights /= weights.max()
        weights = tensor(weights, requires_grad=False, dtype=t_float32).to(Globals.DEVICE_TYPE)

        batch = list(zip(*memories_sample))
        states = cat(batch[0]).to(Globals.DEVICE_TYPE)
        actions = tensor(batch[1], requires_grad=False).to(Globals.DEVICE_TYPE)
        rewards = tensor(batch[2], requires_grad=False, dtype=t_float32).to(Globals.DEVICE_TYPE)
        next_states = cat(batch[3]).to(Globals.DEVICE_TYPE)
        # data type as float to convert into numbers
        terminals = tensor(batch[4], requires_grad=False, dtype=t_float32).to(Globals.DEVICE_TYPE)

        return states, actions, rewards, next_states, terminals, memories_indices, weights
