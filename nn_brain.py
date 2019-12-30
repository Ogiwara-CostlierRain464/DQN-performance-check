import random
from torch import nn
from torch import optim
from typing import List

from table_brain import Brain
from transition import Transition


class NNBrain(Brain):
    capacity = 10000
    batch_size = 32

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(self.capacity)

        self.model = nn.Sequential()
        # 4in -> 32out
        self.model.add_module("fc1", nn.Linear(num_states, 32))
        self.model.add_module("relu1", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(32, 32))
        self.model.add_module("relu2", nn.ReLU())
        self.model.add_module("fc3", nn.Linear(32, num_actions))

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001
        )

    def __replay(self):
        if len(self.memory) < self.batch_size:
            return

        trns = self.memory.sample(self.batch_size)

    def update_q_function(self, transition: Transition):
        pass

    def decide_action(self, observation, step):
        pass


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, trn: Transition):
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = trn
        # self.index+1がself.capacityをオーバーしたらindexを0に初期化
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
