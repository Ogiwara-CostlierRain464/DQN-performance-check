import random
import torch.nn.functional as F
import torch
from torch import nn
from torch import optim
from typing import List
import numpy as np

from table_brain import Brain
from transition import Transition


class NNBrain(Brain):
    capacity = 10000
    batch_size = 32
    # 時間割引率
    gamma = 0.99

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

        # List[Transition]
        trns = self.memory.sample(self.batch_size)
        # Transition(List[int], List[int], List[int], List[int])
        batch = Transition(*zip(*trns))

        # (FloatTensor[1x4])xBATCH_SIZE
        # => FloatTensor[BATCH_SIZEx4]
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None
        ])

        self.model.eval()
        state_action_values = self.model(state_batch).gather(1, action_batch)
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state))
        )

        next_state_values = torch.zeros(self.batch_size)

        next_state_values[non_final_mask] = self.model(
            non_final_next_states
        ).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.gamma * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_q_function(self, transition: Transition):
        # 変換が必要

        state = torch\
            .from_numpy(transition.state)\
            .type(torch.FloatTensor)\
            .unsqueeze(0)

        action = transition.action

        next_state = torch\
            .from_numpy(transition.next_state)\
            .type(torch.FloatTensor)\
            .unsqueeze(0)

        reward = torch.FloatTensor([transition.reward])

        trn = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward
        )

        self.memory.push(trn)
        self.__replay()

    def decide_action(self, observation, episode):
        # 変換が必要
        observation = torch\
            .from_numpy(observation)\
            .type(torch.FloatTensor)\
            .unsqueeze(0)

        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()

            with torch.no_grad():
                action = self.model(observation).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action.item()


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
