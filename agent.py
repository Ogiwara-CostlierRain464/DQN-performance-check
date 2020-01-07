import numpy as np
from typing import Union
from table_brain import TableBrain
from transition import Transition
from nn_brain import NNBrain


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = TableBrain(num_states, num_actions)
        # self.brain = NNBrain(num_states, num_actions)

    def update_q_function(self, trn) -> None:
        self.brain.update_q_function(trn)

    def get_action(self, state: np.ndarray, episode) -> Union[int, float, bool]:
        """行動の決定"""
        return self.brain.decide_action(state, episode)
