import numpy as np


class Transaction:
    def __init__(self, observation: np.ndarray, action, reward: float, observation_next: np.ndarray):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.observation_next = observation_next
