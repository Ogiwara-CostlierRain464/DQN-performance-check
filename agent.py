from table_brain import TableBrain
from nn_brain import NNBrain


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = TableBrain()

    def update_Q_function(self, transaction):
        pass

    def get_action(self, observation, step):
        pass


class Transaction:
    def __init__(self, observation, action, reward, observation_next):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.observation_next = observation_next
