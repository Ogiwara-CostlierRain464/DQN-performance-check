from table_brain import TableBrain

from nn_brain import NNBrain
from transaction import Transaction


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = TableBrain(num_states, num_actions)

    def update_q_function(self, trn: Transaction):
        self.brain.update_q_function(trn)

    def get_action(self, observation, episode):
        """行動の決定"""
        return self.brain.decide_action(observation, episode)
