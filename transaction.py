class Transaction:
    def __init__(self, observation, action, reward, observation_next):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.observation_next = observation_next