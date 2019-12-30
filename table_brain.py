import abc
import numpy as np

from transition import Transition


class Brain:
    @abc.abstractmethod
    def update_q_function(self, transaction: Transition):
        pass

    @abc.abstractmethod
    def decide_action(self, observation, step):
        pass


class TableBrain(Brain):
    # 各状態の離散値への分割数
    num_digitized = 6
    # 学習係数
    eta = 0.5
    # 時間割引率
    gamma = 0.99

    def __init__(self, num_states: int, num_actions: int):
        self.num_actions = num_actions
        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(self.num_digitized ** num_states, num_actions)
        )

    @staticmethod
    def __bins(clip_min, clip_max, num):
        """観測した状態(連続値)を離散値にデジタル変換する閾値を求める"""
        # example
        # lin_space(0,1,3) => [0, 0.5, 1]
        # lin_space(0,1,3)[0:2] => [0, 0.5]
        # so, [0:2] is same as [0,2)
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def __digitize_state(self, observation: np.ndarray) -> float:
        """観測したobservation状態を、離散値に変換する"""
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.__bins(-2.4, 2.4, self.num_digitized)),
            np.digitize(cart_v, bins=self.__bins(-3.0, 3.0, self.num_digitized)),
            np.digitize(pole_angle, bins=self.__bins(-0.5, 0.5, self.num_digitized)),
            np.digitize(pole_v, bins=self.__bins(-2.0, 2.0, self.num_digitized)),
        ]
        return sum([x * (self.num_digitized ** i) for i, x in enumerate(digitized)])

    def update_q_function(self, trn: Transition) -> None:
        state = self.__digitize_state(trn.state)
        state_next = self.__digitize_state(trn.next_state)
        max_q_next = max(self.q_table[state_next][:])
        self.q_table[state, trn.action] = \
            self.q_table[state, trn.action] + self.eta * (
                        trn.reward + self.gamma * max_q_next - self.q_table[state, trn.action])

    def decide_action(self, observation: np.ndarray, episode: int) -> int:
        """ε-greedy法で徐々に最適行動のみを採用する"""
        state = self.__digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # example: np.argmax([0,2,1]) => 1
            # argmax returns the indices of the maximum values.
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
        return action
