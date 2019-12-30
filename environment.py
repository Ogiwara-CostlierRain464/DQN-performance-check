import gym
from transition import Transition
from agent import Agent


class Environment:

    # 最大試行回数
    # この試行回数以内にゲームの攻略を目指す。
    num_episodes = 1000

    # 1試行のstep数
    max_steps = 200

    def __init__(self):
        self.env = gym.make("CartPole-v0")
        # カート位置、カート速度、棒の角度、棒の角速度の四つ
        num_states = self.env.observation_space.shape[0]
        # 左右の二つ
        num_actions = self.env.action_space.n

        self.agent = Agent(num_states, num_actions)

    def run(self):
        # 195step以上連続で立ち続けた試行数
        complete_episodes = 0
        is_episode_finished = False

        # 1episodeはゴールするまでの期間、
        # 1stepは現在の状況に応じ次の選択をする期間、
        # を表す
        for episode in range(self.num_episodes):
            state = self.env.reset()

            for step in range(self.max_steps):
                # 行動を求める
                action = self.agent.get_action(state, episode)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                next_state, _, done, _ = self.env.step(action)

                # 報酬を与える
                if done:
                    if step < 195:
                        reward = -1
                        complete_episodes = 0
                    else:
                        reward = 1
                        complete_episodes += 1
                else:
                    # 途中の報酬は0
                    reward = 0

                # step+1の状態observation_nextを用いて、Q関数を更新する
                self.agent.update_q_function(Transition(state, action, next_state, reward))

                state = next_state

                if done:
                    print("{0} episode finished after {1} time steps".format(episode, step + 1))
                    break

            if is_episode_finished:
                break

            if complete_episodes >= 10:
                print("10回連続成功")
                is_episode_finished = True
