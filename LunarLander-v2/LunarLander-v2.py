__author__ = 'witwolf'

import gym

ENV = 'LunarLander-v2'
EPISODE = 100000
STEP = 300

from dqn import DQN


def main():
    env = gym.make(ENV)
    agent = DQN(env.observation_space.shape[0], env.action_space.n, logdir='/data/log/LunarLander-v2')

    for episode in xrange(EPISODE):
        state = env.reset()

        for step in xrange(STEP):
            env.render()
            action = agent.egreedy_action(state)
            next_state, reward, terminate, _ = env.step(action)
            agent.observe_action(state, action, reward, next_state, terminate)
            state = next_state

            if terminate:
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(5):
                state = env.reset()
                for j in xrange(STEP):
                    env.render()
                    action = agent.action(state)
                    state, reward, terminate, _ = env.step(action)
                    total_reward += reward
                    if terminate:
                        break
            agent.summary(episode, total_reward / 5)


if __name__ == '__main__':
    main()
