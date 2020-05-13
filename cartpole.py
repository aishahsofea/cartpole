import gym
import random
import numpy as np
from numpy import savetxt

BUCKETS = (1, 1, 6, 12)
MAX_STEPS = 200
MAX_EXP_RATE = 1
MIN_EXP_RATE = 0.01
EXP_DECAY_RATE = 0.001


class CartPoleEnvironment():

    def __init__(self, buckets=(1, 1, 6, 12,)):
        self.env = gym.make('CartPole-v1')
        self.buckets = buckets

    def discretize(self, obs):
        """
        Convert continuous observation space into discrete values 
        """
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        upper_bounds = [high[0], high[1] / 1e38, high[2], high[3] / 1e38]
        lower_bounds = [low[0], low[1] / 1e38, low[2], low[3] / 1e38]

        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]

        return tuple(new_obs)


class CartPoleAgent():

    def __init__(self, alpha=0.5, epsilon=1, episodes=5_000):
        self.em = CartPoleEnvironment()
        self.q_table = np.zeros(self.em.buckets + (self.em.env.action_space.n,))
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes

    def update_q_value(self, state, action, reward, new_state):
        """
        Using the Bellman equation, update the Q-value based on state-action pair
        Q(s, a) <- Q(s, a) + alpha(curr_reward + gamma * max(Q(s', a')) - Q(s, a))
        
        where max(Q(s', a') is the best future reward, and gamma = 1
        """
        prev_q = self.q_table[state][action]
        future_reward = self.best_future_reward(new_state)

        self.q_table[state][action] = prev_q + self.alpha * (reward + future_reward - prev_q)

    def best_future_reward(self, state):
        return np.max(self.q_table[state])

    def choose_action(self, state):
        """
        Action is chosen using epsilon-greedy algorithm
        """
        if random.random() > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            random_action = self.em.env.action_space.sample()
            return random_action

    def train(self):
        """
        Train for 10,000 episodes where at each episode the exploration is decayed.
        """
        rewards = []

        for ep in range(self.episodes):
            env = self.em.env
            state = self.em.discretize(env.reset())
            self.epsilon = MIN_EXP_RATE + (MAX_EXP_RATE - MIN_EXP_RATE) * np.exp(-EXP_DECAY_RATE * ep)
            done = False
            episode_rewards = 0

            for step in range(MAX_STEPS):
                action = self.choose_action(state)

                # take action
                new_state, reward, done, _ = env.step(action)
                new_state = self.em.discretize(new_state)
                episode_rewards += reward

                # update Q-table 
                self.update_q_value(state, action, reward, new_state)

                # transition to the new state
                state = new_state

                if done: 
                    print(f"Episode {ep} finished after {step + 1} timesteps.")
                    break
            
            rewards.append(episode_rewards)
        
        rewards_per_thousand_ep = np.split(np.array(rewards), self.episodes/500)
        print(rewards_per_thousand_ep)
        count = 500

        for r in rewards_per_thousand_ep:
            print(f"{count}: {int(sum(r/500))}")
            count += 500


if __name__ == "__main__":
    agent = CartPoleAgent()
    agent.train()