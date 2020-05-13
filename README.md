<h3>Motivation</h3>

I recently finished the [CS50 AI course](https://cs50.harvard.edu/ai/2020/) by Harvard. If you are interested in learning modern AI concepts and looking to do hands-on projects, this course is for you. All you need is basic math and programming knowledge. Also, did I mention that it is completely free? Anyway, in week 4, we were introduced to different types of learning in Machine Learning; supervised learning, unsupervised learning, reinforcement learning along with commonplace algorithms like SVM, KNN-clustering and K-means.

What caught my attention the most was the RL algorithm; Q-learning. Unlike most other algorithms, where we need to prepare the data before training, Q-learning(or just RL in general) collects the data while training, sort of. For the project assignment, we need to implement [Nim](https://en.wikipedia.org/wiki/Nim). Our agent is trained by playing against itself for 10,000 times prior to playing against a human. I would say the outcome was impressive, I mean, I lost 100% of the time. Anyhow, I wanted to reinforce(no pun intended) my understanding and implemented it for a different environment.

<h3>CartPole Problem</h3>

Luckily for us, [Open AI Gym](https://gym.openai.com/) provides a number of environments we can choose from. The most popular one is --_wait for it_-- the CartPole, so I decided to go with that. Refer to [this wiki](https://github.com/openai/gym/wiki/CartPole-v0) for the problem details.

> It is considered solved when reward is greater than or equal to 195 over 100 consecutive trials.

<h5>Challenges</h5>

Data collected during training is stored in Q-table. For problems with finite states like Nim, storing state-action pairs with their respective rewards is not an issue. However, for our cartpole environment, the states are continuous. To get a better idea, below are the minimum and maximum values for each variable.

Maximum values:
`[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]`

Minimum values:
`[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]`

Imagine all the possible numbers between the max and min values, it is simply impossible to evaluate reward at each distinct state. For this reason, we have to descretize the values into buckets. Code to discretize state space is inspired by [sanjitjain2](https://github.com/sanjitjain2/q-learning-for-cartpole/blob/master/qlearning.py) with some minor tweak.

```
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
```

<h5>Training</h5>

Our agent is trained for 5,000 episodes.

For each episode:

- CartPole environment is initialized.
- Initial state is extracted from the environment.
- Exploration rate is decayed, since we want to explore less and exploit more over time.
- Agent can train for a maximum of 200 timesteps.

At each timestep:

- Using epsilon-greedy algorithm, select an action.
- Passing the selected action to the gym's `step()` function, we can get the `new_state`, `reward` and `done`. `done` is true if the pole is no longer upright.
- Update our Q-table using Bellman equation.
- If the pole is no longer upright, break out of the loop and start a new episode.

<h5>Evaluation</h5>

For every 500 episodes, I average out the total rewards.

```
500 : 30
1000: 54
1500: 82
2000: 110
2500: 127
3000: 145
3500: 158
4000: 183
4500: 175
5000: 196
```

After 5000 episodes of training, the average rewards is starting to look good. This means that, on average (episode 4501-5000), the pole was upright up to 196 timesteps. In fact, for the last 300 episodes or so, the pole was upright for 200 timesteps. This proves that our agent indeed learns over time.

**P/S**: Please check out [deeplizard](https://deeplizard.com/learn/video/HGeI30uATws) for Q-learning implementation with Gym. They also have awesome tutorials on topics like Deep Learning, Neural Networks and how to put the knowledge together using tools like Keras and Pytorch.
