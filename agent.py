import numpy as np
import random

class QLearningAgent:
    # See [1]
    NUM_STATES = 64
    NUM_ACTIONS = 4

    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        # See [2]
        self.q_table = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        # See [3]
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state])
        



# ─────────────────────────────────────────────
# REFERENCE NOTES
# ─────────────────────────────────────────────
#
# [1]  Q-Learning agent for the Frozen Lake environment.
#      Learns an optimal policy through trial and error using the Q-Learning algorithm.
#
# [2]  Initialise the agent with hyperparameters.
#      q_table       — 64x4 matrix of state-action values, all starting at 0.0
#      alpha         — learning rate: how fast the agent updates its Q-values
#      gamma         — discount factor: how much future rewards are valued
#      epsilon       — exploration rate: probability of taking a random action
#      epsilon_decay — multiplier applied to epsilon after each episode
#      epsilon_min   — minimum value epsilon can reach
# 
# [3]  Choose an action using epsilon-greedy strategy.
#      With probability epsilon: explore by picking a random action (0-3)
#      Otherwise: exploit by picking the action with the highest Q-value
#      for the current state. Returns the chosen action as an integer.
# ─────────────────────────────────────────────