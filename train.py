import numpy as np
from environment import FrozenLakeEnv
from agent import QLearningAgent

# See [1]
EPISODES = 50000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01

# See [2]
rewards = []
successes = []

# See [3]
env = FrozenLakeEnv()
agent = QLearningAgent(
    alpha = ALPHA,
    gamma = GAMMA,
    epsilon = EPSILON,
    epsilon_decay = EPSILON_DECAY,
    epsilon_min = EPSILON_MIN
)

# See [4]
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    max_steps = 200  # See [5]
    steps = 0

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

    agent.decay_epsilon()
    rewards.append(total_reward)
    successes.append(1 if reward == 1.0 else 0)  # See [6]

    # See [7]
    if (episode + 1) % 1000 == 0:
        success_rate = sum(successes[-1000:]) / 1000 * 100
        print(f"Episode {episode + 1}/{EPISODES} | "
              f"Success Rate: {success_rate:.1f}% | "
              f"Epsilon: {agent.epsilon:.3f}")  
            
# See [8]
print("\n─── Learned Policy ───")

action_symbols = {
    0: "←", 
    1: "↓", 
    2: "→", 
    3: "↑"
}

for row in range(8):
    for col in range(8):
        state = row * 8 + col
        cell = env.MAP[row][col]

        if cell in ("H", "G"):
            print(cell, end=" ")
        else:
            best_action = np.argmax(agent.q_table[state])
            print(action_symbols[best_action], end=" ")
    print()


# ─────────────────────────────────────────────
# REFERENCE NOTES
# ─────────────────────────────────────────────
#
# [1]  Hyperparameters for training.
#      EPISODES      — total number of training episodes
#      ALPHA         — learning rate: controls Q-table update step size
#      GAMMA         — discount factor: how much future rewards are valued
#      EPSILON       — starting exploration rate (1.0 = fully random)
#      EPSILON_DECAY — multiplier applied to epsilon after each episode
#      EPSILON_MIN   — floor value epsilon cannot go below
#
# [2]  Training statistics recorded per episode.
#      rewards   — total reward accumulated each episode
#      successes — 1 if agent reached Goal, 0 otherwise
#
# [3]  Initialise the environment and agent before training begins.
#
# [4]  Main training loop — runs for EPISODES iterations.
#      Each iteration is one full episode: agent starts at S, acts
#      until it reaches H, G, or hits the step limit.
#
# [5]  Step limit per episode prevents infinite loops in early training
#      when the agent has not yet learned to reach a terminal state.
#
# [6]  Success is determined by the final reward of the episode.
#      reward == 1.0 means the agent reached the Goal on its last step.
#
# [7]  Print a progress summary every 1000 episodes.
#      Shows success rate over the last 1000 episodes and current epsilon.
# 
# [8]  Policy extraction — displays the best action for every non-terminal
#      state as a directional arrow. Holes and Goal show their cell letter.
#      Best action = argmax of Q-table row for that state.
# ─────────────────────────────────────────────