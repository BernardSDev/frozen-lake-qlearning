from environment import FrozenLakeEnv
from agent import QLearningAgent
import numpy as np

# See [1]
env = FrozenLakeEnv()
agent = QLearningAgent(
    alpha = 0.1,
    gamma = 0.99,
    epsilon = 0.0,
    epsilon_decay = 1.0,
    epsilon_min = 0.0
)

# See [2]
agent.q_table = np.load("results/q_table.npy")

EVAL_EPISODES = 100

successes = []
rewards = []

# See [3]
for episode in range(EVAL_EPISODES):
    state = env.reset()
    done = False
    max_steps = 200
    steps = 0

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1

    successes.append(1 if reward == 1.0 else 0)
    rewards.append(reward)

# See [4]
print("\n─── Evaluation Results ───")
print(f"Episodes: {EVAL_EPISODES}")
print(f"Successful Runs: {sum(successes)}")
print(f"Failures: {EVAL_EPISODES - sum(successes)}")
print(f"Success Rate: {sum(successes) / EVAL_EPISODES * 100:.1f}%")
print(f"Average Reward: {sum(rewards) / EVAL_EPISODES:.3f}")


# ─────────────────────────────────────────────
# REFERENCE NOTES
# ─────────────────────────────────────────────
#
# [1]  Initialise environment and agent for evaluation.
#      Epsilon is set to 0.0 — pure exploitation, no random actions.
#      The agent always picks the best known action from the Q-table.
#
# [2]  Load the Q-table saved after training.
#      The agent uses this to make decisions without learning further.
#
# [3]  Evaluation loop — runs for EVAL_EPISODES iterations.
#      No Q-table updates and no epsilon decay — agent is not learning.
#      Results are recorded after each episode.
#
# [4]  Print final evaluation summary.
#      Reports success rate, average reward, wins and failures
#      over all evaluation episodes.
# ─────────────────────────────────────────────