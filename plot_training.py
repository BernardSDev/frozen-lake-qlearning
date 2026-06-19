import numpy as np
import matplotlib        # See [1]
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from environment import FrozenLakeEnv
from agent import QLearningAgent

EPISODES = 50_000       # See [2]
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01
MAX_STEPS = 200
WINDOW = 1_000      # See [3]

print("Training agent …")

env = FrozenLakeEnv()
agent = QLearningAgent(
    alpha = ALPHA,
    gamma = GAMMA,
    epsilon = EPSILON,
    epsilon_decay = EPSILON_DECAY,
    epsilon_min = EPSILON_MIN
)

rewards = []    # See [4]
successes = []
epsilon_log = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    reward = 0

    while not done and steps < MAX_STEPS:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

    agent.decay_epsilon()
    rewards.append(total_reward)
    successes.append(1 if reward == 1.0 else 0)     # See [5]
    epsilon_log.append(agent.epsilon)       # See [6]

    if (episode + 1) % 1_000 == 0:      # See [7]
        success_rate = sum(successes[-1_000:]) / 1_000 * 100
        print(f"Episode {episode + 1}/{EPISODES} | "
              f"Success Rate: {success_rate:.1f}% | "
              f"Epsilon: {agent.epsilon:.4f}")

print("Training complete.\n")
np.save("results/q_table.npy", agent.q_table)
print("Q-table saved to results/q_table.npy")

rewards_arr = np.array(rewards)
successes_arr = np.array(successes)

rolling_success = np.array([        # See [8]
    successes_arr[max(0, i - WINDOW):i].mean() * 100
    for i in range(1, EPISODES + 1)
])

kernel = 500
smooth_reward = np.convolve(rewards_arr, np.ones(kernel) / kernel, mode="valid")        # See [9]

plt.style.use("seaborn-v0_8-whitegrid")

BLUE   = "#2563EB"      # See [10]
GREEN  = "#16A34A"
RED    = "#DC2626"
GREY   = "#94A3B8"
BG     = "#F8FAFC"
GRID_C = "#E2E8F0"

fig = plt.figure(figsize=(14, 11), facecolor=BG)        # See [11]
fig.suptitle(
    "Frozen Lake Q-Learning – Training Performance\n"
    "DCIT614  ·  Bernard Sam Apoh  ·  22424670",
    fontsize=14, fontweight="bold", color="#1E293B", y=0.98
)

gs = gridspec.GridSpec(3, 1, figure=fig,        # See [12]
                       hspace=0.45, top=0.92, bottom=0.07,
                       left=0.09, right=0.96)

x_fmt = plt.FuncFormatter(lambda x, _: f"{int(x / 1_000)}k")        # See [13]

ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(BG)
ax1.plot(range(1, EPISODES + 1), rolling_success,
         color=GREEN, linewidth=1.4,
         label=f"Rolling success rate (window = {WINDOW:,})")
ax1.axhline(98, color=RED, linewidth=0.8, linestyle="--", alpha=0.7,
            label="98 % reference line")        # See [14]
ax1.set_xlabel("Episode", fontsize=10)
ax1.set_ylabel("Success Rate (%)", fontsize=10)
ax1.set_title("Rolling Success Rate During Training", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 105)
ax1.set_xlim(0, EPISODES)
ax1.xaxis.set_major_formatter(x_fmt)
ax1.legend(fontsize=9)
ax1.tick_params(labelsize=9)
ax1.grid(color=GRID_C, linewidth=0.8)

ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(BG)
ax2.scatter(range(1, EPISODES + 1), rewards_arr,
            color=BLUE, alpha=0.07, s=0.4,
            label="Episode reward (raw)")       # See [15]
ax2.plot(range(kernel, EPISODES + 1), smooth_reward,
         color=BLUE, linewidth=1.6,
         label=f"{kernel}-episode moving average")
ax2.set_xlabel("Episode", fontsize=10)
ax2.set_ylabel("Reward", fontsize=10)
ax2.set_title("Episode Reward Over Training", fontsize=11, fontweight="bold")
ax2.set_xlim(0, EPISODES)
ax2.set_ylim(-0.05, 1.15)
ax2.xaxis.set_major_formatter(x_fmt)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=9)
ax2.grid(color=GRID_C, linewidth=0.8)

convergence_ep = next(      # See [16]
    (i for i, e in enumerate(epsilon_log) if e <= EPSILON_MIN + 1e-6),
    EPISODES
) + 1

ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(BG)
ax3.plot(range(1, EPISODES + 1), epsilon_log,
         color=RED, linewidth=1.4, label="Epsilon (ε)")
ax3.axhline(EPSILON_MIN, color=GREY, linewidth=0.8, linestyle="--",
            label=f"Minimum ε = {EPSILON_MIN}")
ax3.axvline(convergence_ep, color=GREY, linewidth=0.8, linestyle=":",
            label=f"Reaches ε_min ≈ episode {convergence_ep:,}")
ax3.set_xlabel("Episode", fontsize=10)
ax3.set_ylabel("Epsilon (ε)", fontsize=10)
ax3.set_title("Epsilon Decay Over Training", fontsize=11, fontweight="bold")
ax3.set_xlim(0, EPISODES)
ax3.set_ylim(-0.02, 1.05)
ax3.xaxis.set_major_formatter(x_fmt)
ax3.legend(fontsize=9)
ax3.tick_params(labelsize=9)
ax3.grid(color=GRID_C, linewidth=0.8)

out = "results/training_performance.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)        # See [17]
print(f"\nPlot saved → {out}")


# ─────────────────────────────────────────────
# REFERENCE NOTES
# ─────────────────────────────────────────────
#
# [1]  matplotlib.use("Agg") switches to a non-interactive backend.
#      Required when running without a display (e.g. terminal only).
#      Must be called before importing pyplot.
#
# [2]  Hyperparameters are kept identical to train.py so the visualisation
#      reflects the exact same training run documented in the report.
#
# [3]  WINDOW controls the rolling average width for the success rate plot.
#      1,000 episodes smooths out noise while still showing the learning curve.
#
# [4]  Three lists record one value per episode throughout training.
#      rewards     — total reward accumulated in that episode (0.0 or 1.0)
#      successes   — 1 if the agent reached the Goal, 0 otherwise
#      epsilon_log — epsilon value recorded after decay at the end of each episode
#
# [5]  Success is determined by the final reward of the episode.
#      reward == 1.0 means the agent reached the Goal on its last step.
#      Matches the same logic used in train.py.
#
# [6]  Epsilon is logged after decay so the curve reflects the value
#      the agent will use at the start of the next episode.
#
# [7]  Progress is printed every 1,000 episodes, consistent with train.py.
#
# [8]  Rolling success rate: for each episode i, compute the mean of the
#      last WINDOW successes and convert to a percentage.
#      max(0, i - WINDOW) prevents a negative slice index early in training.
#
# [9]  np.convolve with a uniform kernel of size 500 computes a moving average
#      of episode rewards, smoothing out the noise from sparse rewards.
#      mode="valid" returns only fully-overlapping windows, so the result
#      is shorter than the input by (kernel - 1) elements.
#
# [10] Colour palette chosen for clarity and accessibility.
#      Each plot uses a distinct colour so the three charts are easy to read.
#
# [11] figsize=(14, 11) gives enough vertical space for three stacked subplots
#      without crowding the axis labels or titles.
#
# [12] GridSpec divides the figure into 3 rows with consistent spacing.
#      hspace controls vertical gap between subplots.
#
# [13] FuncFormatter converts raw episode numbers to "10k", "20k" etc.
#      Applied to all three x-axes for consistent, readable tick labels.
#
# [14] The 98% reference line marks the target success rate achieved during
#      training, making it easy to see when the agent crosses that threshold.
#
# [15] Raw episode rewards are plotted as near-transparent scatter points
#      to show the underlying distribution without obscuring the moving average.
#      Low alpha and small marker size (s=0.4) keep the background subtle.
#
# [16] convergence_ep finds the first episode where epsilon reaches its minimum.
#      A tolerance of 1e-6 is added to account for floating-point precision.
#      This episode is marked with a vertical dotted line on the epsilon plot.
#
# [17] dpi=180 produces a high-resolution image suitable for inclusion in
#      the report. bbox_inches="tight" removes excess whitespace around the figure.
# ─────────────────────────────────────────────