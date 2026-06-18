# Frozen Lake Q-Learning

## Introduction

### What is Reinforcement Learning?
Reinforcement Learning (RL) is a type of machine learning where an agent learns 
to make decisions by interacting with an environment. The agent receives a reward 
signal after each action and learns to maximise cumulative reward over time through 
trial and error — without being explicitly told what to do.

### What is Frozen Lake?
Frozen Lake is a grid-world environment where an agent must navigate from a 
starting position (S) to a goal position (G) on an 8×8 grid, while avoiding 
holes (H). The agent learns an optimal policy using Q-Learning.

## Environment Design

### State Representation
The environment is an 8×8 grid with 64 states. Each state is represented as a 
single integer index calculated as:

state = row × 8 + col

- State 0 = Start (row 0, col 0)
- State 63 = Goal (row 7, col 7)

### Action Representation
The agent has 4 possible actions at each state:

| Action | Direction |
|--------|-----------|
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

### Reward Structure
| Event              | Reward |
|--------------------|--------|
| Reaching the Goal  | +1.0   |
| All other steps    | 0.0    |
| Falling in a Hole  | 0.0    |

The episode ends immediately when the agent reaches the Goal or falls into a Hole.

## Q-Learning Algorithm

### Description
Q-Learning is a model-free, off-policy reinforcement learning algorithm. The agent 
maintains a Q-table of size 64×4 (states × actions) that stores the expected 
cumulative reward for each state-action pair. The agent updates this table after 
every step using the Bellman equation.

### Update Equation
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]

Where:
- Q(s,a) — current Q-value for state s and action a
- α (alpha) — learning rate: controls how fast the agent updates its estimates
- r — immediate reward received after taking action a
- γ (gamma) — discount factor: how much future rewards are valued
- max Q(s',a') — best Q-value achievable from the next state s'

### Exploration Strategy
The agent uses an **epsilon-greedy** strategy:
- With probability ε: take a random action (explore)
- With probability 1-ε: take the best known action (exploit)

Epsilon starts at 1.0 (fully random) and decays after each episode until it 
reaches a minimum of 0.01. This ensures the agent explores the environment 
thoroughly early in training before shifting to exploitation.

Ties in Q-values are broken randomly to prevent the agent from always defaulting 
to the same action when Q-values are equal.

## Training Procedure

### Hyperparameters

| Parameter      | Value  | Description                          |
|----------------|--------|--------------------------------------|
| Episodes       | 50000  | Total number of training episodes    |
| Learning Rate  | 0.1    | Controls Q-table update step size    |
| Discount Factor| 0.99   | How much future rewards are valued   |
| Epsilon Start  | 1.0    | Initial exploration rate             |
| Epsilon Decay  | 0.9999 | Multiplier applied after each episode|
| Epsilon Min    | 0.01   | Minimum exploration rate             |
| Max Steps      | 200    | Maximum steps allowed per episode    |

### Training Process
1. Agent starts at state 0 (Start cell) every episode
2. Agent selects an action using epsilon-greedy strategy
3. Environment executes the action and returns new state, reward and done flag
4. Agent updates Q-table using the Q-Learning equation
5. Epsilon is decayed after each episode
6. Process repeats for 50,000 episodes

## Results

### Training Performance
The agent achieved a **98.8% success rate** over the final 1000 training episodes, 
demonstrating that it successfully learned an optimal policy for navigating the 
Frozen Lake map.

### Evaluation Performance
After training, the agent was evaluated over 100 episodes with epsilon set to 0 
(pure exploitation):

| Metric          | Value  |
|-----------------|--------|
| Episodes        | 100    |
| Successful Runs | 100    |
| Failures        | 0      |
| Success Rate    | 100.0% |
| Average Reward  | 1.000  |

### Learned Policy
The following grid shows the best action for every non-terminal state:

↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓

→  →  →  →  ↓  ↓  ↓  ↓

→  →  ↑  H  ↓  ↓  ↓  ↓

↑  ↑  ↑  H  ↓  ↓  ↓  ↓

↑  ↑  ↑  H  →  →  →  ↓

↑  H  H  →  →  ↑  H  ↓

↑  H  →  ↑  H  ↑  H  ↓

↑  ←  ←  H  →  ↑  ←  G

### Discussion
The agent struggled initially due to the sparse reward structure — only receiving 
a reward upon reaching the Goal. The key breakthrough was fixing the action 
selection tie-breaking, which prevented the agent from always defaulting to the 
same action when Q-values were equal early in training.

## Execution Instructions

### Requirements
Python 3.8 or higher

### Installation
```bash
# Clone the repository
git clone https://github.com/BernardSDev/frozen-lake-qlearning.git
cd frozen-lake-qlearning

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

**Train the agent:**
```bash
python train.py
```

**Evaluate the trained agent:**
```bash
python evaluate.py
```

### Project Structure
```
frozen-lake-qlearning/
├── environment.py  — Custom Frozen Lake environment
├── agent.py — Q-Learning agent
├── train.py — Training loop and policy extraction
├── evaluate.py — Evaluation script
├── requirements.txt
├── README.md
├── report.pdf
└── results/
    └── q_table.npy — Saved Q-table after training
```