class FrozenLakeEnv:
    # See [1]

    MAP = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]

    GRID_SIZE   = 8
    NUM_STATES  = 64
    NUM_ACTIONS = 4

    # Action constants — See [2]
    LEFT  = 0
    DOWN  = 1
    RIGHT = 2
    UP    = 3

    def __init__(self):
        # See [3]
        self.current_state = 0
        self.done = False
        self.row = self.current_state // self.GRID_SIZE
        self.col = self.current_state % self.GRID_SIZE

    def reset(self):
        # See [4]
        self.current_state = 0
        self.done = False
        self.row = 0
        self.col = 0
        return self.current_state

    def get_state(self):
        # See [5]
        return self.current_state

    def is_terminal(self):
        # See [6]
        cell = self.MAP[self.row][self.col]
        return cell in ("H", "G")

    def step(self, action):
        # See [7]
        new_row = self.row
        new_col = self.col

        if action == self.LEFT:        # See [8]
            new_col -= 1
        elif action == self.DOWN:
            new_row += 1
        elif action == self.RIGHT:
            new_col += 1
        elif action == self.UP:
            new_row -= 1

        new_row = max(0, min(self.GRID_SIZE - 1, new_row))      # See [9]
        new_col = max(0, min(self.GRID_SIZE - 1, new_col))

        self.row = new_row
        self.col = new_col
        self.current_state = (self.row * self.GRID_SIZE) + self.col

        cell = self.MAP[self.row][self.col]
        reward = 1.0 if cell == "G" else 0.0    # See [10]
        self.done = self.is_terminal()

        return self.current_state, reward, self.done

    def render(self):
        # See [11]
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                if row == self.row and col == self.col:
                    print("A", end=" ")
                else:
                    print(self.MAP[row][col], end=" ")
            print()
        print()


# ─────────────────────────────────────────────
# REFERENCE NOTES
# ─────────────────────────────────────────────
#
# [1]  Custom Frozen Lake 8x8 grid-world environment.
#      Grid cells: S=Start, F=Frozen (safe), H=Hole (terminal), G=Goal (terminal)
#      Actions: 0=Left, 1=Down, 2=Right, 3=Up
#
# [2]  Action constants defined for readability.
#      Use self.LEFT instead of 0 to make the code self-documenting.
#
# [3]  Initialise the environment.
#      Agent starts at state 0 (top-left corner, row=0, col=0).
#      State index formula: state = row * GRID_SIZE + col
#
# [4]  Reset the environment to the starting state.
#      Called at the beginning of every new episode.
#      Returns the initial state (0).
#
# [5]  Return the agent's current state index.
#
# [6]  Check if the current state is terminal.
#      Returns True if the agent is on a Hole (H) or Goal (G), False otherwise.
#
# [7]  Execute one action and transition to the next state.
#      Args:    action (int) — 0=Left, 1=Down, 2=Right, 3=Up
#      Returns: (new_state, reward, done)
#               new_state (int)   — state index after the action
#               reward    (float) — 1.0 if Goal reached, 0.0 otherwise
#               done      (bool)  — True if episode has ended
#
# [8]  Calculate new position based on action.
#      Left/Right change the column. Up/Down change the row.
#
# [9]  Clamp to grid boundaries.
#      Prevents the agent from walking off the edge of the grid.
#      Uses GRID_SIZE - 1 instead of hardcoded 7 for flexibility.
#
# [10] Reward is only awarded for reaching the Goal cell (G).
#      Falling into a Hole (H) gives 0.0 reward.
#
# [11] Print the current grid state to the terminal.
#      Agent's current position is shown as 'A'.
# ─────────────────────────────────────────────