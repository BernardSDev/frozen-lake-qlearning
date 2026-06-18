class FrozenLakeEnv:
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

    GRID_SIZE = 8
    NUM_STATES = 64
    NUM_ACTIONS = 4

    def __init__(self):
        self.current_state = 0
        self.done = False
        self.row = self.current_state // 8
        self.col = self.current_state % 8

    def reset(self):
        self.current_state = 0
        self.done = False
        self.row = 0
        self.col = 0
        return self.current_state
    
    def get_state(self):
        return self.current_state
    
    def is_terminal(self):
        cell = self.MAP[self.row][self.col]
        return cell in ("H", "G")
    
    def step(self, action):
        new_row = self.row
        new_col = self.col

        if action == 0:
            new_col -= 1
        elif action == 1:
            new_row += 1
        elif action == 2:
            new_col += 1
        elif action == 3:
            new_row -= 1

        new_row = max(0, min(7, new_row))
        new_col = max(0, min(7, new_col))

        self.row = new_row
        self.col = new_col
        self.current_state = (self.row * 8) + self.col
        cell = self.MAP[self.row][self.col]
        reward = 1 if cell in ("G") else 0 
        self.done = self.is_terminal()

        return self.current_state, reward, self.done
    
    def render(self):
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                if row == self.row and col == self.col:
                    print("A", end=" ")
                else:
                    print(self.MAP[row][col], end=" ")
            print()
    
