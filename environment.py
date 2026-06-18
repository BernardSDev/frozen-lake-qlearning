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
    