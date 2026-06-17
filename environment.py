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