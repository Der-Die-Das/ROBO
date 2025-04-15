# src/maze_solver_pkg/state.py
from enum import Enum

class State(Enum):
    """Defines the possible operational states of the robot."""
    FOLLOWING = 1           # Robot is following a line segment
    STOPPED_AT_JUNCTION = 2 # Robot is stopped at a junction, deciding direction
    TURNING = 3             # Robot is executing a turn
    ERROR = 4               # Robot encountered an unrecoverable error