import numpy as np
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)
