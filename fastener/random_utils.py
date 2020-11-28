"""
Random utils.
"""

import random as core_random
import numpy.random as np_random

core_random = np_random

seed = core_random.seed
shuffle = core_random.shuffle
choices = core_random.choice
random = core_random.random

get_state = core_random.get_state
set_state = core_random.set_state
