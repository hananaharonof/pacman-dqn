"""
	An implementation of a replay memory as described
	at https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

	The class can store new items into memory, save and/or load the entire memory
	to/from disk and sample in batches from it.

	Author: Hanan Aharonof
"""
import random
from collections import deque

import numpy as np

from objectMapper import ObjectMapper

REPLAY_MEMORY_EXT = 'mem'


class ReplayMemory(ObjectMapper):
	def __init__(self, size):
		ObjectMapper.__init__(self, REPLAY_MEMORY_EXT)
		self.m = deque(maxlen=size)

	def add(self, exp):
		self.m.append(exp)

	def sample(self, batch_size):
		#return random.sample(self.m, batch_size)
		curr_size = len(self.m)
		index = np.random.choice(np.arange(curr_size), size=curr_size, replace=False)
		return [self.m[i] for i in index]
