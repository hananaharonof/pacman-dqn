"""
	An implementation of a replay memory as described
	at https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

	The class can store new items into memory, save and/or load the entire memory
	to/from disk and sample in batches from it.

	Author: Hanan Aharonof
"""
import random
import numpy as np

from collections import deque

from objectMapper import ObjectMapper

REPLAY_MEMORY_EXT = 'mem'


class ReplayMemory(ObjectMapper):
	def __init__(self, size):
		ObjectMapper.__init__(self, REPLAY_MEMORY_EXT)
		self.size = size
		self.m = deque()

	def add_memory(self, state, action, reward, new_state, terminal_state):
		if len(self.m) > self.size:
			self.m.popleft()
		self.m.append((state, action, reward, new_state, terminal_state))

	def sample(self, batch_size):
		sample = random.sample(self.m, batch_size)
		states = []
		actions = []
		rewards = []
		new_states = []
		terminals = []

		for i in sample:
			states.append(i[0])
			actions.append(i[1])
			rewards.append(i[2])
			new_states.append(i[3])
			terminals.append(i[4])

		return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(terminals)
