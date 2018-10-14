import numpy as np
from collections import deque


class FrameStack(object):
	def __init__(self, stack_size, width, height):
		self.size = stack_size
		self.width = width
		self.height = height
		self.stack = self.init_stack()

	def init_stack(self):
		return deque([np.zeros((self.width, self.height), dtype=np.int) for i in range(self.size)], maxlen=self.size)

	def reset(self, frame):
		self.stack = self.init_stack()
		for i in xrange(self.size):
			self.add(frame)

	def add(self, frame):
		self.stack.append(frame)

	def get_stack(self):
		return np.stack(self.stack, axis=2)

