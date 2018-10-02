from collections import deque


class CappedMovingAverage(object):
	def __init__(self, cap):
		self.cap = cap
		self.values = deque()
		self.sum = 0

	def add(self, value):
		if len(self.values) >= self.cap:
			removed = self.values.popleft()
			self.sum -= removed
		self.values.append(value)
		self.sum += value

	def avg(self):
		return self.sum * 1.0 / len(self.values)


