"""
	A utility to calculate moving average with a fix window size.

	Author: Hanan Aharonof
"""

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
		if len(self.values) == 0:
			return 0.0
		return self.sum * 1.0 / len(self.values)
