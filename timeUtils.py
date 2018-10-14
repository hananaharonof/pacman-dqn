import time
from loggingUtils import caller_name, debug


class TimeIt(object):
	def __init__(self):
		self.s = time.time()

	def measure(self):
		debug("Time elapsed: %s seconds." % str(time.time() - self.s), caller_name())
