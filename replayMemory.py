"""
	An implementation of a replay memory as described
	at https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

	The class can store new items into memory, save and/or load the entire memory
	to/from disk and sample in batches from it.

	Author: Hanan Aharonof
"""

from collections import deque
import datetime
import random
import pickle
import time
import os

from loggingUtils import info

DUMP_FILE_PATH = "replay_memory"
TIME_FORMAT = '%d-%m-%Y_%H:%M:%S'
FILE_EXT = 'mem'

class ReplayMemory(object):
	def __init__(self, size):
		self.size = size
		self.m = deque()

	def store(self, s_t, a_t, r_t, s_t1, terminal):
		if len(self.m) > self.size:
			self.m.popleft()
		self.m.append((s_t, a_t, r_t, s_t1, terminal))

	def sample(self, batch_size):
		return random.sample(self.m, batch_size)

	def save(self):
		if not os.path.exists(DUMP_FILE_PATH):
			os.mkdir(DUMP_FILE_PATH)

		dump_file_name = '%s.%s' % (
			datetime.datetime.fromtimestamp(time.time()).strftime(TIME_FORMAT), FILE_EXT)
		dump_file = open(os.path.join(DUMP_FILE_PATH, dump_file_name), 'w')
		pickle.dump(self.m, dump_file)

		for f in os.listdir(DUMP_FILE_PATH):
			if f == dump_file_name:
				continue
			path = os.path.join(DUMP_FILE_PATH, f)
			if os.path.isfile(path) and f.endswith('.' + FILE_EXT):
				os.remove(path)

	def load(self):
		if not os.path.exists(DUMP_FILE_PATH):
			os.mkdir(DUMP_FILE_PATH)

		files = [f for f in os.listdir(DUMP_FILE_PATH) if f.endswith("." + FILE_EXT)]
		if len(files) == 0:
			info("No saved replay memory file was found.")
			return

		files.sort(key=lambda x: datetime.datetime.strptime(x[:-3], TIME_FORMAT))
		path = os.path.join(DUMP_FILE_PATH, files[len(files) - 1])
		m = pickle.load(open(path, 'r'))
		if type(m) is type(self.m):
			self.m = m
			info("Successfully loaded replay memory.")
		else:
			info("Failed to load replay memory from file %s." % path)
