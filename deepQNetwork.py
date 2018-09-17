"""
	An implementation of a deep Q network as described
	at https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

	Author: Hanan Aharonof
"""

import datetime
import time
import os

import tensorflow as tf
import numpy as np

from loggingUtils import info
from replayMemory import ReplayMemory

SAVED_NETWORKS_FOLDER_NAME = "saved_networks"
TIME_FORMAT = '%d-%m-%Y_%H:%M:%S'
FILE_EXT = 'net'

# Network params names
WIDTH = 'width'  # Layout width in pixels
HEIGHT = 'height'  # Layout height in pixels
M = 'm'  # The number of the most recent frames to stack
ACTIONS = 'actions'  # The number of actions
REPLAY_MEMORY_SIZE = 'replay_memory_size'  # Replay memory size
DISCOUNT_FACTOR = 'discount_factor'  # Discount factor
RMS_LEARNING_RATE = 'rms_learning_rate'  # RMS learning rate
RMS_EPSILON = 'rms_epsilon'  # RMS epsilon
# End of network params names


class DeepQNetwork(object):
	def __init__(self, params):
		# 1. Start a TensorFlow session
		self.session = tf.InteractiveSession()

		# 2. Define network input placeholders
		self.s_t = tf.placeholder("float", [None, params[WIDTH], params[HEIGHT], params[M]], name='states')
		self.a_t = tf.placeholder("float", [None, params[ACTIONS]], name='actions')
		self.r_t = tf.placeholder("float", [None], name='rewards')
		self.f_t = tf.placeholder("float", [None], name='finals')
		self.q_t = tf.placeholder("float", [None], name='q-values')

		# 3. Init replay memory
		self.replay_memory = self._init_replay_memory()

		# 4. Define network and load weights
		self.actions_probs = self._define_network()
		self.session_saver = tf.train.Saver()
		self.step_counter = self._load_saved_network_data()

		# 5. Define loss function
		gamma_times_qt = tf.multiply(params[DISCOUNT_FACTOR], self.q_t)
		gamma_times_qt_zero_ft = tf.multiply(1.0 - self.f_t, gamma_times_qt)
		pred = tf.reduce_sum(tf.multiply(self.actions_probs, self.a_t))

		self.loss = tf.reduce_sum(tf.pow(tf.subtract(gamma_times_qt_zero_ft, pred), 2))

		# 6. Define gradient descent
		self.rms_prop = tf.train.RMSPropOptimizer(
			params[RMS_LEARNING_RATE],
			epsilon=params[RMS_EPSILON]).minimize(self.loss, global_step=self.step_counter)

	def _init_replay_memory(self):
		memory = ReplayMemory(self.replay_memory_size)
		memory.load()
		return memory

	def _define_network(self):
		# Create the 2'nd layer as a convolution layer with 32 filters of size 8x8, depth [m] and stride 4
		w1 = tf.Variable(tf.random_normal([8, 8, self.m, 32], stddev=0.01), name='w1')
		b1 = tf.Variable(tf.constant(0.01, shape=[32]), name='b1')
		c1 = tf.nn.conv2d(self.s_t, w1, strides=[1, 4, 4, 1], padding="SAME", name='c1')
		l2_hidden = tf.nn.relu(tf.add(c1 + b1), name='l2')

		# Create the 3'rd layer as a convolution layer with 64 filters of size 4x4, depth 32 and stride 2
		w2 = tf.Variable(tf.random_normal([4, 4, 32, 64], stddev=0.01), name='w2')
		b2 = tf.Variable(tf.constant(0.01, shape=[64]), name='b2')
		c2 = tf.nn.conv2d(l2_hidden, w2, strides=[1, 2, 2, 1], padding="SAME", name='c2')
		l3_hidden = tf.nn.relu(tf.add(c2 + b2), name='l3')

		# Create the 4'th layer as a convolution layer with 64 filters of size 3x3, depth 64 and stride 1
		w3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01), name='w3')
		b3 = tf.Variable(tf.constant(0.01, shape=[64]), name='b3')
		c3 = tf.nn.conv2d(l3_hidden, w3, strides=[1, 1, 1, 1], padding="SAME", name='c3')
		l4_hidden = tf.nn.relu(tf.add(c3 + b3), name='l4')

		# Create the 5'th layer as a full connected layer with 512 rectifier units
		l4_shape = l4_hidden.get_shape().as_list()
		l5_dimension = l4_shape[1] * l4_shape[2] * l4_shape[3]
		l4_flatted = tf.reshape(l4_hidden, [-1, l5_dimension])
		w4 = tf.Variable(tf.random_normal([l5_dimension, 512], stddev=0.01), name='w4')
		b4 = tf.Variable(tf.constant(0.01, shape=[512]), name='b4')
		l5_hidden = tf.nn.relu(tf.matmul(l4_flatted, w4) + b4)

		# Create the 6'th layer as fully connected linear layer with [action] actions
		w5 = tf.Variable(tf.random_normal([512, self.a_t], stddev=0.01), name='w5')
		b5 = tf.Variable(tf.constant(0.01, shape=[self.a_t]), name='b5')
		actions_probs = tf.matmul(l5_hidden, w5) + b5

		return actions_probs

	def _load_saved_network_data(self):
		self.session.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORKS_FOLDER_NAME)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.session_saver.restore(self.session, checkpoint.model_checkpoint_path)
			info("Successfully loaded network weights %s." % checkpoint.model_checkpoint_path)
			return tf.Variable(
				checkpoint.model_checkpoint_path.split('^^^')[1],
				name='step_counter',
				trainable=False)
		else:
			info("No saved network weights found. Using random values.")
			return tf.Variable(0, name='step_counter', trainable=False)

	def train_network(self, s_t, a_t, f_t, s_t1, r_t):
		# 1. Calculate q_t from new states
		q_t = self.session.run(self.actions_probs, feed_dict={self.s_t: s_t1})
		q_t = np.amax(q_t, axis=1)

		# 2. Train
		feed_dict = {
			self.s_t: s_t,
			self.q_t: q_t,
			self.a_t: a_t,
			self.f_t: f_t,
			self.r_t: r_t
		}

		_, count, loss = self.session.run([self.rms_prop, self.step_counter, self.loss], feed_dict=feed_dict)
		return count, loss

	def save_network(self, step_counter):
		dump_file_name = '%s^^^%s.%s' % (
			datetime.datetime.fromtimestamp(time.time()).strftime(TIME_FORMAT),
			step_counter,
			FILE_EXT)
		file_name = os.path.join(SAVED_NETWORKS_FOLDER_NAME, dump_file_name)
		self.session_saver.save(self.session, file_name)
