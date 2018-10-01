"""
	An implementation of a deep Q network as described
	at https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

	Author: Hanan Aharonof
"""

import os

import numpy as np
import tensorflow as tf

from loggingUtils import info
from objectMapper import SAVE_FOLDER_NAME, get_time
from dqnParameters import *

FILE_EXT = 'net'


def _define_weights(filter_size, depth, filters, name):
	return tf.Variable(tf.random_normal(
		[filter_size, filter_size, depth, filters],
		stddev=0.01),
		name=name)


def _define_convolution(input_mat, filters, stride, name):
	return tf.nn.conv2d(input_mat, filters, strides=[1, stride, stride, 1], padding="SAME", name=name)


def _generate_model_file_path(prefix):
	model_file_name = '%s.%s' % (prefix, FILE_EXT)
	return os.path.join(SAVE_FOLDER_NAME, model_file_name)


class DeepQNetwork(object):
	def __init__(self, params):
		self.params = params

		# 1. Start a TensorFlow session
		self.session = tf.Session()

		# 2. Define network vars placeholders
		self.states = tf.placeholder("float",
			[None, params[FRAME_WIDTH], params[FRAME_HEIGHT], params[FRAME_STACK_SIZE]], name='states')
		self.actions = tf.placeholder("float", [None, params[NUM_OF_ACTIONS]], name='actions')
		self.rewards = tf.placeholder("float", [None], name='rewards')
		self.terminals = tf.placeholder("float", [None], name='terminals')
		self.q = tf.placeholder("float", [None], name='q-values')

		# 3. Define network
		self.action_predictor = self._define_network()

		# 4. Load saved network
		self.session_saver = tf.train.Saver()
		self.global_step = self._load_saved_network_data()

		# 4. Define loss function
		self.loss = self._define_loss()

		# 5. Define gradient descent optimizer
		self.optimizer = tf.train.AdamOptimizer(
			self.params[OPTIMIZER_LEARNING_RATE]).minimize(
			self.loss, global_step=self.global_step)

		self.session.run(tf.global_variables_initializer())

	def _define_loss(self):
		gamma_times_qt = tf.multiply(tf.constant(self.params[RL_DISCOUNT_FACTOR]), self.q)
		gamma_times_qt_zero_ft = tf.multiply(1.0 - self.terminals, gamma_times_qt)
		prediction = tf.reduce_sum(tf.multiply(self.action_predictor, self.actions))
		return tf.reduce_sum(tf.pow(tf.subtract(gamma_times_qt_zero_ft, prediction), 2))

	def _define_network(self):
		# Create the 2'nd layer as a convolution layer
		w1 = _define_weights(self.params[L2_FILTER_SIZE], self.params[L2_DEPTH], self.params[L2_FILTERS], 'w1')
		b1 = tf.Variable(tf.constant(0.1, shape=[self.params[L2_FILTERS]]), name='b1')
		c1 = _define_convolution(self.states, w1, self.params[L2_STRIDE], 'c1')
		l2_hidden = tf.nn.relu(tf.add(c1, b1), name='l2')

		# Create the 3'rd layer as a convolution layer
		w2 = _define_weights(self.params[L3_FILTER_SIZE], self.params[L3_DEPTH], self.params[L3_FILTERS], 'w2')
		b2 = tf.Variable(tf.constant(0.1, shape=[self.params[L3_FILTERS]]), name='b2')
		c2 = _define_convolution(l2_hidden, w2, self.params[L3_STRIDE], 'c2')
		l3_hidden = tf.nn.relu(tf.add(c2, b2), name='l3')

		# Create the 4'th layer as a full connected layer
		l4_shape = l3_hidden.get_shape().as_list()
		l4_dimension = l4_shape[1] * l4_shape[2] * l4_shape[3]
		l3_flatted = tf.reshape(l3_hidden, [-1, l4_dimension])
		w3 = tf.Variable(tf.random_normal([l4_dimension, self.params[L4_RECTIFIER_UNITS]], stddev=0.01), name='w3')
		b3 = tf.Variable(tf.constant(0.1, shape=[self.params[L4_RECTIFIER_UNITS]]), name='b3')
		l4_hidden = tf.nn.relu(tf.matmul(l3_flatted, w3) + b3)

		# Create the 5'th layer as fully connected linear layer with [action] actions
		w4 = tf.Variable(tf.random_normal([self.params[L4_RECTIFIER_UNITS], self.params[NUM_OF_ACTIONS]], stddev=0.01),
			name='w4')
		b4 = tf.Variable(tf.constant(0.01, shape=[self.params[NUM_OF_ACTIONS]]), name='b4')
		network_output = tf.matmul(l4_hidden, w4) + b4

		return network_output

	def _load_saved_network_data(self):
		if self.params[MODEL] is None:
			info("Using random network weights.")
			return tf.Variable(0, name='global_step', trainable=False)

		network_file_path = _generate_model_file_path(self.params[MODEL])
		if not os.path.exists(network_file_path):
			info("Model %s does not exist. Using random network weights." % network_file_path)
			return tf.Variable(0, name='global_step', trainable=False)

		self.session_saver.restore(self.session, network_file_path)
		info("Successfully loaded saved weights of model %s." % self.params[MODEL])

		return tf.Variable(self.params[GLOBAL_STEP], name='global_step', trainable=False)

	def train_network(self, states, actions, rewards, new_states, terminals):
		# 1. Calculate q from new states
		feed_dict = {
			self.states: new_states,
			self.q: np.zeros(new_states.shape[0]),
			self.actions: actions,
			self.rewards: rewards,
			self.terminals: terminals
		}

		q = self.session.run(self.action_predictor, feed_dict=feed_dict)
		q = np.amax(q, axis=1)

		# 2. Train
		feed_dict = {
			self.states: states,
			self.q: q,
			self.actions: actions,
			self.rewards: rewards,
			self.terminals: terminals
		}

		_, count, loss = self.session.run([self.optimizer, self.global_step, self.loss], feed_dict=feed_dict)
		return count, loss

	def save_network(self, model=None):
		if model is None:
			model = get_time()
		path = _generate_model_file_path(model)
		self.session_saver.save(self.session, path)

	def predict(self, new_state):
		feed_dict = {
			self.states: new_state,
			self.q: np.zeros(1),
			self.actions: np.zeros((1, 4)),
			self.rewards: np.zeros(1),
			self.terminals: np.zeros(1)
		}

		return self.session.run(self.action_predictor, feed_dict=feed_dict)
