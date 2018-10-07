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
	def __init__(self, params, name, session):
		self.params = params
		self.name = name

		# 1. Start a TensorFlow session
		self.session = session

		with tf.variable_scope(name) as scope:
			# 2. Define network vars placeholders
			self.states = tf.placeholder("float",
				[None, params[FRAME_WIDTH], params[FRAME_HEIGHT], params[FRAME_STACK_SIZE]], name=self.name_it('states'))
			self.actions = tf.placeholder("float", [None, params[NUM_OF_ACTIONS]], name=self.name_it('actions'))
			self.rewards = tf.placeholder("float", [None], name=self.name_it('rewards'))
			self.terminals = tf.placeholder("float", [None], name=self.name_it('terminals'))
			self.q = tf.placeholder("float", [None], name=self.name_it('q-values'))

			# 3. Define network
			self.action_predictor = self._define_network()

			# 4. Define loss function
			self.loss = self._define_loss()

		self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
		self.trainable_vars = {var.name[len(scope.name):]: var for var in self.trainable_vars}

		# 5. Load saved network
		self.session_saver = tf.train.Saver()
		self.global_step = self._load_saved_network_data()

		# 6. Define gradient descent optimizer
		self.optimizer = tf.train.AdamOptimizer(
			self.params[OPTIMIZER_LEARNING_RATE]).minimize(
			self.loss, global_step=self.global_step)

		self.session.run(tf.global_variables_initializer())

	def name_it(self, var):
		return '%s' % (var)

	def _define_loss(self):
		bellman = tf.multiply(tf.constant(self.params[RL_DISCOUNT_FACTOR]), self.q)
		bellman = tf.multiply(1.0 - self.terminals, bellman)
		bellman = tf.add(self.rewards, bellman)
		prediction = tf.reduce_sum(tf.multiply(self.action_predictor, self.actions), reduction_indices=1)
		return tf.reduce_sum(tf.pow(tf.subtract(bellman, prediction), 2))

	def _define_network(self):
		# Create the first convolution layer
		size = self.params[L2_FILTER_SIZE]
		depth = self.params[L2_DEPTH]
		filters = self.params[L2_FILTERS]
		stride = self.params[L2_STRIDE]

		w1 = tf.Variable(tf.random_normal([size, size, depth, filters], stddev=0.01), name=self.name_it('w1'))
		b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.name_it('b1'))
		c1 = tf.nn.conv2d(self.states, w1, strides=[1, stride, stride, 1], padding="SAME", name=self.name_it('c1'))
		conv_1 = tf.nn.relu(tf.add(c1, b1), name=self.name_it('conv1'))

		# Create the second convolution layer
		size = self.params[L3_FILTER_SIZE]
		depth = self.params[L3_DEPTH]
		filters = self.params[L3_FILTERS]
		stride = self.params[L3_STRIDE]

		w2 = tf.Variable(tf.random_normal([size, size, depth, filters], stddev=0.01), name=self.name_it('w2'))
		b2 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.name_it('b2'))
		c2 = tf.nn.conv2d(conv_1, w2, strides=[1, stride, stride, 1], padding="SAME", name=self.name_it('c2'))
		conv_2 = tf.nn.relu(tf.add(c2, b2), name=self.name_it('conv2'))

		# Create the third convolution layer
		size = self.params[L4_FILTER_SIZE]
		depth = self.params[L4_DEPTH]
		filters = self.params[L4_FILTERS]
		stride = self.params[L4_STRIDE]

		w3 = tf.Variable(tf.random_normal([size, size, depth, filters], stddev=0.01), name=self.name_it('w3'))
		b3 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.name_it('b3'))
		c3 = tf.nn.conv2d(conv_2, w3, strides=[1, stride, stride, 1], padding="SAME", name=self.name_it('c3'))
		conv_3 = tf.nn.relu(tf.add(c3, b3), name=self.name_it('conv3'))

		# Create the fully connected layer
		fc_shape = conv_3.get_shape().as_list()
		fc_dimension = fc_shape[1] * fc_shape[2] * fc_shape[3]

		conv_3_flatted = tf.reshape(conv_3, [-1, fc_dimension])

		w_fc = tf.Variable(tf.random_normal([fc_dimension, self.params[FC_RECTIFIER_UNITS]], stddev=0.01), name=self.name_it('w_fc'))
		b_fc = tf.Variable(tf.constant(0.1, shape=[self.params[FC_RECTIFIER_UNITS]]), name=self.name_it('b_fc'))
		fc = tf.nn.relu(tf.add(tf.matmul(conv_3_flatted, w_fc), b_fc))

		# Create the second fully connected linear layer with [action] actions
		w_fc2 = tf.Variable(tf.random_normal([self.params[FC_RECTIFIER_UNITS], self.params[NUM_OF_ACTIONS]], stddev=0.01), name=self.name_it('w_fc2'))
		b_fc2 = tf.Variable(tf.constant(0.01, shape=[self.params[NUM_OF_ACTIONS]]), name=self.name_it('b_fc2'))
		return tf.add(tf.matmul(fc, w_fc2), b_fc2)

	def _load_saved_network_data(self):
		if self.params[MODEL] is None:
			info("Using random network weights.")
			return tf.Variable(0, name=self.name_it('global_step'), trainable=False)

		network_file_path = _generate_model_file_path(self.params[MODEL])
		if not os.path.exists(network_file_path):
			info("Model %s does not exist. Using random network weights." % network_file_path)
			return tf.Variable(0, name=self.name_it('global_step'), trainable=False)

		self.session_saver.restore(self.session, network_file_path)
		info("Successfully loaded saved weights of model %s." % self.params[MODEL])

		return tf.Variable(self.params[GLOBAL_STEP], name=self.name_it('global_step'), trainable=False)

	def estimate_q_values_and_train(self, states, actions, rewards, new_states, terminals):
		# 1. Estimate q-values
		q = self.estimate_q_values(actions, new_states, rewards, terminals)

		# 2. Train
		return self.train(actions, q, rewards, states, terminals)

	def train(self, actions, q, rewards, states, terminals):
		feed_dict = {
			self.states: states,
			self.q: q,
			self.actions: actions,
			self.rewards: rewards,
			self.terminals: terminals
		}
		_, count, loss = self.session.run([self.optimizer, self.global_step, self.loss], feed_dict=feed_dict)
		return count, loss

	def estimate_q_values(self, actions, new_states, rewards, terminals):
		feed_dict = {
			self.states: new_states,
			self.q: np.zeros(new_states.shape[0]),
			self.actions: actions,
			self.rewards: rewards,
			self.terminals: terminals
		}
		q = self.session.run(self.action_predictor, feed_dict=feed_dict)
		return np.amax(q, axis=1)

	def save(self, model=None):
		if model is None:
			model = get_time()
		path = _generate_model_file_path(model)
		self.session_saver.save(self.session, path)

	def predict(self, new_state):
		feed_dict = {
			self.states: new_state,
			self.q: np.zeros(1),
			self.actions: np.zeros((1, self.params[NUM_OF_ACTIONS])),
			self.rewards: np.zeros(1),
			self.terminals: np.zeros(1)
		}

		return self.session.run(self.action_predictor, feed_dict=feed_dict)[0]

	def update(self, other_dqn):
		copy_ops = [target_var.assign(other_dqn.vars()[var_name]) for var_name, target_var in self.trainable_vars.items()]
		self.session.run(tf.group(*copy_ops))

	def vars(self):
		return self.trainable_vars

