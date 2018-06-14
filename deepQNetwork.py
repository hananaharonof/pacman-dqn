"""
An implementation of a deep Q network as described
at https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Author: Hanan Aharonof
"""

import tensorflow as tf
from loggingUtils import info
from replayMemory import ReplayMemory

SAVED_NETWORKS_FOLDER_NAME = "saved_networks"


class DeepQNetwork(object):
	def __init__(self, params):
		self.width = params['width']  # Layout width in pixels
		self.height = params['height']  # Layout height in pixels
		self.m = params['m']  # The number of most recent frames to stack
		self.actions = params['actions']  # The number of actions
		self.replay_memory_size = params['replay_memory_size']  # Replay memory size
		self.discount_factor = tf.constant(params['discount_factor'])

		self.tf_session = tf.InteractiveSession()
		self.replay_memory = self._init_replay_memory()
		self.input, self.output = self._init_network()
		self.cost_function = self._init_cost_function()

	def _init_cost_function(self):
		# Bellman equation
		yj = tf.add(self.rewards, tf.multiply(1.0 - self.terminals, tf.multiply(self.discount_factor, self.q_t)))
		# gives Q with action taken into consideration
		self.Q_pred = tf.reduce_sum(tf.multiply(self.y, self.actions), reduction_indices=1)
		# Loss function
		self.cost = tf.reduce_sum(tf.square(tf.subtract(self.yj, self.Q_pred)))

	def _init_replay_memory(self):
		memory = ReplayMemory(self.replay_memory_size)
		memory.load()
		return memory

	def _init_network(self):
		input_layer, output_layer = self._create_network()
		self._load_network()
		return input_layer, output_layer

	def _create_network(self):
		# Create the input layer (1'st)
		l1 = tf.placeholder("float", [None, self.width, self.height, self.m], name='l1')

		# Create the 2'nd layer as a convolution layer with 32 filters of size 8x8, depth [m] and stride 4
		w1 = tf.Variable(tf.random_normal([8, 8, self.m, 32], stddev=0.01), name='w1')
		b1 = tf.Variable(tf.constant(0.01, shape=[32]), name='b1')
		c1 = tf.nn.conv2d(l1, w1, strides=[1, 4, 4, 1], padding="SAME", name='c1')
		l2 = tf.nn.relu(tf.add(c1 + b1), name='l2')

		# Create the 3'rd layer as a convolution layer with 64 filters of size 4x4, depth 32 and stride 2
		w2 = tf.Variable(tf.random_normal([4, 4, 32, 64], stddev=0.01), name='w2')
		b2 = tf.Variable(tf.constant(0.01, shape=[64]), name='b2')
		c2 = tf.nn.conv2d(l2, w2, strides=[1, 2, 2, 1], padding="SAME", name='c2')
		l3 = tf.nn.relu(tf.add(c2 + b2), name='l3')

		# Create the 4'th layer as a convolution layer with 64 filters of size 3x3, depth 64 and stride 1
		w3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01), name='w3')
		b3 = tf.Variable(tf.constant(0.01, shape=[64]), name='b3')
		c3 = tf.nn.conv2d(l3, w3, strides=[1, 1, 1, 1], padding="SAME", name='c3')
		l4 = tf.nn.relu(tf.add(c3 + b3), name='l4')

		# Create the 5'th layer as a full connected layer with 512 rectifier units
		l4_shape = l4.get_shape().as_list()
		l5_dimension = l4_shape[1] * l4_shape[2] * l4_shape[3]
		l4_flatted = tf.reshape(l4, [-1, l5_dimension])
		w4 = tf.Variable(tf.random_normal([l5_dimension, 512], stddev=0.01), name='w4')
		b4 = tf.Variable(tf.constant(0.01, shape=[512]), name='b4')
		l5 = tf.nn.relu(tf.matmul(l4_flatted, w4) + b4)

		# Create the 6'th layer as fully connected linear layer with [action] actions
		w5 = tf.Variable(tf.random_normal([512, self.actions], stddev=0.01), name='w5')
		b5 = tf.Variable(tf.constant(0.01, shape=[self.actions]), name='b5')
		l6 = tf.matmul(l5, w5) + b5

		return l1, l6  # Return first and last layers

	def _load_network(self):
		self.tf_session.run(tf.initialize_all_variables())
		saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORKS_FOLDER_NAME)
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(self.tf_session, checkpoint.model_checkpoint_path)
			info("Successfully loaded network weights %s." % checkpoint.model_checkpoint_path)
		else:
			info("No saved network weights found. Using random values.")

	def train_network(self):
		pass
