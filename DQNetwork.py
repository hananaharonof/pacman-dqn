import numpy as np
import tensorflow as tf

from dqnParameters import *


class DQNetwork:
	def __init__(self, params, session, name):
		self.params = params
		self.session = session
		self.name = name

		with tf.variable_scope(name):
			self.inputs = tf.placeholder(
				tf.float32,
				[None, params[FRAME_WIDTH], params[FRAME_HEIGHT], params[FRAME_STACK_SIZE]],
				name="inputs")
			self.actions = tf.placeholder(tf.float32, [None, self.params[NUM_OF_ACTIONS]], name="actions")
			self.target_q = tf.placeholder(tf.float32, [None], name="target")

			# Create the first convolution layer
			size = self.params[L2_FILTER_SIZE]
			depth = self.params[L2_DEPTH]
			filters = self.params[L2_FILTERS]
			stride = self.params[L2_STRIDE]

			w1 = tf.Variable(tf.random_normal([size, size, depth, filters], stddev=0.01), name='w1')
			b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name='b1')
			c1 = tf.nn.conv2d(self.inputs, w1, strides=[1, stride, stride, 1], padding="SAME", name='c1')
			conv_1 = tf.nn.relu(tf.add(c1, b1), name='conv1')

			# Create the second convolution layer
			size = self.params[L3_FILTER_SIZE]
			depth = self.params[L3_DEPTH]
			filters = self.params[L3_FILTERS]
			stride = self.params[L3_STRIDE]

			w2 = tf.Variable(tf.random_normal([size, size, depth, filters], stddev=0.01), name='w2')
			b2 = tf.Variable(tf.constant(0.1, shape=[filters]), name='b2')
			c2 = tf.nn.conv2d(conv_1, w2, strides=[1, stride, stride, 1], padding="SAME", name='c2')
			conv_2 = tf.nn.relu(tf.add(c2, b2), name='conv2')

			# Create the third convolution layer
			size = self.params[L4_FILTER_SIZE]
			depth = self.params[L4_DEPTH]
			filters = self.params[L4_FILTERS]
			stride = self.params[L4_STRIDE]

			w3 = tf.Variable(tf.random_normal([size, size, depth, filters], stddev=0.01), name='w3')
			b3 = tf.Variable(tf.constant(0.1, shape=[filters]), name='b3')
			c3 = tf.nn.conv2d(conv_2, w3, strides=[1, stride, stride, 1], padding="SAME", name='c3')
			conv_3 = tf.nn.relu(tf.add(c3, b3), name='conv3')

			# Create the fully connected layer
			fc_shape = conv_3.get_shape().as_list()
			fc_dimension = fc_shape[1] * fc_shape[2] * fc_shape[3]

			conv_3_flatted = tf.reshape(conv_3, [-1, fc_dimension])

			w_fc = tf.Variable(tf.random_normal([fc_dimension, self.params[FC_RECTIFIER_UNITS]], stddev=0.01), name='w_fc')
			b_fc = tf.Variable(tf.constant(0.1, shape=[self.params[FC_RECTIFIER_UNITS]]), name='b_fc')
			fc = tf.nn.relu(tf.add(tf.matmul(conv_3_flatted, w_fc), b_fc))

			# Create the second fully connected linear layer with [action] actions
			w_fc2 = tf.Variable(tf.random_normal([self.params[FC_RECTIFIER_UNITS], self.params[NUM_OF_ACTIONS]], stddev=0.01), name='w_fc2')
			b_fc2 = tf.Variable(tf.constant(0.01, shape=[self.params[NUM_OF_ACTIONS]]), name='b_fc2')
			self.output = tf.add(tf.matmul(fc, w_fc2), b_fc2)

			self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))
			self.loss = tf.reduce_mean(tf.square(self.target_q - self.Q))
			self.optimizer = tf.train.AdamOptimizer(self.params[OPTIMIZER_LEARNING_RATE]).minimize(self.loss)
			self.session.run(tf.global_variables_initializer())

	def predict(self, state):
		feed_dict = {self.inputs: state.reshape(
			(1, self.params[FRAME_WIDTH], self.params[FRAME_HEIGHT], self.params[FRAME_STACK_SIZE]))}
		return self.session.run(self.output, feed_dict=feed_dict)[0]

	def train(self, states_mb, actions_mb, rewards_mb, next_states_mb, terminals_mb):
		target_Qs_batch = []
		Qs_next_state = self.session.run(self.output, feed_dict={self.inputs: next_states_mb})

		for i in range(0, len(states_mb)):
			terminal = terminals_mb[i]
			if terminal:
				target_Qs_batch.append(rewards_mb[i])
			else:
				target = rewards_mb[i] + self.params[RL_DISCOUNT_FACTOR] * np.max(Qs_next_state[i])
				target_Qs_batch.append(target)

		targets_mb = np.array([each for each in target_Qs_batch])

		feed_dict = {
			self.inputs: states_mb,
			self.target_q: targets_mb,
			self.actions: actions_mb
		}
		loss, _ = self.session.run([self.loss, self.optimizer], feed_dict=feed_dict)
