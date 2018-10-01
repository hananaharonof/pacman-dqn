from DQN import DQN
from deepQNetwork import *
from frameConvertor import getStateMatrices
from game import Agent, Directions
from loggingUtils import debug
from objectMapper import load
from replayMemory import ReplayMemory, REPLAY_MEMORY_EXT


def to_action(action_index):
	if action_index == 0:
		return Directions.WEST
	if action_index == 1:
		return Directions.EAST
	if action_index == 2:
		return Directions.NORTH
	if action_index == 3:
		return Directions.SOUTH

	raise Exception('Action index %d is illegal' % action_index)


def to_action_index(action):
	if action == Directions.WEST:
		return 0
	if action == Directions.EAST:
		return 1
	if action == Directions.NORTH:
		return 2
	if action == Directions.SOUTH:
		return 3

	raise Exception('Action %s is illegal' % action)


def _init_dqn_params(args):
	if MODEL not in args:
		info("Using default DQN parameters.")
		params = DQNParameters()
	else:
		params = load(args[MODEL], DQN_PARAMETERS_EXT)
		if params is None:
			info("Using default DQN parameters.")
			params = DQNParameters()
		else:
			info("Successfully loaded saved parameters of model %s." % args[MODEL])

	params[FRAME_WIDTH] = args[FRAME_WIDTH]  # * params[FRAME_CONVERTOR_FACTOR] TODO
	params[FRAME_HEIGHT] = args[FRAME_HEIGHT]  # * params[FRAME_CONVERTOR_FACTOR] TODO
	params[LAYOUT] = args[LAYOUT]
	params[MODEL] = args[MODEL]
	return params


def _init_replay_memory(args):
	if MODEL not in args:
		info("Using empty replay memory.")
		return ReplayMemory(DEFAULT_PARAMS[REPLAY_MEMORY_SIZE])

	replay_memory = load(args[MODEL], REPLAY_MEMORY_EXT)
	if replay_memory is None:
		info("Using empty replay memory.")
		return ReplayMemory(DEFAULT_PARAMS[REPLAY_MEMORY_SIZE])
	info("Successfully loaded saved replay memory of model %s." % args[MODEL])
	return replay_memory


class DQNAgent(Agent):
	def __init__(self, args):
		info("Initializing DQN Agent...")
		self.params = _init_dqn_params(args)
		self.replay_memory = _init_replay_memory(args)

		self.dqn = DQN(self.params)#DeepQNetwork(self.params)

		self.current_state = None
		self.last_state = None
		self.frame_stack = None
		self.last_action = None
		self.last_score = None
		self.last_reward = None
		self.terminal_state = None
		self.won = None
		self.best_q = np.nan

		debug("Using the following parameters: ")
		for param in self.params.keys():
			debug("\t%s:%s" % (param, self.params[param]))
		info("Done initializing DQN Agent.")

	def observationFunction(self, state):
		if self.last_action is not None:
			self._update_score(state)

			# TODO: remove
			self.last_state = np.copy(self.current_state)
			self.current_state = getStateMatrices(state)
			# self._update_frame_stack(state)

			self.replay_memory.add_memory(
				self.last_state, self.last_action, self.last_reward, self.current_state, self.terminal_state)

			if self._should_train():
				states, actions, rewards, new_states, terminals = \
					self.replay_memory.sample(self.params[REPLAY_MEMORY_SAMPLE_BATCH_SIZE])
				self.dqn.train_network(states, actions, rewards, new_states, terminals)

		self.params[FRAMES] += 1
		self.params[RL_EPSILON_CURRENT] = max(
			self.params[RL_EPSILON_END],
			self.params[RL_EPSILON_START] - float(self.params[FRAMES]) / float(self.params[RL_EPSILON_FRAMES_DECAY]))

		if self._should_train() and self._should_save_model():
			model = "%s_%s" % (self.params[LAYOUT], get_time())
			info("Saving model [%s]..." % model)
			self.params.save(model)
			self.replay_memory.save(model)
			self.dqn.save_network(model)

		return state

	def _should_train(self):
		return self.params[FRAMES] > self.params[FRAMES_BEFORE_TRAINING]

	def _should_save_model(self):
		return self.params[FRAMES] % self.params[MODEL_SAVE_INTERVAL_IN_FRAMES] == 0

	def final(self, state):
		self.terminal_state = True
		self.params[EPISODES] += 1
		self.observationFunction(state)
		if self.won:
			self.params[WINS] += 1
		info(
			"Episode #%d | Frames: %d | Wins: %d | Won: %s | Score: %d | Epsilon: %5f | Q: %5f" % (
				self.params[EPISODES],
				self.params[FRAMES],
				self.params[WINS],
				self.won,
				state.getScore(),
				self.params[RL_EPSILON_CURRENT],
				self.best_q))

	def registerInitialState(self, state):  # Called with each new game
		# self.frame_stack = None
		# self._update_frame_stack(state)
		# TODO
		self.last_state = None
		self.current_state = getStateMatrices(state)
		self.last_score = 0
		self.last_reward = 0
		self.terminal_state = False
		self.won = True
		self.best_q = np.nan

	def getAction(self, state):
		if np.random.rand() > self.params[RL_EPSILON_CURRENT]:
			prediction = self.dqn.predict(
				self.current_state.reshape(1, self.params[FRAME_WIDTH], self.params[FRAME_HEIGHT], self.params[FRAME_STACK_SIZE]))
			a_winner = np.argwhere(prediction == np.amax(prediction))
			self.update_max_q(prediction[0][a_winner[0][1]])
			if len(a_winner) > 1:
				move = to_action(a_winner[np.random.randint(0, len(a_winner))][1])
			else:
				move = to_action(a_winner[0][1])
		else:
			move = to_action(np.random.randint(0, 4))
			#move = legal_actions[np.random.randint(0, len(legal_actions))]

		self.last_action = self._to_action_vector(to_action_index(move))

		if move not in state.getLegalActions(0):
			move = Directions.STOP

		return move
		# legal_actions = state.getLegalActions(0)
		# q = np.nan
		# if np.random.rand() > self.params[RL_EPSILON_CURRENT]:
		# 	prediction = self.dqn.predict(
		# 		self.current_state.reshape(1, self.params[FRAME_WIDTH], self.params[FRAME_HEIGHT], self.params[FRAME_STACK_SIZE]))
		# 	ordered_prediction = np.argsort(-prediction)
		# 	selected_action = to_action(ordered_prediction[0][0])
		# 	q = prediction[0][ordered_prediction[0][0]]
		# else:
		# 	if Directions.STOP in legal_actions:
		# 		legal_actions.remove(Directions.STOP)
		# 	selected_action_index = np.random.randint(0, len(legal_actions))
		# 	selected_action = legal_actions[selected_action_index]
		#
		# self.last_action = self._to_action_vector(to_action_index(selected_action))
		# if selected_action in legal_actions:
		# 	if self._should_train():
		# 		self.update_max_q(q)
		# 	return selected_action
		# return Directions.STOP

	def update_max_q(self, q):
		if np.isnan(self.best_q):
			self.best_q = q
		elif q > self.best_q:
			self.best_q = q

	#
	# def _update_frame_stack(self, state):
	# 	frame = convert_frame(state.data)
	# 	if self.frame_stack is None:
	# 		self.frame_stack = np.stack((frame for i in xrange(0, self.params[M])))
	# 	else:
	# 		self.frame_stack = np.insert(self.frame_stack, 0, frame, axis=0)
	# 		self.frame_stack = np.delete(self.frame_stack, self.params[M], axis=0)
	#
	# 	self.last_state = self.current_state
	# 	self.current_state = np.swapaxes(
	# 		self.frame_stack, 0, 2)
	# 	if self.last_state is None:
	# 		self.last_state = self.current_state

	def _update_score(self, state):
		current_score = state.getScore()
		reward = current_score - self.last_score
		self.last_score = current_score
		if reward > 20:
			self.last_reward = 50
		elif reward > 0:
			self.last_reward = 10
		elif reward < -10:
			self.won = False
			self.last_reward = -500
		elif reward < 0:
			self.last_reward = -1

		if self.terminal_state and self.won:
			self.last_reward = 100

	def _to_action_vector(self, index):
		vector = np.zeros(self.params[NUM_OF_ACTIONS])
		vector[index] = 1
		return vector
