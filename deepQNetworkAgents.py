from cappedMovingAverage import CappedMovingAverage
from deepQNetwork import *
from frameConvertor import convert_frame
from frameStack import FrameStack
from game import Agent, Directions
from loggingUtils import debug, info
from objectMapper import get_time, load
from replayMemory import REPLAY_MEMORY_EXT, ReplayMemory


def to_action(action_index):
	if action_index == 0:
		return Directions.WEST
	if action_index == 1:
		return Directions.EAST
	if action_index == 2:
		return Directions.NORTH
	if action_index == 3:
		return Directions.SOUTH


def to_action_index(action):
	if action == Directions.WEST:
		return 0
	if action == Directions.EAST:
		return 1
	if action == Directions.NORTH:
		return 2
	if action == Directions.SOUTH:
		return 3


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

	params[FRAME_WIDTH] = args[FRAME_WIDTH]
	params[FRAME_HEIGHT] = args[FRAME_HEIGHT]
	params[LAYOUT] = args[LAYOUT]
	params[MODEL] = args[MODEL]

	debug("Using the following parameters: ")
	for param in params.keys():
		debug("\t%s:%s" % (param, params[param]))
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
		Agent.__init__(self)

		info("Initializing DQN Agent...")
		tf.reset_default_graph()
		self.session = tf.Session()

		self.params = _init_dqn_params(args)
		self.replay_memory = _init_replay_memory(args)
		self.frame_stack = FrameStack(self.params[FRAME_STACK_SIZE], self.params[FRAME_WIDTH], self.params[FRAME_HEIGHT])
		self.dqn = DeepQNetwork(self.params, self.session, 'online')
		self.run_id = get_time()

		self.first_move = True
		self.current_state = None
		self.last_state = None
		self.last_action = None
		self.last_score = None
		self.last_reward = None
		self.ep_reward = None
		self.terminal_state = None
		self.won = None
		self.best_q = np.nan
		self.last_100_wins_avg = CappedMovingAverage(100)
		self.last_100_reward_avg = CappedMovingAverage(100)

		info("Done initializing DQN Agent.")

	def registerInitialState(self, state):  # Called with each new game
		self._update_frame_stack(state)

		self.first_move = True
		self.last_score = 0
		self.last_reward = 0
		self.ep_reward = 0
		self.terminal_state = False
		self.won = True
		self.best_q = np.nan

	def observationFunction(self, state):
		if not self.first_move:
			self._update_score(state)
			self._update_frame_stack(state)

			self.replay_memory.add((
				self.last_state,
				self.last_action,
				self.last_reward,
				self.current_state,
				self.terminal_state))

			if self._should_train():
				self.dqn.estimate_q_values_and_train(*self._sample_mb())

		self.params[FRAMES] += 1
		self.params[RL_EPSILON_CURRENT] = max(
			self.params[RL_EPSILON_END],
			min(
				self.params[RL_EPSILON_START] - float(self.params[FRAMES] - self.params[FRAMES_BEFORE_TRAINING]) / float(self.params[RL_EPSILON_FRAMES_DECAY]),
				self.params[RL_EPSILON_START]
			))

		self._save_model()
		return state

	def _sample_mb(self):
		batch = self.replay_memory.sample(self.params[REPLAY_MEMORY_SAMPLE_BATCH_SIZE])
		states_mb = []
		actions_mb = []
		rewards_mb = []
		next_states_mb = []
		terminals_mb = []
		for i in xrange(len(batch)):
			states_mb.append(batch[i][0])
			actions_mb.append(batch[i][1])
			rewards_mb.append(batch[i][2])
			next_states_mb.append(batch[i][3])
			terminals_mb.append(batch[i][4])
		states_mb = np.array(states_mb)
		actions_mb = np.array(actions_mb)
		rewards_mb = np.array(rewards_mb)
		next_states_mb = np.array(next_states_mb)
		terminals_mb = np.array(terminals_mb)
		return states_mb, actions_mb, rewards_mb, next_states_mb, terminals_mb

	def _save_model(self):
		if self._should_train() and self._should_save_model():
			model = "%s_%s" % (self.params[LAYOUT], self.run_id)
			info("Saving model [%s]..." % model)
			self.params.save(model)
			self.replay_memory.save(model)
			#self.dqn.save(model) TODO

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
			self.last_100_wins_avg.add(1)
		else:
			self.last_100_wins_avg.add(0)
		self.last_100_reward_avg.add(self.ep_reward)

		info(
			"Episode #%d | Frames: %d | Wins: %d | Won: %s | Score: %d | Epsilon: %5f | Q: %5f | 100 Wins Avg: %3f | 100 Reward Avg: %3f" % (
				self.params[EPISODES],
				self.params[FRAMES],
				self.params[WINS],
				self.won,
				state.getScore(),
				self.params[RL_EPSILON_CURRENT],
				self.best_q,
				self.last_100_wins_avg.avg(),
				self.last_100_reward_avg.avg()))

	def getAction(self, state):
		self.first_move = False
		if np.random.rand() > self.params[RL_EPSILON_CURRENT]:
			move = self.predict_action()
		else:
			move = self.random_action(state)

		self.last_action = self._to_action_vector(to_action_index(move))
		if move not in state.getLegalActions(0):
			move = Directions.STOP
		return move

	def random_action(self, state):
		legal_actions = np.copy(state.getLegalActions(0))
		if Directions.STOP in legal_actions:
			legal_actions = np.delete(legal_actions, np.where(legal_actions == Directions.STOP)[0][0])
		move = legal_actions[np.random.randint(0, len(legal_actions))]
		return move

	def predict_action(self):
		prediction = self.dqn.predict(self.current_state)
		best_actions = np.argwhere(prediction == np.amax(prediction))
		self.update_max_q(prediction[best_actions[0][0]])
		if len(best_actions) > 1:
			move = to_action(best_actions[np.random.randint(0, len(best_actions))][0])
		else:
			move = to_action(best_actions[0][0])

		return move

	def update_max_q(self, q):
		if np.isnan(self.best_q):
			self.best_q = q
		elif q > self.best_q:
			self.best_q = q

	def _update_frame_stack(self, state):
		frame = convert_frame(state.data)
		if self.first_move:
			self.frame_stack.reset(frame)
			self.last_state = np.zeros((self.params[FRAME_WIDTH], self.params[FRAME_HEIGHT], self.params[FRAME_STACK_SIZE]))
		else:
			self.frame_stack.add(frame)
			self.last_state = self.current_state
			self.current_state = self.frame_stack.get_stack()
			if self.last_state is None:
				self.last_state = np.zeros(
					(self.params[FRAME_WIDTH], self.params[FRAME_HEIGHT], self.params[FRAME_STACK_SIZE]))

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

		self.ep_reward += self.last_reward

	def _to_action_vector(self, index):
		vector = np.zeros(self.params[NUM_OF_ACTIONS])
		vector[index] = 1
		return vector

