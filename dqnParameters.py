from objectMapper import ObjectMapper
from loggingUtils import error

# Pacman single frame params

FRAME_WIDTH = 'frame_width'
FRAME_HEIGHT = 'frame_height'
FRAME_STACK_SIZE = 'frame_stack_size'
FRAME_CONVERTOR_FACTOR = 'frame_convertor_factor'
LAYOUT = 'layout'

NUM_OF_ACTIONS = 'actions'

# Reinforcement learning params

RL_DISCOUNT_FACTOR = 'rl_discount_factor'
RL_EPSILON_START = 'rl_epsilon_start'
RL_EPSILON_CURRENT = 'rl_epsilon_current'
RL_EPSILON_END = 'rl_epsilon_end'
RL_EPSILON_FRAMES_DECAY = 'rl_epsilon_frames_decay'

# Replay Memory
REPLAY_MEMORY_SIZE = 'replay_memory_size'
REPLAY_MEMORY_SAMPLE_BATCH_SIZE = 'replay_memory_sample_batch_size'

# Network params
OPTIMIZER_LEARNING_RATE = 'optimizer_rms_learning_rate'
MODEL_SAVE_INTERVAL_IN_FRAMES = 'model_save_interval_in_frames'
TARGET_MODEL_UPDATE_INTERVAL_IN_FRAMES = 'target_model_update_interval_in_frames'
MODEL = 'model'
L2_FILTER_SIZE = 'l2_filter_size'
L2_FILTERS = 'l2_filters'
L2_DEPTH = 'l2_depth'
L2_STRIDE = 'l2_stride'
L3_FILTER_SIZE = 'l3_filter_size'
L3_FILTERS = 'l3_filters'
L3_DEPTH = 'l3_depth'
L3_STRIDE = 'l3_stride'
L4_FILTER_SIZE = 'l4_filter_size'
L4_FILTERS = 'l4_filters'
L4_DEPTH = 'l4_depth'
L4_STRIDE = 'l4_stride'
FC_RECTIFIER_UNITS = 'l4_rectifier_units'

# Training progress
GLOBAL_STEP = 'global_step'
FRAMES_BEFORE_TRAINING = 'frames_before_training'
EPISODES = 'episodes'
FRAMES = 'frames'
WINS = 'wins'

DEFAULT_PARAMS = {
	FRAME_STACK_SIZE: 6,#4, TODO
	FRAME_CONVERTOR_FACTOR: 2,
	NUM_OF_ACTIONS: 4,
	RL_DISCOUNT_FACTOR: 0.95,
	RL_EPSILON_START: 1.0,
	RL_EPSILON_CURRENT: 1.0,
	RL_EPSILON_END: 0.1,
	RL_EPSILON_FRAMES_DECAY: 10000,
	FRAMES_BEFORE_TRAINING: 5000,
	OPTIMIZER_LEARNING_RATE: 0.0002,
	REPLAY_MEMORY_SIZE: 100000,
	REPLAY_MEMORY_SAMPLE_BATCH_SIZE: 32,
	MODEL_SAVE_INTERVAL_IN_FRAMES: 1000000,
	TARGET_MODEL_UPDATE_INTERVAL_IN_FRAMES: 100,
	L2_FILTER_SIZE: 3,
	L2_FILTERS: 8,#16,
	L2_DEPTH: 6,#4,  # Correlated to FRAME_STACK_SIZE TODO
	L2_STRIDE: 1,
	L3_FILTER_SIZE: 3,
	L3_FILTERS: 16,#32,
	L3_DEPTH: 8,#16,
	L3_STRIDE: 1,
	L4_FILTER_SIZE: 4,
	L4_FILTERS: 32,
	L4_DEPTH: 16,
	L4_STRIDE: 1,
	FC_RECTIFIER_UNITS: 256,
	MODEL: None,
	GLOBAL_STEP: 0,
	EPISODES: 0,
	FRAMES: 0,
	WINS: 0
}

DQN_PARAMETERS_EXT = 'params'


class DQNParameters(ObjectMapper):
	def __init__(self, init_params=DEFAULT_PARAMS):
		ObjectMapper.__init__(self, DQN_PARAMETERS_EXT)
		self.params = init_params

	def __getitem__(self, item):
		if item in self.params:
			return self.params[item]
		error('Parameter %s does not exist.')
		return None

	def __setitem__(self, key, value):
		self.params[key] = value

	def keys(self):
		return self.params.keys()
