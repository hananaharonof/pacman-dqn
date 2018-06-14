from game import Agent, Directions


class DummyAgent(Agent):
	def __init__(self):
		print("Initialise DQN Agent")

	def observationFunction(self, state):
		print("observation")

	def final(self, state):
		print("final")

	def registerInitialState(self, state):  # inspects the starting state
		print("register")

	def getAction(self, state):
		print("action")
		return Directions.STOP