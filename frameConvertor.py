"""
	An utility function to convert a legal Pacman state into a square colored image

	Author: Hanan Aharonof
"""

import numpy as np
from PIL import Image, ImageDraw


class FrameConvertor(object):
	def __init__(self, factor):
		self.factor = factor

	def convert_frame(self, state):
		walls = state.layout.walls.data
		width = state.layout.width * self.factor
		height = state.layout.height * self.factor
		im = Image.new('RGB', (width, height), (0, 0, 0))
		dr = ImageDraw.Draw(im)
		self._draw_walls(walls, dr)
		self._draw_food(state.food.data, dr)
		self._draw_capsules(state.capsules, state.layout.height, dr)
		self._draw_agents(state.agentStates, state.layout.height, dr)
		vector = np.array(im.convert('L'))/255.0
		im.close()
		return vector

	def _draw_walls(self, walls, dr):
		size = len(walls[0]) * self.factor
		for c_index, col in enumerate(walls):
			for r_index, row in enumerate(col):
				if row:
					# tlCords = (c_index * self.factor, size - r_index * self.factor)
					# brCords = (
					# 	(c_index + 1) * self.factor, size - (r_index + 1) * self.factor)
					# dr.rectangle((tlCords, brCords), fill='blue')
					dr.point((c_index, r_index), fill='blue')


	def _draw_food(self, food, dr):
		size = len(food[0]) * self.factor
		for c_index, col in enumerate(food):
			for r_index, row in enumerate(col):
				if row:
					# tlCords = (c_index * self.factor, size - r_index * self.factor)
					# brCords = (
					# 	(c_index + 1) * self.factor, size - (r_index + 1) * self.factor)
					# dr.rectangle((tlCords, brCords), fill='gray')
					dr.point((c_index, r_index), fill='gray')

	def _draw_capsules(self, capsules, height, dr):
		for (i, j) in capsules:
			# tlCords = (i * self.factor, (height - 1 - j) * self.factor)
			# brCords = ((i + 1) * self.factor, (height - j) * self.factor)
			# dr.rectangle((tlCords, brCords), fill='white')
			dr.point((i, j), fill='white')

	def _draw_agents(self, agents, height, dr):
		for agent in agents:
			i, j = agent.configuration.pos
			tlCords = (i * self.factor, (height - 1 - j) * self.factor)
			brCords = ((i + 1) * self.factor, (height - j) * self.factor)
			if agent.isPacman:
				# dr.rectangle((tlCords, brCords), fill='yellow')
				dr.point((i, j), fill='yellow')
			else:
				if agent.scaredTimer == 0:
					# dr.rectangle((tlCords, brCords), fill='orange')
					dr.point((i, j), fill='orange')
				else:
					# dr.rectangle((tlCords, brCords), fill='green')
					dr.point((i, j), fill='green')


def getStateMatrices(state):
	""" Return wall, ghosts, food, capsules matrices """

	def getWallMatrix(state):
		""" Return matrix with wall coordinates set to 1 """
		width, height = state.data.layout.width, state.data.layout.height
		grid = state.data.layout.walls
		matrix = np.zeros((height, width), dtype=np.int8)
		for i in range(grid.height):
			for j in range(grid.width):
				# Put cell vertically reversed in matrix
				cell = 1 if grid[j][i] else 0
				matrix[-1 - i][j] = cell
		return matrix

	def getPacmanMatrix(state):
		""" Return matrix with pacman coordinates set to 1 """
		width, height = state.data.layout.width, state.data.layout.height
		matrix = np.zeros((height, width), dtype=np.int8)

		for agentState in state.data.agentStates:
			if agentState.isPacman:
				pos = agentState.configuration.getPosition()
				cell = 1
				matrix[-1 - int(pos[1])][int(pos[0])] = cell

		return matrix

	def getGhostMatrix(state):
		""" Return matrix with ghost coordinates set to 1 """
		width, height = state.data.layout.width, state.data.layout.height
		matrix = np.zeros((height, width), dtype=np.int8)

		for agentState in state.data.agentStates:
			if not agentState.isPacman:
				if not agentState.scaredTimer > 0:
					pos = agentState.configuration.getPosition()
					cell = 1
					matrix[-1 - int(pos[1])][int(pos[0])] = cell

		return matrix

	def getScaredGhostMatrix(state):
		""" Return matrix with ghost coordinates set to 1 """
		width, height = state.data.layout.width, state.data.layout.height
		matrix = np.zeros((height, width), dtype=np.int8)

		for agentState in state.data.agentStates:
			if not agentState.isPacman:
				if agentState.scaredTimer > 0:
					pos = agentState.configuration.getPosition()
					cell = 1
					matrix[-1 - int(pos[1])][int(pos[0])] = cell

		return matrix

	def getFoodMatrix(state):
		""" Return matrix with food coordinates set to 1 """
		width, height = state.data.layout.width, state.data.layout.height
		grid = state.data.food
		matrix = np.zeros((height, width), dtype=np.int8)

		for i in range(grid.height):
			for j in range(grid.width):
				# Put cell vertically reversed in matrix
				cell = 1 if grid[j][i] else 0
				matrix[-1 - i][j] = cell

		return matrix

	def getCapsulesMatrix(state):
		""" Return matrix with capsule coordinates set to 1 """
		width, height = state.data.layout.width, state.data.layout.height
		capsules = state.data.layout.capsules
		matrix = np.zeros((height, width), dtype=np.int8)

		for i in capsules:
			# Insert capsule cells vertically reversed into matrix
			matrix[-1 - i[1], i[0]] = 1

		return matrix

	# Create observation matrix as a combination of
	# wall, pacman, ghost, food and capsule matrices
	# width, height = state.data.layout.width, state.data.layout.height
	width, height = state.data.layout.width, state.data.layout.height
	observation = np.zeros((6, height, width))

	observation[0] = getWallMatrix(state)
	observation[1] = getPacmanMatrix(state)
	observation[2] = getGhostMatrix(state)
	observation[3] = getScaredGhostMatrix(state)
	observation[4] = getFoodMatrix(state)
	observation[5] = getCapsulesMatrix(state)

	observation = np.swapaxes(observation, 0, 2)

	return observation
