"""
	A utility function to convert a legal Pac-Man state into a pixel colored image

	Author: Hanan Aharonof
"""

import numpy as np
from PIL import Image, ImageDraw


def _draw_walls(walls, dr):
	for c_index, col in enumerate(walls):
		for r_index, row in enumerate(col):
			if row:
				dr.point((c_index, r_index), fill='blue')


def _draw_food(food, dr):
	for c_index, col in enumerate(food):
		for r_index, row in enumerate(col):
			if row:
				dr.point((c_index, r_index), fill='gray')


def _draw_capsules(capsules, dr):
	for (i, j) in capsules:
		dr.point((i, j), fill='white')


def _draw_agents(agents, dr):
	for agent in agents:
		i, j = agent.configuration.pos
		if agent.isPacman:
			dr.point((i, j), fill='yellow')
		else:
			if agent.scaredTimer == 0:
				dr.point((i, j), fill='orange')
			else:
				dr.point((i, j), fill='green')


def convert_frame(state):
	walls = state.layout.walls.data
	width = state.layout.width
	height = state.layout.height
	im = Image.new('RGB', (width, height), (0, 0, 0))
	dr = ImageDraw.Draw(im)
	_draw_walls(walls, dr)
	_draw_food(state.food.data, dr)
	_draw_capsules(state.capsules, dr)
	_draw_agents(state.agentStates, dr)
	vector = np.array(im.convert('L'))/255.0
	vector = np.swapaxes(vector, 0, 1)
	im.close()
	return vector
