"""
	An utility function to convert a legal Pacman state into a square colored image

	Author: Hanan Aharonof
"""

import numpy as np
from PIL import Image, ImageDraw

RECT_SIZE_IN_PIXELS = 5


def convert_frame(state):
	walls = state.layout.walls.data
	width = state.layout.width * RECT_SIZE_IN_PIXELS
	height = state.layout.height * RECT_SIZE_IN_PIXELS
	im = Image.new('RGB', (width, height), (0, 0, 0))
	dr = ImageDraw.Draw(im)
	draw_walls(walls, dr)
	draw_food(state.food.data, dr)
	draw_capsules(state.capsules, state.layout.height, dr)
	draw_agents(state.agentStates, state.layout.height, dr)
	vector = np.array(im.convert('L'))
	im.close()
	return vector


def draw_walls(walls, dr):
	size = len(walls[0]) * RECT_SIZE_IN_PIXELS
	for c_index, col in enumerate(walls):
		for r_index, row in enumerate(col):
			if row:
				tlCords = (c_index * RECT_SIZE_IN_PIXELS, size - r_index * RECT_SIZE_IN_PIXELS)
				brCords = (
					(c_index + 1) * RECT_SIZE_IN_PIXELS, size - (r_index + 1) * RECT_SIZE_IN_PIXELS)
				dr.rectangle((tlCords, brCords), fill='blue')


def draw_food(food, dr):
	size = len(food[0]) * RECT_SIZE_IN_PIXELS
	for c_index, col in enumerate(food):
		for r_index, row in enumerate(col):
			if row:
				tlCords = (c_index * RECT_SIZE_IN_PIXELS, size - r_index * RECT_SIZE_IN_PIXELS)
				brCords = (
					(c_index + 1) * RECT_SIZE_IN_PIXELS, size - (r_index + 1) * RECT_SIZE_IN_PIXELS)
				dr.rectangle((tlCords, brCords), fill='gray')


def draw_capsules(capsules, height, dr):
	for (i, j) in capsules:
		tlCords = (i * RECT_SIZE_IN_PIXELS, (height - 1 - j) * RECT_SIZE_IN_PIXELS)
		brCords = ((i + 1) * RECT_SIZE_IN_PIXELS, (height - j) * RECT_SIZE_IN_PIXELS)
		dr.rectangle((tlCords, brCords), fill='white')


def draw_agents(agents, height, dr):
	for agent in agents:
		i, j = agent.configuration.pos
		tlCords = (i * RECT_SIZE_IN_PIXELS, (height - 1 - j) * RECT_SIZE_IN_PIXELS)
		brCords = ((i + 1) * RECT_SIZE_IN_PIXELS, (height - j) * RECT_SIZE_IN_PIXELS)
		if agent.isPacman:
			dr.rectangle((tlCords, brCords), fill='yellow')
		else:
			if agent.scaredTimer == 0:
				dr.rectangle((tlCords, brCords), fill='orange')
			else:
				dr.rectangle((tlCords, brCords), fill='green')
