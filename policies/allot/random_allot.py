"""
This file contains the function for random allotment
Each territory held by the player has an equal probability of being selected
Output is a one-hot vector corresponding to which territory to place armies on

"""

import numpy as np
import random

global MAX_ARMIES #max armies per player

def random_allot(state_vector):
	"""
	Function for executing maximum battle success
	:param state_vector: np-array 1D vector of armies on territory
	:return action_vector: np-array 1D vector of edges to attack along
	"""


	T = len(state_vector)
	# Leaving edge_matrix in as a visualization
	# edge_matrix[row, col] = action_vector[row*T + col]
	# edge_matrix = np.zeros((T,T), dtype=int)
	action_vector = np.zeros(T, dtype=int)
	valid_terrs = []

	for terr in range(T):
		if state_vector[terr] > 0:
			valid_terrs.append(terr)

	allot_choice = random.choice(valid_terrs)
	action_vector[allot_choice] = 1

	return action_vector

	