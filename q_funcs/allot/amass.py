"""
This file contains the Q function for alloting based on army size
The q function here is intended to be used with a np.random.choice() call
Rather than an epsilon-greedy policy
Each territory held by the player has an equal probability of being selected

"""

import numpy as np
import random

global MAX_ARMIES #max armies per player

class Amass():
	"""
	Class to hold the maximum success Q function
	"""
	def __init__(self, T, act_list):
		"""
		Constructor so RandomAllot can be held as an object
		:param T: int the length of the state vector
		:param act_list: 2D list mapping edges to territories
		:return : none
		"""
		self.T = T

		# Note: act_list unused by allot policies

		return

	def call_Q(self, state_vector, update=None, action_taken=None, target=None, loss_weights=None):
		"""
		Function for executing maximum battle success
		:param state_vector: np-array 1D vector of armies on territory
		:return action_vector: np-array 1D vector of edges to attack along
		"""

		action_vector = np.zeros(self.T)
		for terr in range(self.T):
			if state_vector[0][terr] > 0:
				action_vector[terr] = state_vector[0][terr]
			# Negatives left as zero for opposing player

		return action_vector

		