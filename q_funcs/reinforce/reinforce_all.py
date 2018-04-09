"""
This file contains the Q function for reinforce_all


Env will determine how many troops to move
TODO: How to determine this?  Is there an optimal decision?

"""

import numpy as np
import random

global MAX_ARMIES #max armies per player

class ReinforceAll():
	"""
	Class to hold the maximum success policy
	"""
	def __init__(self, T, act_list):
		"""
		Constructor so RandomFortify can be held as an object
		:param T: int the length of the state vector
		:param act_list: 2D list mapping edges to territories
		:return : none
		"""
		self.T = T
		self.act_list = act_list
		return

	def call_Q(self, state_vector, update=None, action_taken=None, target=None, loss_weights=None):
		"""
		Function for executing maximum battle success
		:param state_vector: np-array 1D vector of armies on territory
		:return action_vector: np-array 1D vector of edges to attack along
		"""

		# This corresponds to moving as many armies as possible
		action = 1

		return action

		