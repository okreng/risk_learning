"""
This file contains the Q function for random allotment
Each territory held by the player has an equal probability of being selected

"""

import numpy as np
import random

##### Working from root directory #####
import repackage
repackage.up(2)
import utils
##### End Working from root directory #####

global MAX_ARMIES #max armies per player

class RandomAllot():
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
		self.act_list = act_list
		return

	def call_Q(self, state_vector, update=None, action_taken=None, target=None, loss_weights=None):
		"""
		Function for executing maximum battle success
		:param state_vector: np-array 1D vector of armies on territory
		:return action_vector: np-array 1D vector of edges to attack along
		"""


		# T = len(state_vector[0])
		# Leaving edge_matrix in as a visualization
		# edge_matrix[row, col] = action_vector[row*T + col]
		# edge_matrix = np.zeros((T,T), dtype=int)
		
		# action_vector = np.random.rand(self.T)	

		# Code below will work but is unnecessary
		# Since action_vector can be naive to game validity, a random vector will suffice
		# action_vector = np.zeros(T, dtype=int)
		# valid_terrs = []
		# for terr in range(T):
		# 	if state_vector[terr] > 0:
		# 		valid_terrs.append(terr)

		# allot_choice = random.choice(valid_terrs)
		# action_vector[allot_choice] = 1

		############ Code for new action space ##############3
		action_vector = np.random.rand(self.T)
		return action_vector

	def get_action(self, state_vector, valid_mask, update=None, action_taken=None, target=None, loss_weights=None):
		"""
		Chooses an action based on the state vector and valid_mask inputted
		"""
		q = self.call_Q(state_vector)
		valid_q = utils.validate_q_func_for_argmax(q, valid_mask)
		action = utils.epsilon_greedy_valid(q, valid_mask, 0)
		return action
