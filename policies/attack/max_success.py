"""
This file contains the function for attacks prioritized by highest chance of battle success
Based on the table on page 11 of http://c4i.gr/xgeorgio/docs/RISK-board-game%20_rev-3.pdf
Equal armies is a >50%  success chance for all matchups
(-) armies is a -50% success chance for all matchups
So the pass-turn point comes when all borders are negative army difference

Note: 
"Battle" refers to a sequence of attacks between two territories
"Attack" refers to a single engagement

This policy assumes a knowledge of the game 
i.e. not a policy for learning agents

State representation:
Vector of army numbers by territory, size Tx1
Positive numbers represent player armies
Negative numbers represent opponent armies (all opponents represented the same way)


Action representation:
Vector of edges for each connection
Size ((T^2) + 1)x1
Final element represents choice to end attack phase
Policy is naive to which edges are valid attacks

Effectively a flattened matrix of size TxT
Row corresponds to attacking territory
Column corresponds to defending territory
Element is difference of player and opponent armies

Environment will translate maxmimum valid element into attack in the game

"""

import numpy as np

global MAX_ARMIES #max armies per player

class MaxSuccess():
	"""
	Class to hold the maximum success policy
	"""
	def __init__(self, T):
		"""
		Empty constructor so MaxSuccess can be held as an object
		:params: none
		:return : none
		"""
		return

	def enact_policy(self, state_vector):
		"""
		Function for executing maximum battle success
		:param state_vector: np-array 1D vector of armies on territory
		:return action_vector: np-array 1D vector of edges to attack along
		"""

		# TODO: Remove this code once global armies number is defined
		MAX_ARMIES = 100 # For testing

		# For prioritizing bad actions between player and opponent over player-player actions
		# Will be the threshold value at which the player will pass the turn
		army_offset = MAX_ARMIES + 1

		# Pass value is prioritized below 0 difference matchups
		pass_value = army_offset - 1


		T = len(state_vector[0])
		# Leaving edge_matrix in as a visualization
		# edge_matrix[row, col] = action_vector[row*T + col]
		# edge_matrix = np.zeros((T,T), dtype=int)
		action_vector = np.zeros((T**2 + 1), dtype=int)


		# Can assume state_vector will be a (None, nS) length vector
		for terr_row in range(T):
			for terr_col in range(T):
				terr_row_armies = state_vector[0, terr_row]
				terr_col_armies = state_vector[0, terr_col]
				if np.sign(terr_row_armies) == np.sign(terr_col_armies): # Can't attack yourself!
					# edge_matrix[terr_row, terr_col] = 0
					action_vector[terr_row*T + terr_col] = 0
				elif terr_row_armies < 0: # Opponent territories
					# edge_matrix[terr_row, terr_col] = 0
					action_vector[terr_row*T + terr_col] = 0
				else:
					army_difference = abs(terr_row_armies) - abs(terr_col_armies)
					if army_difference < 0: # If worse than passing, decrement so less than pass_value
						# edge_matrix[terr_row, terr_col] = 0
						action_vector[terr_row*T + terr_col] = army_difference + army_offset -1
					else:
						# edge_matrix[terr_row, terr_col] = army_difference + army_offset
						action_vector[terr_row*T + terr_col] = army_difference + army_offset

		action_vector[-1] = pass_value

		return action_vector

		