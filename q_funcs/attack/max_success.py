"""
This file contains the function for attacks solely based on army difference
Based on the table on page 11 of http://c4i.gr/xgeorgio/docs/RISK-board-game%20_rev-3.pdf
Equal armies is a >50%  success chance for all matchups with armies >3
(-) armies is a -50% success chance for all matchups
So the pass-turn point comes when all borders are negative army difference



Important: Different from army_difference action
Army_difference will act differently than this strategy for self army numbers < 4

Note: 
"Battle" refers to a sequence of attacks between two territories
"Attack" refers to a single engagement

This q function assumes a knowledge of the game 
i.e. not a q function for learning agents

State representation:
Vector of army numbers by territory, size Tx1
Positive numbers represent player armies
Negative numbers represent opponent armies (all opponents represented the same way)


Action representation:
act_list is a 2D list with each element corresponding to an edge in the graph, 
holding the territories it connects
Final element in the list is -1, indicating pass turn action
size = # edges + 1

Environment will translate maxmimum valid element into attack in the game

"""

import numpy as np

global MAX_ARMIES #max armies per player

class MaxSuccess():
	"""
	Class to hold the maximum success q function
	"""
	def __init__(self, T, act_list):
		"""
		Constructor so ArmyDifference can be held as an object
		:param T: int, length of state, ignored
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

		# TODO: Remove this code once global armies number is defined
		MAX_ARMIES = 12 # For testing

		# For prioritizing bad actions between player and opponent over player-player actions
		# Will be the threshold value at which the player will pass the turn
		army_offset = MAX_ARMIES + 1

		# Pass value is prioritized below 0 difference matchups
		pass_value = army_offset - 1


		################ Updated code for new action space ################
		action_vector = np.zeros(len(self.act_list))
		for act_index in range(len(self.act_list)):
			if self.act_list[act_index][0] == -1:  # Only element with single value
				action_vector[-1] = pass_value
			else:
				terr_0_armies = state_vector[0,self.act_list[act_index][0]]
				terr_1_armies = state_vector[0,self.act_list[act_index][1]]
				if not (np.sign(terr_0_armies) == np.sign(terr_1_armies)):
					if terr_0_armies > 0:
						player_armies = terr_0_armies
						enemy_armies = terr_1_armies
					else:
						player_armies = terr_1_armies
						enemy_armies = terr_0_armies
					army_difference = abs(player_armies) - abs(enemy_armies)
					if player_armies > 3:  # Attacking with 3 armies
						if army_difference == 0:
							action_vector[act_index] = army_offset
						elif army_difference > 0:
							action_vector[act_index] = army_offset + army_difference
						elif army_difference < 0:
							action_vector[act_index] = army_offset + army_difference - 1
						else:
							print("Neither army is player, exiting")
							exit()
					elif player_armies == 3:  # Don't attack 2 against 2
						if (army_difference == 0) or (army_difference == 1):
							action_vector[act_index] = pass_value + army_difference - 2
						elif army_difference > 2:  # Attack 2 against 1
							action_vector[act_index] = army_offset + army_difference
						elif army_difference < 0:
							action_vector[act_index] = army_offset + army_difference - 1
					elif player_armies == 2:  # Do not attack with only 2 armies
						if army_difference == 0:
							action_vector[act_index] = pass_value - 2
						elif army_difference > 0:
							action_vector[act_index] = pass_value - 1
						elif army_difference < 0:
							action_vector[act_index] = army_offset + army_difference - 1
						else:
							print("You broke math, exiting")
				####### No 1 - army case needed because attacks will not be valid ###


		action_vector[-1] = pass_value

		return action_vector

		