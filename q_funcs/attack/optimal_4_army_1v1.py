"""
This file contains the function for attacks solely based on army difference
Based on the table on page 11 of http://c4i.gr/xgeorgio/docs/RISK-board-game%20_rev-3.pdf
Equal armies is a >50%  success chance for all matchups with armies >3
(-) armies is a -50% success chance for all matchups
So the pass-turn point comes when all borders are negative army difference

Important: Different from max_success action
Max_success will act differently than this policy for self army numbers < 4

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

class Optimal4Army1V1():
	"""
	Class to hold the army difference q function
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
		MAX_ARMIES = 4 # For testing

		# For prioritizing bad actions between player and opponent over player-player actions
		# Will be the threshold value at which the player will pass the turn
		army_offset = MAX_ARMIES + 1

		# Pass value is prioritized below 0 difference matchups
		pass_value = army_offset - 1

		for state in state_vector[0]:
			if abs(state) > 4:
				print("Policy only valid for 4 army, 1v1 board")
				exit()

		################ Updated code for new action space ################
		action_vector = np.zeros(2)
		# print("action vector is {}".format(action_vector))
		for act_index in range(1):              #len(self.act_list)):
			# if self.act_list[act_index][0] == -1:  # Only element with single value
			# 	action_vector[-1] = pass_value
			# else:
			terr_0_armies = state_vector[0,0]         #[0,self.act_list[act_index][0]]
			terr_1_armies = state_vector[0,1]         #,self.act_list[act_index][1]]
			if not (np.sign(terr_0_armies) == np.sign(terr_1_armies)):
				if terr_0_armies > 0:
					player_armies = abs(terr_0_armies)
					enemy_armies = abs(terr_1_armies)
				else:
					player_armies = abs(terr_1_armies)
					enemy_armies = abs(terr_0_armies)
				army_difference = player_armies - enemy_armies
				for ii in range(4):
					for jj in range(4):
						if (jj+1 == 1) and (not ii+1 == 1):
							if (player_armies == ii+1) and (enemy_armies == jj+1):
								action_vector[act_index] = army_offset + army_difference
						elif (ii+1 == 4) and (not jj+1 == 4):
							if player_armies == ii+1 and enemy_armies == jj+1:
								action_vector[act_index] = army_offset + army_difference


		action_vector[-1] = pass_value

		return action_vector

		