"""
This file provides the environment for a player to interact with the Risk game board
# TODO: smart importing
"""

import tensorflow as tf
import numpy as np
import random
import time
import matplotlib as plot
import sys
import argparse
import risk_graph as rg

class RiskGame():
	"""
	This class provides the functions for a player to interact with a Risk board game
	"""
	def __init__(self, board, num_players, verbose):
		"""
		The constructor for the risk environment
		:param board: the string of the .risk file to be loaded
		"""
		self.num_players = num_players
		self.graph = rg.RiskGraph(board, verbose)
		self.player_list = []
		for player_id in range(self.num_players):
			self.add_player(player_id)
			self.activate_player(player_id)
		if verbose:
			print("Created player: {}".format(player_id))


		self.player_placement_order = np.random.permutation(len(self.player_list))
		self.terr_placement_order = np.random.permutation(self.graph.total_territories)
		for terr_id in self.terr_placement_order:
			player_choice = self.player_placement_order[terr_id%self.num_players]
			self.assign_territory(terr_id, player_choice)
			if verbose:	
				print("Territory {} assigned to player {}".format(terr_id, player_choice))

			# TOOD - determine how many armies to put on each territory

		self.player_turn_order = np.random.permutation(len(self.player_list))

		for terr in self.graph.territory_list:
			if terr.player_id == -1:
				print("Not all territories assigned")
				exit()
		for player in self.player_list:
			if player.isAlive == False:
				print("Not all players alive")
				exit()
			if player.isActive == False:
				print("Not all players active")
				exit()
			if player.total_armies < 1:
				print("Player {} has no armies".format(player.player_id))
				exit()
			if player.total_territories < 1:
				print("Player {} has no territories".format(player.player_id))
				exit()

		return

	def reinforce(self, player_id):
		"""
		A function to execute the reinforcement strategy of the player
		"""
		pass

	def assign_territory(self, terr_id, player_id):
		"""
		Assigns a territory in the game to a new player
		NOTE - This changes the number of armies, but only happens at beginning of game
		:param terr_id: int the unique id of the territory
		:param player_id: int the unique id of the player
		:return : none
		"""

		terr = self.graph.get_terr_by_id(terr_id)
		player = self.get_player_from_id(player_id)

		# Player effects
		player.total_territories += 1
		player.add_armies(1)
		player.territory_list.append(terr)

		# Territory effects
		if terr.player_id == player_id:
			print("Assigning territory to current occupant")
		terr.player_id = player_id
		terr.add_armies(1)
		return

	def set_player_id_to_terr(self, player_id, terr_id):
		"""
		This function sets a graph to a given player ID
		:params player_id: the unique ID of a player
		:params terr_id: the unique ID of a territory
		"""
		self.graph.get_terr_by_terr_id(self.graph, terr_id).set_player_id(player_id)
		return

	def add_player(self, player_id):
		"""
		This function adds a player to the game
		:param player_id: the unique ID of the player
		:return : No return value
		"""
		new_player = Player(player_id)
		self.player_list.append(new_player)
		return

	def activate_player(self, player_id):
		"""
		This function sets a player's isActive tag to True
		:param player_id: int the unique ID of the player
		"""
		player = self.get_player_from_id(player_id)
		player.isActive = True
		player.isAlive = True
		return

	def player_loses(self, player_id):
		"""
		This function sets a player's isActive tag to False
		:param player_id: int the unique ID of the player
		:return : no return value
		"""
		player =get_player_from_id(player_id)
		player.isAlive = False
		return

	def get_player_from_id(self, player_id):
		"""
		This function returns a player object when its unique ID is called
		:param player_id: the unique ID of the player
		:return player: Player the player specified by the unique ID
		"""
		if player_id >= self.num_players:
			print("Attempted to fetch nonexistent player")
			exit()
		return self.player_list[player_id]


class Player():
	"""
	This class defines a player for the Risk board game
	"""
	def __init__(self, player_id, name="unassigned", isAgent=False, policy="unassigned", isActive=False):
		"""
		The constructor for a player
		:param int: unique of the player
		:param name: string name of the player
		:param isAgent: boolean whether the player is a learning agent
		:param isActive: boolean whether the player is still in the game
		:param policy: string policy taken by the player
		:return : no return value
		"""
		self.player_id = player_id
		self.name = name
		self.isAgent = isAgent
		self.isActive = isActive
		self.isAlive = False
		self.territory_list = []

		# Note: armies or territories less than 1 corresponds to defeat once game is initialized
		self.total_armies = 0
		self.total_territories = 0

		# TODO: Add policies to correspond to agent and bot actions
		if not isAgent:
			self.policy_name = policy
		else:
			self.policy_name = policy

	def set_player_attributes(self, name, isAgent, policy="unassigned"):
		"""
		Defines player attributes
		:param name: string name of the player
		:param isAgent: boolean whether player is being trained
		:param policy: string which policy the player executes
		"""
		self.name = name
		self.isAgent = isAgent
		self.policy = policy

	def print_player_details(self):
		"""
		Prints the player's member variables
		:param : none
		:return : none
		"""
		if (self.isAgent):
			print("Player {}, {}, is an agent following {} policy".format(self.player_id, self.name, self.policy))
		else:
			print("Player {}, {}, is a bot following {} policy".format(self.player_id, self.name, self.policy))
		if self.isActive:
			if not self.isAlive:
				print("But it has lost the game")
			else:
				print("It is still in the game with {} territories and {} armies".format(self.total_territories,self.total_armies))
			return
		else:
			print("But it has not been activated yet")

	def add_armies(self, num_armies):
		"""
		Add armies of the same player to a territory
		Will print out to the console if number of armies is over 30
		:param num_armies: the number of armies to add to the territory
		:return : No return value
		"""
		self.total_armies += num_armies
		return

