"""
This file provides the environment for a player to interact with the Risk game board
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
			new_player = self.add_player(player_id)
		if verbose:
			print("Created empty player: {}".format(player_id))

		#print(self.graph.get_player_id_by_terr_id(0))

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
	def __init__(self, player_id, name="unassigned", isAgent=False, isActive=False, policy=""):
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

		# Note: armies or territories less than 1 corresponds to defeat once game is initialized
		self.total_armies = 0
		self.total_territories = 0

		# TODO: Add policies to correspond to agent and bot actions
		if not isAgent:
			self.policy_name = policy
		else:
			self.policy_name = policy

	def set_player_attributes(self, name, isAgent, isActive, policy):
		"""
		Defines player attributes
		:param name: string name of the player
		:param isAgent: boolean whether player is being trained
		:param isActive: boolean whether the player is alive in the game
		:param policy: string which policy the player executes
		"""
		self.name = name