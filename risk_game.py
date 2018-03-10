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
		self.graph = rg.RiskGraph(board, verbose)
		self.player_list = range(num_players)

		#print(self.graph.get_player_id_by_terr_id(0))


	def set_player_id_to_terr(self, player_id, terr_id):
		"""
		This function sets a graph to a given player ID
		:params player_id: the unique ID of a player
		:params terr_id: the unique ID of a territory
		"""
		self.graph.get_terr_by_terr_id(self.graph, terr_id).set_player_id(player_id)

		pass

class Player():
	"""
	This class defines a player for the Risk board game
	"""
	def __init__(self, name, isAgent=False, isActive=True, policy=""):
		"""
		The constructor for a player
		:param name: string name of the player
		:param isAgent: boolean whether the player is a learning agent
		:param isActive: boolean whether the player is still in the game
		:param policy: string policy taken by the player
		"""
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



		