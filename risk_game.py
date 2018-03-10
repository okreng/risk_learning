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

class RiskEnv():
	"""
	This class provides the functions for a player to interact with a Risk board game
	"""
	def __init__(self, board, num_players):
		"""
		The constructor for the risk environment
		:param board: the string of the .risk file to be loaded
		"""
		self.graph = rg.RiskGraph(board)
		self.player_list = range(num_players)

		print(self.graph.get_player_id_by_terr_id(0))


	def set_player_id_to_terr(self, player_id, terr_id):
		"""
		This function sets a graph to a given player ID
		:params player_id: the unique ID of a player
		:params terr_id: the unique ID of a territory
		"""
		self.graph.get_terr_by_terr_id(self.graph, terr_id).set_player_id()

		pass
