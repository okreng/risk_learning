"""
This file builds a risk graph from a .risk file
# TODO: smart importing

"""

import tensorflow as tf
import numpy as np
import random
import time
import matplotlib as plot
import sys
import argparse

class RiskGraph():
	"""
	This class defines a graph for a game of Risk
	The graph is based on the Risk board game
	KEY ASSUMPTION: The agent will act optimally for a single battle
	"""
	def __init__(self, board, verbose):
		"""
		This is the constructor for the RiskGraph class
		The function assumes this file is in the boards folder
		:param board: string the filename of the .risk file being used as a board
		:param verbose: bool whether to print a lot of information
		"""

		# Member variable - territory_list - list of territory_list
		self.territory_list = []

		# Read through the .risk file and convert it to graph
		if verbose:
			print('Reading file: {}'.format('./boards/' + str(board) + '.risk'))
		with open('./boards/' + board + '.risk') as fboard:
			lines = fboard.readlines()

			terr_id = 0
			for line in lines:

				# Parse .risk file into territory_list
				terr_edges = line.split(': ')
				neighbor_names = terr_edges[1].split(', ')
				neighbor_names[-1] = neighbor_names[-1].strip('\n')
				new_territory = Territory(terr_edges[0], neighbor_names, len(neighbor_names), terr_id)
				self.territory_list.append(new_territory)
				if verbose:
					print('Created territory {}: {}'.format(terr_id, new_territory.name))
				terr_id += 1

			fboard.close()

		# Member variables - total_territories - The number of states
		self.total_territories = len(self.territory_list)

		# Member variable - edge_set - set of  edges accessible by unique id
		# Note - edges always referred to in min-max order to prevent aliasing
		self.edge_set = EdgeSet()
		edge_id = 0
		for terr_id in range(self.total_territories):
			for neighbor_name in (self.get_terr_by_id(terr_id).neighbor_names):
				dest_terr_id = self.get_terr_id_by_name(neighbor_name)
				if not (dest_terr_id == -1):  #Check if territory is a neighobr
					new_edge = Edge(self,terr_id, dest_terr_id)
					if (self.edge_set.add_edge(new_edge, edge_id)):
						edge_id += 1
		if verbose:
			self.edge_set.print_edge_list()

		print ("Graph Initialized")
		return
		

	def get_terr_by_id(self, terr_id):
		"""
		Function that returns territory_list from environment by id
		:param terr_id: int unique id of territory
		:return Territory: Territory identified by ID
		"""
		return self.territory_list[terr_id]

	def get_terr_id_by_name(self, terr_name):
		"""
		Function that returns a territory ID given its name
		:param terr_name: string the name of the territory
		:return Territory: returns a territory or -1 if unsuccessful
		"""
		for territory in self.territory_list:
			if (territory.name == terr_name):
				return territory.terr_id
		return -1

	def get_player_id_by_terr_id(self, terr_id):
		"""
		Function to retrieve which player is in control of a territory
		:param terr_id: int the unique ID of the territory
		:return player_id: int the unique ID of the player occupying the territory
		"""
		return self.get_terr_by_id(terr_id).player_id

	def get_armies_by_terr_id(self, terr_id):
		"""
		Function to retrieve how many armies are on a territory
		:param terr_id: int the unique ID of the territory
		:return armies: int the number of armies on the territory
		"""
		return self.get_terr_by_id(terr_id).armies


class Edge():
	"""
	This object is an edge with a unique ID that points to 2 territory_list (nodes)
	"""

	def __init__(self, risk_graph, terr_1_id, terr_2_id):
		"""
		Constructor for the Edge class
		:param risk_graph: RiskGraph object holding list of all territories
		:param terr_1_id: int the unique ID of the first territory on the edge
		:param terr_2_id: int the unique ID of the second territory on the edge
		:return : none
		"""


		# Member variables:
		if (terr_1_id == terr_2_id):
			print("Cannot create an edge within a territory")
			exit()
		else:
			self.node_id_1 = min(terr_1_id, terr_2_id)
			self.node_1 = risk_graph.get_terr_by_id(self.node_id_1)
			self.node_id_2 = max(terr_1_id, terr_2_id)
			self.node_2 = risk_graph.get_terr_by_id(self.node_id_2)

		# Give invalid value until assignment
		self.edge_id = -1
		return

	def assign_id(self, edge_id):
		"""
		Assigns a unique ID to an edge
		:param edge_id: int a unique edge_id
		:return : none
		"""
		self.edge_id = edge_id
		return

	def get_node_id_1(self):
		"""
		Fetches the first node ID
		:param : none
		:return node_id_1: int the ID of the first territory in the edge
		"""
		return self.node_id_1

	def get_node_1(self):
		"""
		Fetches the first territory pointed to by the edge
		:param : none
		:return node_1: Territory the first territory pointed to by the edge
		"""
		return self.node_1

	def get_node_2(self):
		"""
		Fetches the second node ID
		:param : none
		:return node_id_1: int the ID of the second territory in the edge
		"""
		return self.node_2

	def get_node_id_2(self):
		"""
		Fetches the first territory pointed to by the edge
		:param : none
		:return node_1: Territory the first territory pointed to by the edge
		"""
		return self.node_id_2


class EdgeSet():
	"""
	This object holds a set of unique edges that can be accessed by id
	This provides random access while preventing duplicate edges in the graph
	"""

	def __init__(self):
		"""
		Constructor for the edge set class
		:params : none
		:return : none
		"""

		# Member variable: a set of unique edges
		self.edges = set()
		self.edge_list = []
		self.num_edges = 0
		return

	def add_edge(self, new_edge, edge_id):
		"""
		Add an edge to the set, if the edge already exists, return false and do not add'
		:param new_edge: Edge object created with two node ID''s
		:return : boolean True if edge object had unique ID's and it was added to the set,
		false otherwise
		"""

		if (self.edges.isdisjoint([(new_edge.get_node_id_1(),new_edge.get_node_id_2())])):
			self.edges.add((new_edge.get_node_id_1(), new_edge.get_node_id_2()))
			self.edge_list.append(new_edge)
			new_edge.assign_id(edge_id)
			self.num_edges += 1
	
			return True
		else:
			return False

	def get_edge_by_id(self, edge_id):
		"""
		Returns edge object by unique ID
		:param edge_id: int unique ID of an edge
		:return Edge: returns the edge in the graph
		"""
		return self.edge_list[edge_id]

	def print_edge_list(self):
		"""
		Prints details about the whole list of edges
		:params : none
		:return : none
		"""
		print("Created edges between:")
		for edge in range(self.num_edges):
			print("\t{} and {}".format(self.edge_list[edge].get_node_1().name,self.edge_list[edge].get_node_2().name))
		return


class Territory():
	"""
	This class defines a Territory or node in the graph
	"""

	def __init__(self, name, neighbor_names, edge_num, terr_id, armies=0, player_id=-1):
		"""
		This is the constructor for the Territory class
		:param name: string the name of the territory
		:param neighbor_names: list of territory names - the names of bordering territory_list
		:param edge_num: int the number of territories attached to this Territory
		:param terr_id: int the unique id of the territory on the board
		:param armies: int the number of armies on the territory
		:param player_id: int the unique ID of the player occupying the territory
		"""
		self.name = name
		self.neighbor_names = neighbor_names
		self.edge_num = edge_num
		self.terr_id = terr_id
		
		# Note - armies of size <= 0 and player_id of <0 will produced errors after initalization
		self.armies = armies
		self.player_id = player_id

	def set_player_id(self, player_id):
		"""
		This function sets the player ID to a new value
		:param player_id: the player ID to set the territory to
		:return : No return value
		"""
		self.player_id = player_id

	def remove_armies(self, num_armies):
		"""
		This function removes armies from a territory
		:param num_armies: the number of armies to remove
		:return defeated: True if the number of armies on a territory is 0 or less after removal
		"""
		self.armies -= num_armies
		if self.armies <= 0:
			return True
		else:
			return False

	def add_armies(self, num_armies):
		"""
		Add armies of the same player to a territory
		Will print out to the console if number of armies is over 30
		:param num_armies: the number of armies to add to the territory
		:return : No return value
		"""
		self.armies += num_armies
		if self.armies > 30:
			print("More than 30 armies on {}".format(self.name))
		return

	def set_armies(self, num_armies):
		"""
		Set armies of the same player to a territory
		Will print out to the console if number of armies is over 30
		:param num_armies: the number of armies to add to the territory
		:return : No return value
		"""
		self.armies = num_armies
		if self.armies > 30:
			print("More than 30 armies on {}".format(self.name))
		return

	def get_armies(self):
		"""
		Fetcharmies of the same player to a territory
		:param : none
		:return self.num_armies: 
		"""
		return self.armies