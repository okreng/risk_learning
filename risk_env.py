# This file defines a risk environment for reinforcement learning

import tensorflow as tf
import numpy as np
import random
import time
import matplotlib as plot
import sys
import argparse

class RiskEnv():
	# This class defines an environment with which an agent can interact
	# The environment is based on the Risk board game
	# KEY ASSUMPTION: The agent will act optimally for a single battle

	def __init__(self, board):
		# This is the constructor for the Risk_Env class
		# Arguments:
		# board - string - the filename of the .risk file being used as a board
		# The function assumes this file is in the boards folder

		# Member variable - territories - list of territories
		self.territories = []

		# Read through the .risk file and convert it to graph
		print('Opening file: {}'.format('./boards/' + str(board) + '.risk'))
		with open('./boards/' + board + '.risk') as fboard:
			lines = fboard.readlines()

			terr_id = 0
			for line in lines:

				# Parse .risk file into territories
				terr_edges = line.split(': ')
				neighbor_names = terr_edges[1].split(', ')
				neighbor_names[-1] = neighbor_names[-1].strip('\n')
				new_territory = Territory(terr_edges[0], neighbor_names, len(neighbor_names), terr_id)
				self.territories.append(new_territory)
				terr_id += 1

			fboard.close()

		# Member variables - total_territories - The number of states
		self.total_territories = len(self.territories)

		# Member variable - edges - set of edges by ID
		# Note - edges always referred to in min-max order to prevent aliasing
		self.edge_set = EdgeSet()
		self.ordered_edge_list = []
		edge_id = 0
		for terr_id in range(self.total_territories):
			for neighbor_name in (self.get_terr_by_id(terr_id).neighbor_names):
				dest_terr_id = self.get_terr_id_by_name(neighbor_name)
				if not (dest_terr_id == -1):
					new_edge = Edge(terr_id, dest_terr_id)
					if (self.edge_set.add_edge(new_edge, edge_id)):
						edge_id += 1


		for edge_num in range(edge_id):
			this_edge = self.edge_set.get_edge_by_id(edge_num)
			#print("Edge connects nodes: {}, {}".format(this_edge.get_node_1, this_edge.get_node_2))

					

		# print("Edge set is: \n{}".format(self.edges))
		return
		

	def get_terr_by_id(self, terr_id):
		#Function that returns territories from environment by id
		return self.territories[terr_id]

	def get_terr_id_by_name(self, terr_name):
		# Function that returns a territory ID given its name
		for territory in self.territories:
			if (territory.name == terr_name):
				return territory.terr_id
		return -1


class Edge():
	# This object is an edge with a unique ID that points to 2 territories by id

	def __init__(self, terr_1_id, terr_2_id):
		# Member variables:
		if (terr_1_id == terr_2_id):
			print("Cannot create an edge within a territory")
			exit()
		else:
			self.node_1 = min(terr_1_id, terr_2_id)
			self.node_2 = max(terr_1_id, terr_2_id)
			# print("Creating edge between nodes {} and {}".format(self.node_1, self.node_2))


		# Give invalid value until assignment
		self.edge_id = -1
		return

	def assign_id(self, edge_id):
		# Set unique edge_id
		self.edge_id = edge_id
		return

	def get_node_1(self):
		return self.node_1

	def get_node_2(self):
		return self.node_2

class EdgeSet():
	# This object holds a set of unique edges

	def __init__(self):
		# Member variable: a set of unique edges
		self.edges = set()
		self.edge_list = []
		self.num_edges = 0
		return

	def add_edge(self, new_edge, edge_id):
		# Add an edge to the set, if the edge already exists, return false and do not add
		if (self.edges.isdisjoint((0,1))):
			self.edges.add((new_edge.get_node_1(), new_edge.get_node_2()))
			self.edge_list.append(new_edge)
			print("Adding edge between: {} and {}".format(new_edge.get_node_1(),new_edge.get_node_2()))
			print(self.edges)
			new_edge.assign_id(edge_id)
			self.num_edges += 1
	
			return True
		else:
			return False

	def get_edge_by_id(self, edge_id):
		return self.edge_list[edge_id]


class Territory():
	# This class defines a Territory or node in the graph

	def __init__(self, name, neighbor_names, edge_num, terr_id, armies=0, player_id=0):
		# This is the constructor for the Territory_ class
		# Arguments:
		# name - string - the name of the territory
		# neighbor_names - list of territory names - the names of bordering territories
		# edge_num - the maximum number of possible actions for a given territory
		# terr_id - int - the unique id of the territory on the board
		# armies - int - the number of armies on the territory
		# player_id - int - the unique ID of the player occupying the territory
		self.name = name
		self.neighbor_names = neighbor_names
		self.edge_num = edge_num
		self.terr_id = terr_id
		self.armies = armies

		print('Created territory {}: {} \n\tNeighboring: {}'.format(terr_id, name, neighbor_names))

		# Note - the default player_id of 0 will give an error in the environment
		self.player_id = player_id

def parse_arguments():
	# This function helps main read command line arguments
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('--board',dest='board',type=str)
	return parser.parse_args()


def main(args):
	# The main function for this file will print out environment details 
	args = parse_arguments()
	board = args.board

	environment = RiskEnv(board)


# This is something you have to do in Python... I don't really know why	
if __name__ == '__main__':
	main(sys.argv)