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


				terr_edges = line.split(': ')
				neighbor_names = terr_edges[1].split(', ')
				neighbor_names[-1] = neighbor_names[-1].strip('\n')
				new_territory = Territory(terr_edges[0], neighbor_names, len(neighbor_names), terr_id)

############### WIP ######################

				# Check if territory is valid
				# for existing_terr in self.territories:
				# 	for existing_neighbor_name in existing_terr.neighbor_names:
				# 		# print(existing_neighbor_name)
				# 		for new_neighbor_name in new_territory.neighbor_names:
				# 			print(new_neighbor_name)
				# 			if (new_neighbor_name is existing_terr.name) | (existing_neighbor_name is new_neighbor_name):
				# 				print("Error in {}.risk. Territories {} and {} have uneven edges."
				# 					  .format(board,new_territory.name, existing_terr.name))
				# 				exit()

				self.territories.append(new_territory)

				terr_id += 1

			fboard.close()

		# Member variables - total_states - The number of states
		self.total_states = terr_id



		return


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