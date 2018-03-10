"""
This file defines the environment in which an agent can play the Risk game
"""

import tensorflow as tf
import numpy as np
import random
import time
import matplotlib as plot
import sys
import argparse
import risk_game as gm

class RiskEnv():
	"""
	This class converts the game of risk into a Markov Decision Process environment
	"""
	def __init__(self, board, matchup="default", verbose=False):
		"""
		Environment constructor
		:param board: string name of the board being played
		:param matchup: string name of file containing players in the game (Corresponding to policies)
		:param verbose: boolean whether to print large amounts of text
		"""
		
		self.players = []

		if verbose:
			print('Reading file: {}'.format('./matchups/' + matchup + '.mu'))
		with open('./matchups/' + matchup + '.mu') as fmu:
			line = fmu.read()
			fmu.close()
		self.player_names = line.split(', ')
		self.player_names[-1] = self.player_names[-1].strip('\n')
		num_players = len(self.player_names)

		self.game = gm.RiskGame(board,num_players,verbose)

		for player_num in range(num_players):
			isAgent = False

			# TODO - refer to list of agents
			if self.player_names[player_num] is "agent":
				isAgent = True

			# TODO - add policy with player
			new_player = self.game.get_player_from_id(player_num)
			new_player.set_player_attributes(self.player_names[player_num], isAgent, isActive=True, policy="")
			if verbose:
				print("Created player {}: {}".format(player_num, self.player_names[player_num]))


		return


def parse_arguments():
	# This function helps main read command line arguments
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('--board', dest='board', type=str)
	parser.add_argument('--matchup', dest='matchup', type=str, default="default")
	parser.add_argument('-v', dest='verbose', type=str, default='False')
	return parser.parse_args()


def main(args):
	# The main function for this file will print out environment details 
	args = parse_arguments()
	board = args.board
	matchup = args.matchup
	verbose_str = args.verbose
	if (verbose_str is 'True'):
		verbose = True
	else:
		verbose = False

	environment = RiskEnv(board, matchup, verbose)


# This is something you have to do in Python... I don't really know why	
if __name__ == '__main__':
	main(sys.argv)