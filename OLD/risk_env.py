"""
This file defines the environment in which an agent can play the Risk game
# TODO: smart importing
"""

import sys
import argparse
from OLD import risk_game as gm


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
		
		# parse .mu file into player types
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
			new_player.set_player_attributes(self.player_names[player_num], isAgent)
			if verbose:
				new_player.print_player_details()

		self.game.random_start(verbose)
		self.game.random_start(verbose)


		return


def parse_arguments():
	"""
	This function helps main read command line arguments
	:params : none
	:return Parser: parser object containing arguments passed from command line
	"""
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('-b', dest='board', type=str)
	parser.add_argument('-m', dest='matchup', type=str, default="default")
	parser.add_argument('-v', dest='verbose', type=bool, default=False)
	return parser.parse_args()


def main(args):
	"""
	This function initializes a game inside an environment and uses is to train an agent
	:param args: Command line arguments
	:return : this function does not return
	"""

	args = parse_arguments()
	board = args.board
	matchup = args.matchup
	verbose = args.verbose

	environment = RiskEnv(board, matchup, verbose)


# This is something you have to do in Python... I don't really know why	
if __name__ == '__main__':
	main(sys.argv)