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
	def __init__(self, board):
		"""
		The constructor for the risk environment
		:param board: the string of the .risk file to be loaded
		"""
		self.graph = rg.RiskGraph(board)



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