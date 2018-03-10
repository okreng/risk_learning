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
import risk_env as env











def parse_arguments():
	# This function helps main read command line arguments
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('--board',dest='board',type=str)
	parser.add_argument('--players',dest='players', type=str, default="")
	return parser.parse_args()


def main(args):
	# The main function for this file will print out environment details 
	args = parse_arguments()
	board = args.board
	num_players = args.num_players

	environment = RiskEnv(board, num_players)


# This is something you have to do in Python... I don't really know why	
if __name__ == '__main__':
	main(sys.argv)