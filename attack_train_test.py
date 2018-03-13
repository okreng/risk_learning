"""
This function contains the code to test if a
training Q function approximator can learn the most basic
attack state-action value function
Note, this file does not work with agents because 
it is only using the attack function type
"""

import sys, argparse, random
import numpy as np

from q_funcs.attack import linear_attack_net
from q_funcs.attack import max_success


def parse_arguments():
	parser = argparse.ArgumentParser(description='Agent Argument Parser')
	return parser.parse_args()

def main(args):

	args = parse_arguments()

	# Simplest graph possible
	state_vector = np.zeros(2)

	T = len(state_vector)
	state_vector = np.reshape(state_vector, (1, -1))

	agent = linear_attack_net.LinearAttackNet(T)
	opponent = max_success.MaxSuccess(T)




if __name__ == '__main__':
	main(sys.argv)