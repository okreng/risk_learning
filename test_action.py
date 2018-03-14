"""
This file contains the function for testing action functions
The functions are passed in game states
They return resulting game states
The results are all printed
"""


import numpy as np
import sys
import argparse
import importlib


def parse_arguments():
	# This function helps main read command line arguments
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('--module',dest='module',type=str)
	parser.add_argument('--function',dest='act_func',type=str)
	return parser.parse_args()

def main(args):
	"""
	Reads a policy filepath and runs state vectors on the given policy
	Prints results to console
	:param policy_filepath: string that contains directory and policy function dir.policy
	:return : none
	"""

	#import attack.max_success as pol0
	args = parse_arguments()
	module = args.module
	act_func_str = args.act_func

	act_module = importlib.import_module(module, package=None)
	action_function = getattr(act_module, act_func_str)

	# Define a list of state vectors
	s_v_list = []

	# Add state vectors to the list
	s_v_list.append(np.array([1, -1]))
	s_v_list.append(np.array([-1, 1]))
	s_v_list.append(np.array([-1, -1]))
	s_v_list.append(np.array([2, -1]))
	s_v_list.append(np.array([-2, 1]))


	# Begin test
	print("Testing Q-function: {}".format(module))
	for s_v in s_v_list:
		s_v = np.reshape(s_v, (1, -1))
		print("Testing action: {} on input state : {}".format(act_func_str, s_v))
		for ii in range(5):
			print("\tResult {}: {}".format(ii, action_function(s_v, 0, 1)) )
		print("\n")
	return



if __name__ == '__main__':
    main(sys.argv)