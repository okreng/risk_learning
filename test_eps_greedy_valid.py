"""
File that tests the epsilon greedy valid policy
"""

import numpy as np
import sys
import argparse
import importlib

import attack_train_test as att



############ UNUSED BUT LEAVE IN BECAUSE WE MAY MOVE THE FUNCTION PATH ############
def parse_arguments():
	# This function helps main read command line arguments
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	# parser.add_argument('--module',dest='module',type=str)
	# parser.add_argument('--function',dest='pol_func',type=str)
	return parser.parse_args()

	############ UNUSED BUT LEAVE IN BECAUSE WE MAY MOVE THE FUNCTION PATH ############


def main(args):
	"""
	Reads a policy filepath and runs state vectors on the given policy
	Prints results to console
	:param module: string that contains module for policy
	:param function: string thet contains the name of the function
	:return : none
	"""



############ UNUSED BUT LEAVE IN BECAUSE WE MAY MOVE THE FUNCTION PATH ############
	# args = parse_arguments()
	# module = args.module
	# pol_func_str = args.pol_func

	# pol_module = importlib.import_module(module, package=None)
	# eps_greedy_valid = getattr(pol_module, pol_func_str)

############ UNUSED BUT LEAVE IN BECAUSE WE MAY MOVE THE FUNCTION PATH ############


	# Define a list of Q functions and valid masks
	q_list = []
	valid_list = []

	######### 2 element set ##############
	q_list.append([1, 2])
	valid_list.append([0, 0])
	valid_list.append([1, 0])
	valid_list.append([0, 1])
	valid_list.append([1, 1])


	EPSILON = 0.5  # Value that should give informative results

	# Begin test
	print("Testing Epsilon Greedy Valid Policy")
	for q in q_list:
		print("Q function: {}".format(q))
		for mask in valid_list:
			print("Mask: {}".format(mask))
			for ii in range(10):
				print("\tEGV Choice: {}".format(att.epsilon_greedy_valid(q, mask, EPSILON)))
			print("\n")


if __name__ == '__main__':
	main(sys.argv)