"""
A file for testing different state vector inputs on policies
This function will print out which state vectors are being tested and the results
Policy functions must share name with file
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
	parser.add_argument('--class',dest='q_func',type=str)
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
	q_func_str = args.q_func

	q_func_module = importlib.import_module(module, package=None)
	q_func_class = getattr(q_func_module, q_func_str)

	# Define a list of state vectors
	s_v_list = []

	# Add state vectors to the list
	s_v_list.append(np.array([1, -1]))
	s_v_list.append(np.array([1, 2, -1]))
	s_v_list.append(np.array([2, -2, -1]))
	s_v_list.append(np.array([1, -2, -1]))
	s_v_list.append(np.array([3, -2, -1]))
	s_v_list.append(np.array([4, -2, -1]))


	# Add act_lists to the list
	act_lists_list = []
	act_lists_list.append([[0, 1],[-1]])
	act_lists_list.append([[0,1],[0, 2],[1, 2],[-1]])
	act_lists_list.append([[0,1],[0, 2],[1, 2],[-1]])
	act_lists_list.append([[0,1],[0, 2],[1, 2],[-1]])
	act_lists_list.append([[0,1],[0, 2],[1, 2],[-1]])
	act_lists_list.append([[0,1],[0, 2],[1, 2],[-1]])
	act_lists_list.append([[0,1],[0, 2],[1, 2],[-1]])




	# Begin test
	print("Testing Q-function: {}".format(module))
	for test_num in range(len(s_v_list)):
		q_func_obj = q_func_class(len(s_v_list[test_num]), act_lists_list[test_num])
		s_v = np.reshape(s_v_list[test_num], (1, -1))
		print("State : {}".format(s_v))
		print("Action list : {}".format(act_lists_list[test_num]))
		print("Action:")
		print(q_func_obj.call_Q(s_v))
		print("\n")
	return



if __name__ == '__main__':
    main(sys.argv)
