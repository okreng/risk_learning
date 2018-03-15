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
	Reads an action filepath and runs state vectors on the given action
	Prints results to console
	:param module: string that contains the module of the function
	:param function: string name of the function
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

	# TODO: These can be toggled on and off
	# Add state vectors to the list for 2 element test
	############# for attack#########
	# s_v_list.append(np.array([1, -1]))
	# s_v_list.append(np.array([2, -1]))
	# s_v_list.append(np.array([3, -1]))
	# s_v_list.append(np.array([4, -1]))
	# s_v_list.append(np.array([-2, 2]))
	# s_v_list.append(np.array([-3, 2]))
	# s_v_list.append(np.array([-4, 2]))
	# s_v_list.append(np.array([3, -1]))

	# for 3 (n-) element states
	s_v_list.append(np.array([4, 1, -3]))
	s_v_list.append(np.array([-5, 2, -2]))



	############ end for attack##########

	# Add indices to test n-element states
	indice_list = []
	indice_list.append(np.array([0, 1]))
	indice_list.append(np.array([1, 0]))
	indice_list.append(np.array([0, 2]))
	indice_list.append(np.array([2, 0]))
	indice_list.append(np.array([1, 2]))
	indice_list.append(np.array([2, 1]))


	# Begin test
	print("Testing action module: {}".format(module))
	for s_v in s_v_list:
		s_v = np.reshape(s_v, (1, -1))

		# TODO: This is for 2 element states
		# print("Testing action: {} on input state : {}".format(act_func_str, s_v))
		# for ii in range(5):
		# 	print("\tResult {}: {}".format(ii, action_function(s_v, 0, 1)) )
		# print("\n")
		########### end 2 element states test ########3

		for indices in indice_list:
			print("Testing action: {} on (input state, indices): ({}, {})".format(act_func_str, s_v, indices))
			for ii in range(5):
				print("\tResult {}: {}".format(ii, action_function(s_v, indices[0], indices[1])))
			print("\n")
	return



if __name__ == '__main__':
    main(sys.argv)