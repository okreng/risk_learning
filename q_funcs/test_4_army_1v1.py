"""
A file for testing different state vector inputs on policies
This function will print out which state vectors are being tested and the results
Policy functions must share name with file
"""

import numpy as np
import sys
import argparse
import importlib

from attack import linear_attack_net
from attack import max_success
from attack import random_attack
from attack import army_difference
from attack import optimal_4_army_1v1
from attack import three_layer_attack_net
from attack import leaky_relu_3_layer
from attack import optimal_4_army_1v1

# def parse_arguments():
# 	# This function helps main read command line arguments
# 	parser = argparse.ArgumentParser(description=
# 		'Risk Environment Argument Parser')
# 	parser.add_argument('--module',dest='module',type=str)
# 	parser.add_argument('--class',dest='q_func',type=str)
# 	return parser.parse_args()

def main(args):
	"""
	Reads a policy filepath and runs state vectors on the given policy
	Prints results to console
	:param policy_filepath: string that contains directory and policy function dir.policy
	:return : none
	"""

	#import attack.max_success as pol0
	# args = parse_arguments()
	# module = args.module
	# q_func_str = args.q_func

	# q_func_module = importlib.import_module(module, package=None)
	# q_func_class = getattr(q_func_module, q_func_str)

	# Define a list of state vectors
	s_v_list = []

	# Add state vectors to the list
	s_v_list.append(np.array([1, -1]))
	s_v_list.append(np.array([1, -2]))
	s_v_list.append(np.array([1, -3]))
	s_v_list.append(np.array([1, -4]))
	s_v_list.append(np.array([2, -1]))
	s_v_list.append(np.array([2, -2]))
	s_v_list.append(np.array([2, -3]))
	s_v_list.append(np.array([2, -4]))
	s_v_list.append(np.array([3, -1]))
	s_v_list.append(np.array([3, -2]))
	s_v_list.append(np.array([3, -3]))
	s_v_list.append(np.array([3, -4]))
	s_v_list.append(np.array([4, -1]))
	s_v_list.append(np.array([4, -2]))
	s_v_list.append(np.array([4, -3]))
	s_v_list.append(np.array([4, -4]))

	# s_v_list.append(np.array([-1, 1]))
	# s_v_list.append(np.array([-1, 2]))
	# s_v_list.append(np.array([-1, 3]))
	# s_v_list.append(np.array([-1, 4]))
	# s_v_list.append(np.array([-2, 1]))
	# s_v_list.append(np.array([-2, 2]))
	# s_v_list.append(np.array([-2, 3]))
	# s_v_list.append(np.array([-2, 4]))
	# s_v_list.append(np.array([-3, 1]))
	# s_v_list.append(np.array([-3, 2]))
	# s_v_list.append(np.array([-3, 3]))
	# s_v_list.append(np.array([-3, 4]))
	# s_v_list.append(np.array([-4, 1]))
	# s_v_list.append(np.array([-4, 2]))
	# s_v_list.append(np.array([-4, 3]))
	# s_v_list.append(np.array([-4, 4]))


	act_list = [[0,1],[-1]]


	model_instance = '0-14-4'
	checkpoint_number = -1
	LEARNING_RATE = 0.0001
	perform_update = False

	T = 2

	# agent = optimal_4_army_1v1.Optimal4Army1V1(T, act_list)
	# agent = max_success.MaxSuccess(T, act_list)
	# agent = army_difference.ArmyDifference(T, act_list)
	agent = linear_attack_net.LinearAttackNet(T, act_list, model_instance, checkpoint_number, LEARNING_RATE)
	# agent = three_layer_attack_net.ThreeLayerAttackNet(T, act_list, model_instance, checkpoint_number, LEARNING_RATE)
	# agent = leaky_relu_3_layer.LeakyRelu3Layer(T, act_list, model_instance, checkpoint_number, LEARNING_RATE)



	# q_func_obj = q_func_class(len(s_v_list[0]), act_lists_list[0])
	# Begin test
	for test_num in range(len(s_v_list)):
		s_v = np.reshape(s_v_list[test_num], (1, -1))
		print("State : {}".format(s_v))
		print("Action:")
		q = agent.call_Q(s_v)
		print(q)
		if q[0] > q[1]:
			print("Action: attack")
		else:
			print("Action: pass")
		print("\n")
	return



if __name__ == '__main__':
    main(sys.argv)
