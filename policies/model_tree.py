"""
This module holds the function for creating and loading models
"""

import sys
import os


# TODO: figure out general way to do this
# import repackage
# repackage.up(1)
# from risk_definitions import ROOT_DIR

# Testing purposes
# GLOBAL_NAME = 'I IS A GLOB'
# print(GLOBAL_NAME)

def model_tree(model_instance, checkpoint_number, module_name, action_type_name, verbose):
	"""
	Returns a model string of the model name to be loaded
	Updates .instance files so subsequent models will have unique ID's
	Creates a new .instance file with new model_instance $, holding 0
	:param model_instance: string in the form c1-c2-c3... branches from head
	:param module_name: string the name of the module, to locate the log folder
	:param action_type_name: string the name of the action type, to locate the log folder
	:return string: filepath (not including log file) of instance to be read
	"""
	print("Calling model tree with arguments: {}, {}, {}".format(model_instance, module_name, action_type_name))
	instance_path = './policies/' + action_type_name + '/' + module_name + '.logs/' + model_instance
	
	is_Instance = os.path.isfile(instance_path + '.instance')
	if is_Instance:
		if verbose:
			print("Branching off of {}".format(instance_path))
		with open(instance_path + '.instance', 'r') as instance:
			instance_num = int(instance.read())
			instance.close()
		with open(instance_path + '.instance', 'w') as instance:
			instance.write(str(instance_num+1))
			new_instance_path = instance_path + "-" + str(instance_num)
			# print(instance_num)
			# print(new_instance_path)
			instance.close()
		if verbose:
			print("Creating new instance {}.instance".format(new_instance_path))
		with open(new_instance_path + '.instance', 'w+') as new_instance:
			new_instance.write(str(0))
			new_instance.close()
	else:
		print("Path: {} is not a valid instance".format(instance_path))
		# instance_path = action_type_name + '/' + module_name + '.logs/head.instance'

	return
	
