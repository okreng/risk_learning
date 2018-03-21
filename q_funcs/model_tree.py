"""
This module holds the function for creating and loading models
"""

import sys
import os

# TODO: figure out general way to do this
# import repackage
# repackage.up(1)
# from risk_definitions import ROOT_DIR

def model_tree(model_instance, continue_on, module_name, action_type_name, verbose):
	"""
	Returns a model string of the model name to be loaded
	Updates .instance files so subsequent models will have unique ID's
	Creates a new .instance file with new model_instance $, holding 0
	:param model_instance: string in the form c1-c2-c3... branches from head
	:param module_name: string the name of the module, to locate the log folder
	:param action_type_name: string the name of the action type, to locate the log folder
	:return bool: whether the model_instance existed
	:return string: filepath (not including log file) of new instance to be written
	"""
	instance_path = './q_funcs/' + action_type_name + '/' + module_name + '.logs/' + model_instance
	
	is_Instance = os.path.isfile(instance_path + '.instance')
	if not is_Instance:
		print("Path: {}.instance is not a valid instance".format(instance_path))
		instance_path = './q_funcs/' + action_type_name + '/' + module_name + '.logs/0'
		continue_on = False
	if (not continue_on) or (instance_path[-2:] == '/0'):
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
		if not os.path.exists(new_instance_path):
			os.makedirs(new_instance_path)
			if verbose:
				print("Creating new folder {}".format(new_instance_path))
		else:
			print("Folder already exists, aborting attempt to overwrite data, check network")
			exit()

		return new_instance_path, instance_path
	else:  # No new instances for continuing on training the same model
		print("Continuing on branch {}".format(instance_path))
		return instance_path, instance_path
