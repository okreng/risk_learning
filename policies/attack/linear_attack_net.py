"""
This file contains the class LinearAttackNet which implements policies based on a linear neural net
This is a simple net designed to be easy to train
Once this net is developed and works on simple models, deeper nets will be developed
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys


# Define Root directory
# TODO: figure out best way to do this
# TODO: Until then, this must be run from the root directory of the project
import repackage
repackage.up(1)
# statement below appears to re-run from inner directory
# from risk_definitions import ROOT_DIR
from policies.model_tree import model_tree


class LinearAttackNet():
	"""
	Class to hold a linear neural network
	Will be used to learn Attacks in RISK
	"""
	def __init__(self,num_territories, is_training=False, model_instance='0', checkpoint_number=-1, verbose=True):
		"""
		Creates a session of the tensorflow graph defined in this module
		:param num_territories: int required, will throw error if does not agree 
		with model/checkpoint, this one number fully defines state and action space
		:param is_training: boolean whether to backpropagate and learn or use the model to predict
		:param model_instance: string Which model_instance to load.  
		The num.instance file will hold the next instance number
		If this parameter is not specified a new random model will be instantiated under 0.instance
		:param chekpoint_number: int the checkpoint number to load from
		Defaults to latest checkpoint if model_instance is specified, otherwise no load is performed
		:return success: boolean whether the model could be loaded as specified
		"""

		# TODO: Check what other options are good for running multiple networks simultaneously
		# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
		gpu_ops = tf.GPUOptions(allow_growth=True)
		config = tf.ConfigProto(gpu_options=gpu_ops)
		self.sess = tf.Session(config=config)

		self.module_string = 'linear_attack_net'
		self.action_type_string = 'attack'

		restore_path = model_tree(model_instance, checkpoint_number, self.module_string, self.action_type_string, verbose)
		# print (restore_path)

		return