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
	def __init__(self, nS, model_instance='0', checkpoint_number=-1, learning_rate = 0.001, verbose=True):
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

		restore_path = model_tree(model_instance, self.module_string, self.action_type_string, verbose)
		# print (restore_path)

		self.nS = nS
		self.nA = nS**2  # Specific to this state-action representation
		nA = self.nA

		# the graph itself is here:
		# features = tf.placeholder(dtype = tf.int32, shape = [None, nS])
		# act = tf.placeholder(dtype = tf.float32, shape = [None, self.nA])

		# # Labels will be set using TD(0) learning in most cases
		# labels = tf.placeholder(dtype = tf.float32, shape = [None, self.nA])

		# # This will be the mask of which action was actually performed
		# loss_weights = tf.placeholder(dtype = tf.int32, shape = [None, self.nA])

		# hidden_layer = tf.layers.dense(inputs = features, units = nS, activation = None, use_bias = True, name = 'hidden_layer')
		# output = tf.layers.dense(inputs = hidden_layer, units = self.nA, activation = None, use_bias = True, name = 'output')

		# # TODO: Determine what is the best loss function
		# loss = tf.losses.mean_squared_error(labels=labels, predictions=output, weights=loss_weights)

		# print(tf.trainable_variables())

		# TODO: Define good learning rates
		# optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		# train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())




		# features = tf.reshape(features_,[1,nS])
		features = tf.placeholder(dtype = tf.float32, shape = [1, nS])
		act = tf.placeholder(dtype = tf.int32)
		labels = tf.placeholder(dtype = tf.float32, shape = [1, nA])
		loss_weights = tf.placeholder(dtype = tf.float32, shape = [1, nA])
		
		# W = tf.Variable(tf.random_uniform([nS,nA], 0, 0.01))
		# output = tf.matmul(features, W)

		####### Model #######
		# TODO: CHECK THE MODEL SOMETHING PROBABLY WRONG!

		# Input layer
		input_layer = features # tf.reshape(features_, [-1, 1])

		# Dense Layer
		# dense = tf.layers.dense(inputs = input_layer, units = nS, activation = None, name = 'dense')
		dense1 = tf.layers.dense(inputs = input_layer, units = 32, activation = tf.nn.relu, name = 'dense1', use_bias=True)
		dense2 = tf.layers.dense(inputs = dense1, units = 32, activation = tf.nn.relu, name='dense2', use_bias=True)
		dense3 = tf.layers.dense(inputs = dense2, units = 32, activation = tf.nn.relu, name='dense3', use_bias=True)
		# dense4 = tf.layers.dense(inputs=dense3, units=256, activation=tf.nn.relu, name='dense4', use_bias=True)

		# Output Layer
		output = tf.layers.dense(inputs = dense3, units = nA, name = 'output')
		
		#####################

		# predict = output[0, act]
		# predict_ = tf.reshape(predict, [1,1])
		# labels_ = output
		# labels_[0, act] = labels
		loss = tf.losses.mean_squared_error(labels=labels, predictions=output, weights=loss_weights)

		# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)

		








		self.saver = tf.train.Saver()



		return