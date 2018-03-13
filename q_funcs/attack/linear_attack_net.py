"""
This file contains the class LinearAttackNet which calls Q functions based on a linear neural net
This is a simple net designed to be easy to train
Once this net is developed and works on simple models, deeper nets will be developed
The action_vector outputted is equivalent to a Q-function that will be learned
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys


# Define Root directory
# TODO: figure out best way to do this
# TODO: Until then, this must be run from the root directory of the project
# # statement below appears to re-run from inner directory
# # from risk_definitions import ROOT_DIR


##### Working from root directory #####
import repackage
repackage.up(1)
from model_tree import model_tree
##### End Working from root directory #####

class LinearAttackNet():
	"""
	Class to hold a linear neural network
	Will be used to learn Attacks in RISK
	"""
	def __init__(self, nS, model_instance='0', checkpoint_index=-1, learning_rate = 0.001, verbose=True):
		"""
		Creates a session of the tensorflow graph defined in this module
		:param nS: int required, will throw error if does not agree, the number of territories on the graph
		with model/checkpoint, this one number fully defines state and action space
		:param model_instance: string Which model_instance to load.  
		The num.instance file will hold the next instance number
		If this parameter is not specified a new random model will be instantiated under 0.instance
		:param chekpoint_index: int the checkpoint index in the checkpoint file of all_model_checkpoint_paths
		Defaults to latest checkpoint
		:return success: boolean whether the model could be loaded as specified

		"""

		# TODO: Check what other options are good for running multiple networks simultaneously
		# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
		gpu_ops = tf.GPUOptions(allow_growth=True)
		config = tf.ConfigProto(gpu_options=gpu_ops)

		tf.reset_default_graph()

		self.sess = tf.Session(config=config)

		self.module_string = 'linear_attack_net'
		self.action_type_string = 'attack'
		self.num_updates = 0
		self.last_save = 0
		self.exact_load = True

		self.save_path, self.restore_path = model_tree(model_instance, self.module_string, self.action_type_string, verbose)

		############ DO NOT DELETE - Valuable in the event of overwrite ###########
		# print ('save_path is {}'.format(save_path))
		# print ('restore_path is {}'.format(restore_path))
		############ End DO NOT DELETE #######################

		self.nS = nS
		self.nA = nS**2 + 1  # Specific to this state-action representation

		# Define the graph

		self.features = tf.placeholder(dtype = tf.float32, shape = [None, self.nS])
		self.act = tf.placeholder(dtype = tf.int32)

		# Labels will be set using TD(0) learning
		self.labels = tf.placeholder(dtype = tf.float32, shape = [None, self.nA])

		# mask of action performed to backpropagate
		self.loss_weights = tf.placeholder(dtype = tf.float32, shape = [None, self.nA])

		# Single hidden Layer
		self.dense = tf.layers.dense(inputs = self.features, units = self.nS, activation = None, use_bias = True, name = 'dense')
	
		# Output Layer
		self.output = tf.layers.dense(inputs = self.dense, units = self.nA, use_bias = True, name = 'output')
		
		#####################
		self.loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.output, weights=self.loss_weights)

		# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)

		# TODO: Define good learning rate
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		self.train_op = self.optimizer.minimize(loss = self.loss, global_step = tf.train.get_global_step())

		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()

		# TODO: Load specified checkpoint, default to latest
		# self.saver.restore(restore_path + '.checkpoint ')

		# Load model
		if not (model_instance is '0'):  # Not random initialization
			ckpt = tf.train.get_checkpoint_state(self.restore_path)
			if ckpt and ckpt.model_checkpoint_path:
				if checkpoint_index == -1:
					if verbose:
						print("Loading model: ", ckpt.model_checkpoint_path)
					self.saver.restore(self.sess, ckpt.model_checkpoint_path)
				else:
					if (checkpoint_index < len(all_model_checkpoint_paths)):
						if verbose:
							print("Loading model: ", ckpt.all_model_checkpoint_paths[checkpoint_index])
						self.saver.restore(self.sess, ckpt.all_model_checkpoint_paths[checkpoint_index])
					else:
						print("Checkpoint index did not exist, random initialization")
						self.exact_load = False
			else:
				print("Failed to load model from {}: random initialization".format(self.restore_path))
				self.exact_load = False  

		# Save first copy of model 
		self.checkpoint_path = self.save_path + '/model.ckpt'
		self.saver.save(self.sess, self.checkpoint_path, global_step=self.num_updates)
		if verbose:
			print("Saved first copy in: {}".format(self.checkpoint_path))


		return # False indicates the model was randomly initialized

	def call_Q(self, state_vector, is_training=False, action_taken=0, target=0, loss_weights=None):
		"""
		This Q function will output the action specified by the function approximator
		:param state_vector: int the state of the board
		:param is_training: boolean whether to backpropagate
		:param reward: 
		:return action_vector: float The Q-function outputted by the network
		"""
		# print(is_training)

		if not is_training:
			return self.sess.run([self.output], feed_dict={self.features:state_vector})
		else:
			self.updates += 1
			_, q_function, loss = self.sess.run([train_op, self.output, self.loss], feed_dict={self.features:state_vector, self.act: action_taken, self.labels:target, self.loss_weights:loss_weights})
			return q_function, loss