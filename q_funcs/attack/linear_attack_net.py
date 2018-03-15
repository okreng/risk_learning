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
	def __init__(self, nS, act_list, model_instance='0', checkpoint_index=-1, learning_rate = 0.001, verbose=True):
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


		self.module_string = 'linear_attack_net'
		self.action_type_string = 'attack'
		self.next_save = 1
		self.max_saves = 10
		self.exact_load = True

		if checkpoint_index == -1:
			continue_on = True
		else:
			continue_on = False

		self.save_folder, self.restore_folder = model_tree(model_instance, continue_on, self.module_string, self.action_type_string, verbose)

		num_chars = len(model_instance)
		if (not self.restore_folder[-num_chars:] is model_instance) or (model_instance is '0'):
			self.exact_load = False

		############ DO NOT DELETE - Valuable in the event of overwrite ###########
		print ('save_folder is {}'.format(self.save_folder))
		print ('restore_folder is {}'.format(self.restore_folder))
		############ End DO NOT DELETE #######################

		self.nS = nS
		# self.nA = nS**2 + 1  # Specific to this state-action representation
		self.nA = len(act_list)  # Length of 2D list corresponds to the edges of the graph

		# Define the graph

		self.features = tf.placeholder(dtype = tf.float32, shape = [None, self.nS], name='features')
		self.act = tf.placeholder(dtype = tf.int32, name='action_taken')

		# Labels will be set using TD(0) learning
		self.labels = tf.placeholder(dtype = tf.float32, shape = [None, self.nA], name='labels')

		# mask of action performed to backpropagate
		self.loss_weights = tf.placeholder(dtype = tf.float32, shape = [None, self.nA], name='loss_weights')

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

		# Begin session, initialize variables
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())
		
		# Create saver
		self.saver = tf.train.Saver(max_to_keep=self.max_saves, keep_checkpoint_every_n_hours=1)

		# Load model
		if (self.save_folder is self.restore_folder) or (not checkpoint_index == -1):  # Building off existing branch
			ckpt = tf.train.get_checkpoint_state(self.restore_folder)
			if ckpt and ckpt.model_checkpoint_path:
				if checkpoint_index == -1:
					if verbose:
						print("Loading model: ", ckpt.model_checkpoint_path)
					self.saver.restore(self.sess, ckpt.model_checkpoint_path)
					self.num_updates = tf.train.get_global_step()
				else:
					if (checkpoint_index < len(ckpt.all_model_checkpoint_paths)):
						if verbose:
							print("Loading model: ", ckpt.all_model_checkpoint_paths[checkpoint_index])
						self.saver.restore(self.sess, ckpt.all_model_checkpoint_paths[checkpoint_index])
						self.num_updates = tf.train.get_global_step()
					else:
						print("Checkpoint index did not exist, random initialization")
						self.exact_load = False
			else:
				print("Failed to load model from {}: random initialization within folder".format(self.restore_folder))
				self.exact_load = False  

		# Save first copy of model if new instance
		if not (self.exact_load):
			self.num_updates = 0
			self.checkpoint_path = self.save_folder + '/model.ckpt'
			self.saver.save(self.sess, self.checkpoint_path, global_step=self.num_updates)
			if verbose:
				print("Saved first copy in: {}".format(self.checkpoint_path))

		else:
			print("---------------WARNING--------------\nModel did not load or save correctly")

		return


############### TODO: Figure out how to safely save at the end of sessions #############

# def __del__(self):
# 	"""
# 	Destructor for attack networks - useful for printing and saving
# 	:param : none
# 	: return : none
# 	"""
# 	print("Saving module {} to {}, checkpoint: {}".format(self.module_string, self.save_folder, self.num_updates))
# 	self.saver.save(self.sess, self.checkpoint_path, global_step=self.num_updates)
# 	return


	def call_Q(self, state_vector, update=False, action_taken=0, target=0, loss_weights=None):
		"""
		This Q function will output the action specified by the function approximator
		:param state_vector: int the state of the board
		:param is_training: boolean whether to backpropagate
		:param reward: 
		:return action_vector: float The Q-function outputted by the network
		"""
		# print(is_training)

		if not update:
			return self.sess.run([self.output], feed_dict={self.features:state_vector})
		else:
			self.num_updates += 1
			_, q_function, loss = self.sess.run([self.train_op, self.output, self.loss], feed_dict={self.features:state_vector, self.act: action_taken, self.labels:target, self.loss_weights:loss_weights})
			if self.num_updates == self.next_save:
				self.saver.save(self.sess, self.checkpoint_path, global_step=self.num_updates)
				self.next_save += np.ceil(np.sqrt(self.num_updates))
			return q_function, loss