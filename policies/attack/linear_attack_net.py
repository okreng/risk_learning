"""
This file contains the class LinearAttackNet which implements policies based on a linear neural net
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
# import repackage
# repackage.up(1)
# # statement below appears to re-run from inner directory
# # from risk_definitions import ROOT_DIR
# from policies.model_tree import model_tree


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

        tf.reset_default_graph()

        self.sess = tf.Session(config=config)

        self.module_string = 'linear_attack_net'
        self.action_type_string = 'attack'

        # restore_path = model_tree(model_instance, self.module_string, self.action_type_string, verbose)
        # print (restore_path)

        self.nS = nS
        self.nA = nS**2 + 1  # Specific to this state-action representation

        # Define the graph

        self.features = tf.placeholder(dtype = tf.float32, shape = [None, self.nS])
        self.act = tf.placeholder(dtype = tf.int32)

        # Labels will be set using TD(0) learning
        self.labels = tf.placeholder(dtype = tf.float32, shape = [None, self.nA])

        # mask of action performed to backpropagate
        self.loss_weights = tf.placeholder(dtype = tf.float32, shape = [None, self.nA])

        # Input layer
        # self.input_layer = features # tf.reshape(features_, [-1, 1])

        # Dense Layer
        self.dense = tf.layers.dense(inputs = self.features, units = self.nS, activation = None, use_bias = True, name = 'dense')

        # Output Layer
        self.output = tf.layers.dense(inputs = self.dense, units = self.nA, use_bias = True, name = 'output')

        #####################
        self.loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.output, weights=self.loss_weights)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)

        # TODO: Define good learning rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        self.train_op = self.optimizer.minimize(loss = self.loss, global_step = tf.train.get_global_step())

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        # TODO: Load specified checkpoint, default to latest
        # self.saver.restore(restore_path + '.checkpoint ')

        return

    def enact_policy(self, state_vector, is_training=False, action_taken=0, target=0, loss_weights=None):
        """
        This policy will output the action specified by the function approximator
        :param state_vector: int the state of the board
        :param is_training: boolean whether to backpropagate
        :param reward:
        :return action_vector: float The Q-function outputted by the network
        """
        # print(is_training)

        if not is_training:
            return self.sess.run([self.output], feed_dict={self.features:state_vector})
        else:
            _, q_function, loss = self.sess.run([train_op, self.output, self.loss], feed_dict={self.features:state_vector, self.act: action_taken, self.labels:target, self.loss_weights:loss_weights})
            return q_function