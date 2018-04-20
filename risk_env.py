"""
This file defines the environment in which an agent can play the Risk game
# TODO: smart importing
"""

import sys
import argparse
# from OLD import risk_game as gm
import risk_game as gm
from risk_game import ActionType
import numpy as np
import agent
import utils

import matplotlib
import matplotlib.pyplot as plt

TRAIN_EPSILON = 0
TEST_EPSILON = 0.05
TIMEOUT_STATES = 10000


class RiskEnv():
	"""
	This class converts the game of risk into a Markov Decision Process environment
	"""
	def __init__(self, board, matchup="default", verbose=False):
		"""
		Environment constructor
		:param board: string name of the board being played
		:param matchup: string name of file containing players in the game (Corresponding to policies)
		:param verbose: boolean whether to print large amounts of text
		"""
		
		# parse .mu file into player types
		self.verbose = verbose
		if verbose:
			print('Reading file: {}'.format('./matchups/' + matchup + '.mu'))
		with open('./matchups/' + matchup + '.mu') as fmu:
			line = fmu.read()
			fmu.close()
		self.player_names = line.split(', ')
		self.player_names[-1] = self.player_names[-1].strip('\n')
		num_players = len(self.player_names)

		self.game = gm.RiskGame(board,num_players,verbose)

		for player_num in range(num_players):
			isAgent = False

			# TODO - refer to list of agents
			# if self.player_names[player_num] is "agent":
			# 	isAgent = True

			# TODO - add policy with player
			new_player = self.game.get_player_from_id(player_num)
			new_player.set_player_attributes(self.player_names[player_num], isAgent)
			if verbose:
				new_player.print_player_details()

		self.agent_list = []  # Maps agent id to agent object
		# ag_id = 0
		# for player_name in self.player_names:  # Same order as player_id in game
		for ag_id in range(len(self.player_names)):
			unique_player = True
			for ag_id_2 in range(len(self.player_names)):  # Create pointer to earlier agent if the same
				if (self.player_names[ag_id] == self.player_names[ag_id_2]) and (ag_id_2 < ag_id):
					if self.verbose:
						print("Creating player {}: {}, as reference to player {}".format(ag_id, self.player_names[ag_id], ag_id_2))
					self.agent_list.append(self.agent_list[ag_id_2])
					unique_player = False
					break

			if unique_player == True:
				if self.verbose:
					print("Creating player {}: {}".format(ag_id, self.player_names[ag_id]))
				self.agent_list.append(self.player_name_to_agent(self.player_names[ag_id], ag_id))
			# ag_id += 1

	def player_name_to_agent(self, player_name, player_id):
		"""
		Create an agent based on a pre-specified player name
		return:
		Whether the agent was created successfully
		"""
		if player_name == "conservative":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "max_success", "skip_fortify", self.verbose)
		elif player_name == "random":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "random_attack", "skip_fortify", self.verbose)
		elif player_name == "agent":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "three_layer_attack_net", "skip_fortify", self.verbose)
		elif player_name == "aggressive":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "army_difference", "skip_fortify", self.verbose)
		elif player_name == "im_learner_1":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "three_layer_attack_net", "skip_fortify", self.verbose)
		elif player_name == "im_learner_2":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "two_layer_attack_net", "skip_fortify", self.verbose)

		print("Player name not recognized")
		return None



	def play_game(self, player_id_list, player_action_list, train, verbose=False):
		"""
		Generates a state vector for each player in player_id_list for what they saw in each action type
		:param player_id_list: the ids of the players to generate state vectors for
		:param player_action_list: list of lists: the action types for which state vectors are desired, per player
		:return states: list of lists of numpy arrays of states seen during gameplay
		:return actions: list of lists of numpy arrays, actions taken in those states
		:return rewards: list of lists of numpy arrays, the rewards earned by those actions
		"""

		if train:
			g_states = {}  # maps player_id to states seen by that player
			g_actions = {}  # maps player_id to actions taken by that player
			# g_rewards = {}  # maps player_id to rewards earned by action
			g_masks = {}  # maps player_id to valid masks in states

			#### Only used by array implementation
			g_steps = {}
			num_recorded_players = 0
			for player in range(max(player_id_list)+1):
				if player in player_id_list:
					
					player_states = {}  # maps action_type to states seen by that player, action
					player_actions = {}  # maps action_type to actions taken by that player_action
					# player_rewards = {}  # maps action_type to rewards earned by that player_action
					player_masks = {}  # maps action_type to masks for that player, action
					player_steps = {}
					
					g_states[player] = player_states
					g_actions[player] = player_actions
					# g_rewards[player] = player_rewards
					g_masks[player] = player_masks

					g_steps[player] = player_steps
					
					for action_type in range(max(player_action_list[num_recorded_players])+1):
						if action_type in player_action_list[num_recorded_players]:
							############ NOTE: Large lists dramatically slow down python, using arrays with resizing instead
							# player_action_states = np.zeros((TIMEOUT_STATES, self.game.graph.total_territories))
							# player_action_actions = np.zeros((TIMEOUT_STATES, len(self.game.graph.edge_list)))
							# player_action_rewards = np.zeros((TIMEOUT_STATES, 1))

							################### TOO SLOW TO CONVERT BETWEEN ARRAYS AND LISTS ################
							player_action_states = []
							player_action_actions = []
							# player_action_rewards = []
							player_action_masks = []
							player_action_steps = 0

							############## New code #################
							g_states[player][action_type] = player_action_states
							g_actions[player][action_type] = player_action_actions
							# g_rewards[player][action_type] = player_action_rewards
							g_masks[player][action_type] = player_action_masks

							g_steps[player][action_type] = player_action_steps
					num_recorded_players += 1

		num_states = 0
		num_turns = 0
		old_player_turn = -1

		game_state, valid = self.game.random_start(verbose)
		raw_state, new_player_turn, new_action_type, u_armies, r_edge, winner = self.unpack_game_state(game_state)

		while (winner == -1):
			if valid:
				num_states += 1
				if train:
					if num_states > TIMEOUT_STATES:
						print("Game exceeded timeout states: {}".format(TIMEOUT_STATES))
						return None, None, None, None, None, False
			old_action_type = new_action_type
			old_player_turn = new_player_turn
			state = self.translate_2_state(raw_state, old_player_turn)
			action, valid_mask = self.game_state_2_action(state, old_player_turn, old_action_type, train)

			if verbose:
				print("Player {} performs {} for action type {}".format(old_player_turn, action, old_action_type))
			game_state, valid = self.game.act(action, old_player_turn, old_action_type)
			raw_state, new_player_turn, new_action_type, u_armies, r_edge, winner = self.unpack_game_state(game_state)

			if verbose:
				if (old_player_turn != new_player_turn):
					num_turns += 1
			############### KEEP FOR DEBUGGING ################
			# if action_type == ActionType.ATTACK:
			# 	print("first mask")
			# 	print(valid_mask)


			if train and valid and (old_player_turn in player_id_list):
				if (int(old_action_type) in player_action_list[player_id_list.index(old_player_turn)]):  ## Non-valid states are exactly what was passed

					############ TODO: Save for troubleshooting
					# print("Appending player: {}, action: {}".format(old_player_turn, old_action_type))

					############# Create one-hot vectors of attack ################
					if old_action_type == ActionType.ALLOT:
						action_vector = np.zeros(self.game.graph.total_territories)
					elif old_action_type == ActionType.ATTACK:
						action_vector = np.zeros(len(self.game.graph.edge_list))
					elif old_action_type == ActionType.FORTIFY:
						action_vector = np.zeros(len(self.game.graph.edge_list))
					action_vector[action] = 1

					############### Faster method ################3
					# g_states[player_turn][int(action_type)][g_steps[player_turn][int(action_type)]] = state
					# g_actions[player_turn][int(action_type)][g_steps[player_turn][int(action_type)]] = action
					
					############ KEEP FOR DEBUGGING ################
					# print(g_steps[player_turn][int(action_type)])

					################## OLD: TOO SLOW #####################
					g_states[old_player_turn][int(old_action_type)].append(state)
					g_actions[old_player_turn][int(old_action_type)].append(action_vector)
					g_masks[old_player_turn][int(old_action_type)].append(valid_mask)

					############ KEEP FOR DEBUGGING ################
					# print("second mask")
					# print(valid_mask)

					# if old_player_turn == winner:
					# 	reward = 1000
					# else:
					# 	reward = 0  ################ NOTE: State-based rewards will be computed in post-processing by generate_reinforcement_learning_episodes #######
					################### OLD: TOO SLOW ###################
					# g_rewards[old_player_turn][int(old_action_type)].append(reward)

					##################### Faster method ###################
					# g_rewards[player_turn][int(action_type)][g_steps[player_turn][int(action_type)]] = reward
					g_steps[old_player_turn][int(old_action_type)] += 1


			if verbose:
				print("{}, player:{}, {}, winner:{}".format(raw_state, old_player_turn, old_action_type, winner))

		############## Resize all return arrays ##############
		# for player in player_id_list:
		# 	for action_type in player_action_list[player_id_list.index(player)]:
		# 		g_states[player][action_type] = np.resize(g_states[player][action_type], (g_steps[player][action_type], self.game.graph.total_territories))

			# raw_state, player_turn, action_type, u_armies, r_edge, winner
		if verbose:
			print("Player {} wins in {} turns".format(winner, num_turns))

		if train:
			# return winner, g_states, g_actions, g_rewards, g_masks, g_steps, True
			return winner, g_states, g_actions, g_masks, g_steps, True
		else:
			return winner, None, None, None, None, True

	def unpack_game_state(self, game_state):
		"""
		Returns all the items in the game_state tuple
		"""
		return ((info) for info in game_state)

	def translate_2_state(self, raw_state, player_id):
		"""
		Translates a state vector into positive and negative elements
		############### NOTE ##################
		Currently, all enemies are simply represented as negative
		##################################################
		:param: raw_state - raw_state from the game
		:param: player_id - the player to transform the state for
		:return state: the state to be fed into a q function
		"""
		state = np.zeros(len(raw_state))
		for territory in range(len(raw_state)):
			if raw_state[territory][0] == player_id:
				state[territory] = raw_state[territory][1]
			else:  
				state[territory] = -raw_state[territory][1]
		return state


	def game_state_2_action(self, state, player_id, action_type, train):
		"""
		Returns an action based on the agent corresponding to player_id and action type
		"""
		agent = self.agent_list[player_id]

		# Wrap state to work with general q functions
		state = np.array([state])

		########### TODO wrap strategies around q functions  ###########

		if action_type == ActionType.ALLOT:
			valid_mask = self.allot_valid(state)
			# return agent.allot_q_func.get_action(state, valid_mask)
			# q = agent.allot_q_func.call_Q(state)
			# action = utils.choose_by_weight(np.multiply(valid_mask, q))
			action = agent.allot_q_func.get_action(state, valid_mask, update=train)
		elif action_type == ActionType.ATTACK:
			valid_mask = self.attack_valid(state)
			action = agent.attack_q_func.get_action(state, valid_mask, update=train)
			# q = agent.attack_q_func.call_Q(state, valid_mask=valid_mask)
			# # q_valid = utils.validate_q_func_for_argmax(q, valid_mask)
			# # action = np.argmax(q_valid)
			# if train:
			# 	action = utils.epsilon_greedy_valid(q, valid_mask, TRAIN_EPSILON)
			# else:
			# 	action = utils.epsilon_greedy_valid(q, valid_mask, TEST_EPSILON)
		elif action_type == ActionType.REINFORCE:
			q = agent.reinforce_q_func.call_Q(state)
			action = q
			######## Replace once true reinforce in place
			return action, None
		elif action_type == ActionType.FORTIFY:
			# q = agent.fortify_q_func.call_Q(state)
			######### Activate this if anything is not using skip_fortify ######
			valid_mask = self.fortify_valid(state)
			# q_valid = utils.validate_q_func_for_argmax(q, valid_mask)
			# action = np.argmax(q_valid)
			action = agent.fortify_q_func.get_action(state, valid_mask, update=train)
			###############################3#############
			# action = np.argmax(q)

		else:
			print("ENVIRONMENT ERROR: Action type cannot be interpreted")
			return None

		return action, valid_mask

	def allot_valid(self, state_vector):
		"""
		Returns a mask of valid allotments
		Mask is applied to list of territories
		"""
		valid_mask = np.zeros(len(state_vector[0]))
		good_mask = False
		for state in range(len(state_vector[0])):
			armies = state_vector[0][state]
			if armies > 0 and armies < self.game.graph.MAX_ARMIES:
				good_mask = True
				valid_mask[state] = 1

		######### TODO: allow the game to accept over-sized allotment, to no effect
		############### except for losing the unallocated army
		########## Remove this warning statement once complete
		if good_mask == False:
			# print("VALIDATION WARNING: All owned territories have {} armies".format(self.game.graph.MAX_ARMIES))
			valid_mask = np.zeros(len(state_vector[0]))
			for state in range(len(state_vector[0])):
				armies = state_vector[0][state]
				if armies > 0:
					valid_mask[state] = 1

		# print("Allot valid mask is: {}".format(valid_mask))
		return valid_mask

	def attack_valid(self, state_vector):
		"""
		Returns a mask of valid attacks
		Mask is applied to list of edges
		"""
		total_edges = len(self.game.graph.edge_list)
		valid_mask = np.zeros(total_edges)
		for edge in range(total_edges-1):
			t1 = self.game.graph.edge_list[edge][0]
			t2 = self.game.graph.edge_list[edge][1]
			if np.sign(state_vector[0][t1]) == np.sign(state_vector[0][t2]):
				## Invalid edge
				continue
			if state_vector[0][t1] > 0:
				p_terr = t1
			else:
				p_terr = t2
			if state_vector[0][p_terr] > 1:
				valid_mask[edge] = 1

		valid_mask[-1] = 1 ## Pass action always valid
		# print("Attack valid mask is {}".format(valid_mask))
		return valid_mask

	def fortify_valid(self, state_vector):
		"""
		Returns a mask of valid fortifications
		Mask is applied to list of edges
		"""
		total_edges = len(self.game.graph.edge_list)
		valid_mask = np.zeros(total_edges)
		for edge in range(total_edges-1):
			t1 = self.game.graph.edge_list[edge][0]
			t2 = self.game.graph.edge_list[edge][1]
			arms_1 = state_vector[0][t1]
			arms_2 = state_vector[0][t2]
			if (arms_1 < 0) or (arms_2 < 0):
				## Invalid edge
				continue
			if arms_1 + arms_2 > 2:
				valid_mask[edge] = 1

		valid_mask[-1] = 1 ## Skip action always valid
		# print("Fortify valid mask is {}".format(valid_mask))
		return valid_mask



def imitation_learn(board, matchup, verbose, print_game, train=False, num_games=100, num_epochs=100000):
	"""
	Plays num_games games between the specified matchup
	If train is specified, uses these games to train a model
	The model is not an argument, it is specified below
	:param board: the .risk file in the boards folder to import
	:param matchup: the .matchup file in the matchups folder to import, determines which players are being used
	:param verbose: whether to print, recommended
	:param print_game: prints every action for the games, used for debugging
	:param num_games: number of games to run
	:param epochs: number of epochs to train for
	:return: nothing
	"""

	USEFUL_LIFE = 1000
	VALIDATION_GAMES = 10
	MODEL_INSTANCE = '0'
	LEARNING_RATE = 0.0001

	######### 0-44 is conservative-conservative ###############
	######### 0... is conservative-aggressive #################

	environment = RiskEnv(board, matchup, verbose)


	if train == False:
		generate_winners_episodes(environment, num_games, verbose=verbose, print_game=print_game)

	if train:

		################# This can be modified ####################
		# from q_funcs.attack import three_layer_attack_net
		# training_policy = three_layer_attack_net.ThreeLayerAttackNet(environment.game.graph.total_territories, environment.game.graph.edge_list, MODEL_INSTANCE, -1, LEARNING_RATE)
		
		from q_funcs.attack import two_layer_attack_net
		training_policy = two_layer_attack_net.TwoLayerAttackNet(environment.game.graph.total_territories, environment.game.graph.edge_list, MODEL_INSTANCE, -1, LEARNING_RATE)

		##########################################################

		num_players = environment.game.num_players


		all_player_list = range(num_players)
		all_players_attack_action = []
		for player in all_player_list:
			all_players_attack_action.append([int(ActionType.ATTACK)])
		######### three player list ##########
		player_list = range(num_players)
		players_attack_action = []
		for player in player_list:
			players_attack_action.append([int(ActionType.ATTACK)])


		epoch = 0
		train_loss = []
		while epoch < num_epochs:

			################### GENERATE TRAINING SET #####################
			if (epoch%USEFUL_LIFE) == 0: 
				if verbose:
					print("Generating {} new games".format(num_games), end='')
				_, t_states, t_actions, t_masks, _ = generate_winners_episodes(environment, num_games, player_list, players_attack_action, train=True)
				if verbose:
					print(" complete, training for {} epochs".format(USEFUL_LIFE))

			batch_size = len(t_states)
			batch = np.random.permutation(batch_size)
			for index in range(batch_size):
				batch_state = t_states[batch[index]]
				batch_action = t_actions[batch[index]]
				batch_mask = t_masks[batch[index]]
				train_loss.append(np.mean(training_policy.batch_train(batch_state, batch_action, batch_mask)))

			epoch += 1

			################## CREATE TRAINING LOSS PLOT ##############3
			if (epoch%(USEFUL_LIFE/100)) == 0:
				train_mean = np.mean(train_loss)
				train_std = np.std(train_loss)
				plt.errorbar(epoch, train_mean, yerr=train_std, fmt='--o', color='red')
				plt.draw()
				train_loss = []

			################### GENERATE VALIDATION SET #################
			if (epoch%(USEFUL_LIFE/10)) == 0:
				v_winners, v_states, v_actions, v_masks, _ = generate_winners_episodes(environment, VALIDATION_GAMES, player_list, players_attack_action, train=True)
				v_loss = []
				v_batch_size = len(v_states)
				for index in range(v_batch_size):
					v_loss.append(np.mean(training_policy.batch_train(v_states[index], v_actions[index], v_masks[index], update=False)))
				loss_mean = np.mean(v_loss)
				loss_std = np.std(v_loss)
				plt.errorbar(epoch, loss_mean, yerr=loss_std, fmt='--o', color='blue')
				plt.title("Loss: red-training, blue-validation")
				plt.xlabel('Training epochs')
				plt.ylabel('Mean validation loss over {} games'.format(VALIDATION_GAMES))
				plt.draw()
				if (verbose):
					print("Completed epoch {}".format(epoch))
					print("Validation loss: {}".format(loss_mean))

		training_policy.close()
		save_path = './plots/' + MODEL_INSTANCE + '-training'
		plt.savefig(save_path)
		plt.show()
		# states, acts, rewards = environment.play_game(0,1,verbose)

	return



def generate_winners_episodes(env, num_games, player_list=None, player_action_list=None, train=False, verbose=False, print_game=False):
	"""
	Runs num_games and returns lists describing the games, depending on player_list and player_action_list
	:param env: The environment object holding the game to be played
	:param num_games: the number of games to play
	:param player_list: which players information is desired about
	:param player_action_list: which actions information is needed for those players
	:param train: Whether to train (true) or test (false)
	:param print_game: whether to print game information, used for debugging, not recommended
	:return record: ndarray of size (num_player,), how many games each player won, respectively
	:return states: list of ndarrays containing the states seen by the winner of the game
	:return actions: list of ndarrays containing the actions (one-hot) taken by the winner of each game
	:return rewards: list of ndarrays containing the rewards earned by the winner of the game 
	:return masks: list of ndarrays containing the valid actions available to the winner for each action taken
	:return num_states: list of the number of total states the game advanced through
	"""
	##################### IN USE ################

	if train:
		imitation_states = []
		imitation_actions = []
		imitation_masks = []

	num_players = env.game.num_players
	record = np.zeros(num_players)
	for i in range(num_games):
		timeout = False
		while not timeout:
			# winner, states, actions, rewards, masks, num_states, timeout = env.play_game(player_list, player_action_list, train, print_game)
			winner, states, actions, masks, num_states, timeout = env.play_game(player_list, player_action_list, train, print_game)
		
		##################### Specific to attack action ####################
		################## NOTE: Cast to np.array upon return ##################
		# imitation_states = np.concatenate([imitation_states, np.array(states[winner][int(ActionType.ATTACK)])])
		# print(states[winner][int(ActionType.ATTACK)].shape)
		# print(np.array(actions[winner][int(ActionType.ATTACK)]).shape)
		# print(np.array(masks[winner][int(ActionType.ATTACK)]).shape)

		############### TODO: Keep as reference for types
		# print(type(imitation_states))
		# print(type(imitation_actions))
		# print(type(np.array(imitation_states)))
		# print(type(np.array(imitation_actions)))

		################ Faster method #########
		# imitation_states = np.concatenate([imitation_states, states[winner][int(ActionType.ATTACK)]])
		# imitation_actions = np.concatenate([imitation_actions, actions[winner][int(ActionType.ATTACK)]])

		# print(states[winner][int(ActionType.ATTACK)])
		# print(actions[winner][int(ActionType.ATTACK)])

		################# IN USE #################
		if train and (winner in player_list):
			imitation_states.append(np.array(states[winner][int(ActionType.ATTACK)]))
			imitation_actions.append(np.array(actions[winner][int(ActionType.ATTACK)]))
			imitation_masks.append(np.array(masks[winner][int(ActionType.ATTACK)]))

		for player in range(len(record)):
			if player == winner:
				record[player] +=1
		if verbose:
			if i%(num_games/10) == 0:
				print("Completed game {}".format(i))
	if verbose:
		for player in range(len(record)):
			print("Player {} won {} games".format(player, record[player]))

	if train:
		return record, imitation_states, imitation_actions, imitation_masks, num_states
	else:
		return record, None, None, None, None, None


def reinforcement_learn(board, matchup, verbose, num_games=100, n=250):
	"""
	Plays num_games games between the specified matchup
	If train is specified, uses these games to train a model
	The model is not an argument, it is specified below
	:param board: the .risk file in the boards folder to import
	:param matchup: the .matchup file in the matchups folder to import, determines which players are being used
	:param verbose: whether to print, recommended
	:param print_game: prints every action for the games, used for debugging
	:param num_games: number of games to run
	:param epochs: number of epochs to train for
	:return: nothing
	"""

	environment = RiskEnv(board, matchup, verbose)

	MODEL_INSTANCE = '0'
	LEARNING_RATE = 0.0001

	################# This can be modified ####################
	# from q_funcs.attack import three_layer_attack_net
	# training_policy = three_layer_attack_net.ThreeLayerAttackNet(environment.game.graph.total_territories, environment.game.graph.edge_list, MODEL_INSTANCE, -1, LEARNING_RATE)
	# from q_funcs.attack import two_layer_attack_net
	# training_policy = two_layer_attack_net.TwoLayerAttackNet(environment.game.graph.total_territories, environment.game.graph.edge_list, MODEL_INSTANCE, -1, LEARNING_RATE)
	training_policy = environment.agent_list[0].attack_q_func

	plt.title("Loss w.r.t. winner:")
	plt.xlabel('Training games')
	plt.ylabel('Mean training loss over {} games'.format(num_games))

	##########################################################

	num_players = environment.game.num_players

	all_player_list = range(num_players)
	all_players_attack_action = []
	for player in all_player_list:
		all_players_attack_action.append([int(ActionType.ATTACK)])
	######### n player list ##########
	player_list = range(num_players)
	players_attack_action = []
	for player in player_list:
		players_attack_action.append([int(ActionType.ATTACK)])

	timeout = False

	train_loss = []
	for game in range(num_games):
		reinforcement_states = []
		reinforcement_actions = []
		reinforcement_masks = []
		reinforcement_targets = []
		batch_size = []

		for player in player_list:


			winner, states, actions, masks, targets, steps = generate_reinforcement_learning_episodes(environment, 1, player_list, players_attack_action, verbose)

			reinforcement_states.append(np.array(states[player][int(ActionType.ATTACK)][0]))
			reinforcement_actions.append(np.array(actions[player][int(ActionType.ATTACK)][0]))
			reinforcement_masks.append(np.array(masks[player][int(ActionType.ATTACK)][0]))
			reinforcement_targets.append(np.array(targets[player][int(ActionType.ATTACK)][0]))
			batch_size.append(steps[player][int(ActionType.ATTACK)][0])

		for player in player_list:
			# print("States:")
			# print(reinforcement_states[player])
			# print("Actions:")
			# print(reinforcement_actions[player])
			# print("Masks:")
			# print(reinforcement_masks[player])
			# print("Targets:")
			# print(reinforcement_targets[player])
			# print("Batch size:")
			# print(batch_size[player])

			if player == winner and (batch_size[player] != 0):
				train_loss.append(np.mean(training_policy.batch_train(state_vector=reinforcement_states[player], action_vector=reinforcement_targets[player], valid_mask=reinforcement_masks[player], update=True, batch_size=batch_size[player])))
			elif (batch_size[player] != 0):
				training_policy.batch_train(state_vector=reinforcement_states[player], action_vector=reinforcement_targets[player], valid_mask=reinforcement_masks[player], update=True, batch_size=batch_size[player])

		################## CREATE TRAINING LOSS PLOT ##############3
		if (game%(num_games/100) == 0) and not (game == 0):
			train_mean = np.mean(train_loss)
			train_std = np.std(train_loss)
			plt.errorbar(game, train_mean, yerr=train_std, fmt='--o', color='red')
			plt.draw()
			train_loss = []
			if (verbose):
				print("Completed game {}".format(game))
				print("training loss w.r.t. winner: {}".format(train_mean))

	training_policy.close()
	save_path = './plots/' + MODEL_INSTANCE + '-REINFORCE'
	plt.savefig(save_path)
	plt.show()
	# states, acts, rewards = environment.play_game(0,1,verbose)	


	return 

def generate_reinforcement_learning_episodes(env, num_games, player_list=None, player_action_list=None, verbose=False):
	"""
	Similar to generate_winners_episode, but includes targets for a2c
	"""
	GAMMA = 1
	################ Should be significantly larger than army difference can change from turn to turn
	WINNING_REWARD = 10

	# winner, states, actions, rewards, masks, num_states, timeout = env.play_game(player_list, player_action_list, train=True, verbose=False)

	e_states = {}
	e_actions = {}
	e_masks = {}
	e_targets = {}
	e_steps = {}


	for game in range(num_games):
		# winner, states, actions, rewards, masks, num_states, timeout = env.play_game(player_list, player_action_list, train=True, verbose=False)
		timeout = False
		while not timeout:
			winner, states, actions, masks, num_states, timeout = env.play_game(player_list, player_action_list, train=True, verbose=False)

		for player in player_list:

			player_states = {}
			player_actions = {}
			player_masks = {}
			player_targets = {}
			player_steps = {}

			e_states[player] = player_states
			e_actions[player] = player_actions
			e_masks[player] = player_masks
			e_targets[player] = player_targets
			e_steps[player] = player_steps


			for action_type in player_action_list[player]:
				player_action_states = []
				player_action_actions = []
				player_action_masks = []
				player_action_targets = []
				player_action_steps = []

				e_states[player][action_type] = player_action_states
				e_actions[player][action_type] = player_action_actions
				e_masks[player][action_type] = player_action_masks
				e_targets[player][action_type] = player_action_targets
				e_steps[player][action_type] = player_action_steps

		# for action_type in player_action_list[winner]:
		action_type = int(ActionType.ATTACK)
		# if (winner in player_list):
		for player in player_list:
			################# Currently only for single action type - attack ####################
			e_states[player][action_type].append(np.array(states[player][int(ActionType.ATTACK)]))
			e_actions[player][action_type].append(np.array(actions[player][int(ActionType.ATTACK)]))
			e_masks[player][action_type].append(np.array(masks[player][int(ActionType.ATTACK)]))
			e_steps[player][action_type].append(num_states[player][int(ActionType.ATTACK)])


# ##################### For use with a2c in hw3, not yet implemented ############################
# 			T = winner_steps[action_type][game]
# 			R = np.zeros(T)
# 			V_end = np.zeros(T)
# 			# actor_target = np.zeros((T, ACTION_SPACE))
# 			actor_target = np.zeros((T, len(env.game.graph.edge_list)))
# 			for t in reversed(range(T)):
# 			# Note: V_end = 0 case is default, handled by zero initialization
# 				if (t+n < T):
# 					V_end[t] = e_Vw[t+n]
# 				R[t] = (GAMMA**n) * V_end[t]
# 				for k in range(n):
# 					if (t+k < T):
# 						R[t] += (GAMMA**k) * rewards[winner][action_type][t+k]
# 				actor_target[t, :] = R[t] - e_Vw[t] 
# 				actor_target[t,:] = np.multiply(actor_target[t, :], actions[winner][action_type][t])

# 			winner_R[action_type].append(R)
# 			winner_target[action_type].append(actor_target)
			T = e_steps[player][action_type][game]
			returns_scalar = np.zeros(T)
			game_returns = np.zeros((T, len(env.game.graph.edge_list)))
			for t in reversed(range(T)):
				if (t == T-1):
					if player == winner:
						reward = WINNING_REWARD
					else:
						reward = -WINNING_REWARD
					returns_scalar[t] = reward
					game_returns[t, :] = returns_scalar[t]
				else:
					reward = change_in_army_difference_reward(e_states[player][action_type][game][t+1], e_states[player][action_type][game][t])
					returns_scalar[t] = reward + GAMMA*returns_scalar[t+1]
					game_returns[t, :] = returns_scalar[t]
				game_returns[t, :] = np.multiply(game_returns[t,:], e_actions[player][action_type][game][t])
			################## In order to scale properly ###############
			game_returns /= 2*WINNING_REWARD
			###########################################
			e_targets[player][action_type].append(game_returns)

		# else:
		# 	return winner, e_states, e_actions, e_masks, None, None, None

	return winner, e_states, e_actions, e_masks, e_targets, e_steps


def change_in_army_difference_reward(new_state, old_state):
	"""
	This function returns the change of army difference from the old state to the new state
	"""
	old_state_difference = compute_army_difference(old_state)
	new_state_difference = compute_army_difference(new_state)
	return new_state_difference - old_state_difference

def compute_army_difference(state_vector):
	"""
	This function computes player_armies - opponent_armies
	"""
	opponent_armies = 0
	player_armies = 0
	for territory in range(len(state_vector)):
		armies = state_vector[territory]
		if armies < 0:
			opponent_armies -= armies  # Negate and add
		else:
			player_armies += armies
	return player_armies-opponent_armies


def parse_arguments():
	"""
	This function helps main read command line arguments
	:params : none
	:return Parser: parser object containing arguments passed from command line
	"""
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('-b', dest='board', type=str, default='Original')
	parser.add_argument('-m', dest='matchup', type=str, default="balance_7")
	parser.add_argument('-v', dest='verbose', action='store_true', default=False)
	parser.add_argument('-p', dest='print_game', action='store_true', default=False)
	parser.add_argument('-t', dest='train', action='store_true', default=False)
	parser.add_argument('--num-games', dest='num_games', default=100)
	parser.add_argument('--num-epochs', dest='num_epochs', default=100000)
	parser.add_argument('--training-type', dest='training_type', default='I')
	parser.set_defaults(verbose=True)
	parser.set_defaults(print_game=False)
	parser.set_defaults(train=False)
	return parser.parse_args()



def main(args):
	"""
	This function initializes a game inside an environment and uses is to train an agent
	:param args: Command line arguments
	:return : this function does not return
	"""

	args = parse_arguments()
	board = args.board
	matchup = args.matchup
	verbose = args.verbose
	print_game = args.print_game
	train = args.train
	training_type = args.training_type
	num_games = int(args.num_games)
	num_epochs = int(args.num_epochs)


	if training_type == "R":
		reinforcement_learn(board, matchup, verbose, num_games)
	else:
		imitation_learn(board, matchup, verbose, print_game, train, num_games, num_epochs)



import signal
def signal_handler(signal, frame):
    print('WARNING: PLOT NOT SAVED, SCREENSHOT TO KEEP')
    plt.show()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# This is something you have to do in Python... I don't really know why	
if __name__ == '__main__':
	main(sys.argv)