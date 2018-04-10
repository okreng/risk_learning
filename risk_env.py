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
			if self.player_names[player_num] is "agent":
				isAgent = True

			# TODO - add policy with player
			new_player = self.game.get_player_from_id(player_num)
			new_player.set_player_attributes(self.player_names[player_num], isAgent)
			if verbose:
				new_player.print_player_details()

		self.agent_list = []  # Maps agent id to agent object
		ag_id = 0
		for player_name in self.player_names:  # Same order as player_id in game
			self.agent_list.append(self.player_name_to_agent(player_name, ag_id))
			ag_id += 1

	def player_name_to_agent(self, player_name, player_id):
		"""
		Create an agent based on a pre-specified player name
		return:
		Whether the agent was created successfully
		"""
		if self.verbose:
			print("Creating player {}: {}".format(player_id, player_name))
		if player_name == "conservative":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "max_success", "skip_fortify", self.verbose)
		elif player_name == "random":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "random_allot", "random_attack", "skip_fortify", self.verbose)
		elif player_name == "agent":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "three_layer_attack_net", "skip_fortify", self.verbose)
		if player_name == "aggressive":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "army_difference", "skip_fortify", self.verbose)

		print("Player name not recognized")
		return None



	def play_game(self, player_id_list, player_action_list, verbose=False):
		"""
		Generates a state vector for each player in player_id_list for what they saw in each action type
		:param player_id_list: the ids of the players to generate state vectors for
		:param player_action_list: list of lists: the action types for which state vectors are desired, per player
		:return states: list of lists of numpy arrays of states seen during gameplay
		:return actions: list of lists of numpy arrays, actions taken in those states
		:return rewards: list of lists of numpy arrays, the rewards earned by those actions
		"""

		num_states = 0
		g_states = {}  # maps player_id to states seen by that player
		g_actions = {}  # maps player_id to actions taken by that player
		g_rewards = {}  # maps player_id to rewards earned by action
		g_masks = {}  # maps player_id to valid masks in states

		#### Only used by array implementation
		# g_steps = {}
		num_recorded_players = 0
		for player in range(max(player_id_list)+1):
			if player in player_id_list:
				
				player_states = {}  # maps action_type to states seen by that player, action
				player_actions = {}  # maps action_type to actions taken by that player_action
				player_rewards = {}  # maps action_type to rewards earned by that player_action
				player_masks = {}  # maps action_type to masks for that player, action
				player_steps = {}
				
				g_states[player] = player_states
				g_actions[player] = player_actions
				g_rewards[player] = player_rewards
				g_masks[player] = player_masks

				# g_steps[player] = player_steps
				
				for action_type in range(max(player_action_list[num_recorded_players])+1):
					if action_type in player_action_list[num_recorded_players]:
						############ NOTE: Large lists dramatically slow down python, using arrays with resizing instead
						# player_action_states = np.zeros((TIMEOUT_STATES, self.game.graph.total_territories))
						# player_action_actions = np.zeros((TIMEOUT_STATES, len(self.game.graph.edge_list)))
						# player_action_rewards = np.zeros((TIMEOUT_STATES, 1))

						################### TOO SLOW TO CONVERT BETWEEN ARRAYS AND LISTS ################
						player_action_states = []
						player_action_actions = []
						player_action_rewards = []
						player_action_masks = []
						# player_action_steps = 0

						############## New code #################
						g_states[player][action_type] = player_action_states
						g_actions[player][action_type] = player_action_actions
						g_rewards[player][action_type] = player_action_rewards
						g_masks[player][action_type] = player_action_masks

						# g_steps[player][action_type] = player_action_steps
				num_recorded_players += 1

		game_state, valid = self.game.random_start(verbose)
		raw_state, new_player_turn, new_action_type, u_armies, r_edge, winner = self.unpack_game_state(game_state)

		while (winner == -1):
			if valid:
				num_states += 1
				if num_states > TIMEOUT_STATES:
					print("Game exceeded timeout states: {}".format(TIMEOUT_STATES))
					return None, None, None, None, None, False
			old_action_type = new_action_type
			old_player_turn = new_player_turn
			state = self.translate_2_state(raw_state, old_player_turn)
			action, valid_mask = self.game_state_2_action(state, old_player_turn, old_action_type)

			if verbose:
				print("Player {} performs {} for action type {}".format(old_player_turn, action, action_type))
			game_state, valid = self.game.act(action, old_player_turn, old_action_type)
			raw_state, new_player_turn, new_action_type, u_armies, r_edge, winner = self.unpack_game_state(game_state)

			############### KEEP FOR DEBUGGING ################
			# if action_type == ActionType.ATTACK:
			# 	print("first mask")
			# 	print(valid_mask)


			if valid and (old_player_turn in player_id_list):
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

					if old_player_turn == winner:
						reward = 1
					else:
						reward = 0
					################### OLD: TOO SLOW ###################
					g_rewards[old_player_turn][int(old_action_type)].append(reward)

					##################### Faster method ###################
					# g_rewards[player_turn][int(action_type)][g_steps[player_turn][int(action_type)]] = reward
					# g_steps[player_turn][int(action_type)] += 1


			if verbose:
				print("{}, player:{}, {}, winner:{}".format(raw_state, player_turn, action_type, winner))

		############## Resize all return arrays ##############
		# for player in player_id_list:
		# 	for action_type in player_action_list[player_id_list.index(player)]:
		# 		g_states[player][action_type] = np.resize(g_states[player][action_type], (g_steps[player][action_type], self.game.graph.total_territories))

			# raw_state, player_turn, action_type, u_armies, r_edge, winner
		if verbose:
			print("Player {} wins".format(winner))

		return winner, g_states, g_actions, g_rewards, g_masks, num_states, True

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

	def game_state_2_action(self, state, player_id, action_type):
		"""
		Returns an action based on the agent corresponding to player_id and action type
		"""
		agent = self.agent_list[player_id]

		# Wrap state to work with general q functions
		state = np.array([state])

		########### TODO wrap strategies around q functions  ###########

		if action_type == ActionType.ALLOT:
			q = agent.allot_q_func.call_Q(state)
			valid_mask = self.allot_valid(state)
			action = utils.choose_by_weight(np.multiply(valid_mask, q))
		elif action_type == ActionType.ATTACK:
			q = agent.attack_q_func.call_Q(state)
			valid_mask = self.attack_valid(state)
			q_valid = utils.validate_q_func_for_argmax(q, valid_mask)
			action = np.argmax(q_valid)
		elif action_type == ActionType.REINFORCE:
			q = agent.reinforce_q_func.call_Q(state)
			action = q
			######## Replace once true reinforce in place
			return action, None
		elif action_type == ActionType.FORTIFY:
			q = agent.fortify_q_func.call_Q(state)
			######### Activate this if anything is not using skip_fortify ######
			valid_mask = self.fortify_valid(state)
			q_valid = utils.validate_q_func_for_argmax(q, valid_mask)
			action = np.argmax(q_valid)
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



def parse_arguments():
	"""
	This function helps main read command line arguments
	:params : none
	:return Parser: parser object containing arguments passed from command line
	"""
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('-b', dest='board', type=str, default='Original')
	parser.add_argument('-m', dest='matchup', type=str, default="default")
	parser.add_argument('-v', dest='verbose', action='store_true', default=False)
	parser.add_argument('-p', dest='print_game', action='store_true', default=False)
	parser.add_argument('--num-games', dest='num_games', default=1)
	parser.add_argument('--num-epochs', dest='num_epochs', default=100)
	parser.set_defaults(verbose=False)
	parser.set_defaults(print_game=False)
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
	num_games = int(args.num_games)
	num_epochs = int(args.num_epochs)

	environment = RiskEnv(board, matchup, verbose)
	num_players = environment.game.num_players

	all_player_list = range(num_players)
	all_players_attack_action = []
	for player in all_player_list:
		all_players_attack_action.append([int(ActionType.ATTACK)])

	################ NOTE: Save to clarify form of query  ##########################
	# player_0_action_list = [[int(ActionType.ALLOT), int(ActionType.ATTACK)]]
	# player_0_action_list = [[int(ActionType.ATTACK)]]

	imitation_states = np.empty((0, environment.game.graph.total_territories))
	imitation_actions = np.empty((0, len(environment.game.graph.edge_list)))
	imitation_masks = np.empty((0, len(environment.game.graph.edge_list)))

	record = np.zeros(num_players)
	for i in range(num_games):
		timeout = False
		while not timeout:
			winner, states, actions, rewards, masks, num_turns, timeout = environment.play_game(all_player_list, all_players_attack_action, print_game)
		
		##################### Specific to attack action ####################
		################## NOTE: Cast to np.array upon return ##################
		# imitation_states = np.concatenate([imitation_states, np.array(states[winner][int(ActionType.ATTACK)])])
		# print(states[winner][int(ActionType.ATTACK)].shape)
		# print(np.array(actions[winner][int(ActionType.ATTACK)]).shape)
		# print(np.array(masks[winner][int(ActionType.ATTACK)]).shape)


		################ Faster method #########
		# imitation_states = np.concatenate([imitation_states, states[winner][int(ActionType.ATTACK)]])
		# imitation_actions = np.concatenate([imitation_actions, actions[winner][int(ActionType.ATTACK)]])

		# print(states[winner][int(ActionType.ATTACK)])
		# print(actions[winner][int(ActionType.ATTACK)])


		############### IN USE ###########
		imitation_states = np.concatenate([imitation_states, np.array(states[winner][int(ActionType.ATTACK)])])
		imitation_actions = np.concatenate([imitation_actions, np.array(actions[winner][int(ActionType.ATTACK)])])
		imitation_masks = np.concatenate([imitation_masks, np.array(masks[winner][int(ActionType.ATTACK)])])

		for player in range(len(record)):
			if player == winner:
				record[player] +=1
		if verbose:
			if i%100 == 0:
				print("Completed game {}".format(i))
	for player in range(len(record)):
		print("Player {} won {} games".format(player, record[player]))

############### TODO: Keep as reference for types
	# print(type(imitation_states))
	# print(type(imitation_actions))
	# print(type(np.array(imitation_states)))
	# print(type(np.array(imitation_actions)))

	from q_funcs.attack import three_layer_attack_net
	training_policy = three_layer_attack_net.ThreeLayerAttackNet(environment.game.graph.total_territories, environment.game.graph.edge_list, '0', -1, 0.0001)
	# print("Initialized three layer policy")

	# print(imitation_states.shape)
	# print(imitation_actions.shape)

	loss = []
	for epoch in range(num_epochs):
		loss.append(training_policy.batch_train(imitation_states, imitation_actions, imitation_masks))
		if epoch%(num_epochs/100) == 0:
			print("Completed epoch {}".format(epoch))
			mean = np.mean(loss)
			std = np.std(loss)
			plt.errorbar(epoch, mean, yerr=std, fmt='--o')
			plt.title("Mean loss over training")
			plt.xlabel('Training epochs')
			plt.ylabel('Mean loss every {} epochs'.format(num_epochs/100))
			plt.draw()


	training_policy.close()

	plt.show()
	# states, acts, rewards = environment.play_game(0,1,verbose)

import signal
def signal_handler(signal, frame):
    print('Displaying training on exit:')
    plt.show()
    

signal.signal(signal.SIGTSTP, signal_handler)


# This is something you have to do in Python... I don't really know why	
if __name__ == '__main__':
	main(sys.argv)