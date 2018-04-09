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
		if player_name == "expert":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "max_success", "skip_fortify", self.verbose)
		elif player_name == "random":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "random_allot", "random_attack", "skip_fortify", self.verbose)
		elif player_name == "agent":
			return agent.Agent(player_id, self.game.graph.total_territories, self.game.graph.edge_list, "amass", "three_layer_attack_net", "skip_fortify", self.verbose)

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
		game_state, valid = self.game.random_start(self.verbose)
		raw_state, player_turn, action_type, u_armies, r_edge, winner = self.unpack_game_state(game_state)
		while (winner == -1):
			state = self.translate_2_state(raw_state, player_turn)
			# print(state)
			# TODO: Implement actual actions based off state
			# action = np.random.randint(0,2)
			action = self.game_state_2_action(state, player_turn, action_type)

			# print(action)
			if verbose:
				print("Player {} performs {} for action type {}".format(player_turn, action, action_type))
			# raw_state, player_turn, action_type, u_armies, r_edge, winner, valid = self.game.act(action, player_turn, action_type)
			game_state, valid = self.game.act(action, player_turn, action_type)
			raw_state, player_turn, action_type, u_armies, r_edge, winner = self.unpack_game_state(game_state)


			if verbose:
				print("{}, player:{}, {}, winner:{}".format(raw_state, player_turn, action_type, winner))


			# raw_state, player_turn, action_type, u_armies, r_edge, winner
		if verbose:
			print("Player {} wins".format(winner))

		return winner

	def unpack_game_state(self, game_state):
		"""
		Returns all the items in the game_state tuple
		"""
		return ((info) for info in game_state)

	def translate_2_state(self, raw_state, player_id):
		"""
		Translates a state vector into positive and negative elements
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
			action = np.argmax(q)
		elif action_type == ActionType.ATTACK:
			q = agent.attack_q_func.call_Q(state)
			action = np.argmax(q)
		elif action_type == ActionType.REINFORCE:
			q = agent.reinforce_q_func.call_Q(state)
			action = q
		elif action_type == ActionType.FORTIFY:
			q = agent.fortify_q_func.call_Q(state)
			action = np.argmax(q)

		else:
			print("ENVIRONMENT ERROR: Action type cannot be interpreted")
			return None

		return action

def parse_arguments():
	"""
	This function helps main read command line arguments
	:params : none
	:return Parser: parser object containing arguments passed from command line
	"""
	parser = argparse.ArgumentParser(description=
		'Risk Environment Argument Parser')
	parser.add_argument('-b', dest='board', type=str, default='Duel')
	parser.add_argument('-m', dest='matchup', type=str, default="default")
	parser.add_argument('-v', dest='verbose', action='store_true', default=False)
	parser.add_argument('-p', dest='print_game', action='store_true', default=False)
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

	environment = RiskEnv(board, matchup, verbose)
	environment.play_game(0,1,print_game)


	#################### For testing randomness ################
	wins = 0
	for i in range(1000):
		wins += environment.play_game(0,1,verbose)
	print(wins)

	# states, acts, rewards = environment.play_game(0,1,verbose)


# This is something you have to do in Python... I don't really know why	
if __name__ == '__main__':
	main(sys.argv)