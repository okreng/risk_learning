"""
This function contains the code to test if a
training Q function approximator can learn the most basic
attack state-action value function
Note, this file does not work with agents because 
it is only using the attack function type
"""

import sys, argparse, random
import numpy as np

from q_funcs.attack import linear_attack_net
from q_funcs.attack import max_success


def parse_arguments():
	parser = argparse.ArgumentParser(description='Agent Argument Parser')
	return parser.parse_args()

def main(args):
	"""
	Function to train the simplest type of attack network
	:param args: string command line arguments (none currently)
	:return : none
	"""

	args = parse_arguments()

	# Simplest graph possible
	state_vector = np.zeros(2)

	T = len(state_vector)
	state_vector = np.reshape(state_vector, (1, -1))

	######### Hyperparameters  ########3
	model_instance = '0'
	checkpoint_number = -1
	learning_rate = 0.001
	verbose = True
	gamma = 0.9

	agent = linear_attack_net.LinearAttackNet(T, model_instance, checkpoint_number, learning_rate)
	opponent = max_success.MaxSuccess(T)

	game_state = np.random.random_integers(1,12,size=(2))
	enemy_territory = np.random.random_integers(0,1)
	agent_territory = abs(1-enemy_territory)
	game_state[enemy_territory] = -game_state[enemy_territory]
	game_state = np.reshape(game_state,(1,-1))

	whose_turn = np.random.random_integers(0,1)
	winner = -1

	# print(game_state)

	# Initially set as a reference
	enemy_game_state = game_state 
	looking_ahead = False

	while(winner == -1):

		# Opponent strategy
		while whose_turn == 1:
			# Enemy acts the same regardless of real game action or simulated for reward
			action = opponent.call_Q(enemy_game_state)
			# Attack action, valid only if enemy has more than 1 army
			if action[1] > action[-1] and enemy_game_state[0, enemy_territory] > 1:  # attack action
				enemy_game_state = attack(enemy_game_state, enemy_territory, agent_territory)
				if game_state[0, agent_territory] == 0:  # Only true for game state, not enemy_game_state
					winner = 1
					break
			else:
				whose_turn = 0

		if winner == 1:
			print("Enemy wins")
			break

		# Player strategy
		while whose_turn == 0:
			if looking_ahead:
				if next_game_state[0, enemy_territory] = 0:  # This shouldn't be possible
					print("WARNING: Enemy lost on own turn during copy game")
					exit()
				else:
					reward = 0  # We know that the current action is pass (i.e. -1)
					target_q_func = agent.call_Q(enemy_game_state) # Run without update
					
					# TODO: Update indexing for improved state space
					loss_weights = np.zeros(5)
					loss_weights[4] = 1
					target = reward + gamma * max(target_q_func[1], target_q_func[-1])  # max value
					updated_q_func = agent.call_Q(state_vector=game_state, isTraining=True, action_taken=4, target=target, loss_weights=loss_weights)
					
					# Go back to real game once next state has been updated
					looking_ahead = 0
					complete_pass_action = True

			action = agent.call_Q(game_state)

			# TODO: Fix indexing for reasonable action space
			if action[1] > action[-1] and complete_pass_action == False:  # choose to attack
				next_game_state = attack(game_state, agent_territory, enemy_territory)
				if next_game_state[0, enemy_territory] = 0:  # Win condition for simple env
					# terminal Q update
					reward = 1
					loss_weights = np.zeros(5)
					loss_weights[1] = 1
					target = reward
					updated_q_func = updated_q_func = agent.call_Q(state_vector=game_state, isTraining=True, action_taken=1, target=target, loss_weights=loss_weights)

					# Set winner and break out of turn loop
					winner == 0
					break

				else:  # non-terminal q update
					reward = 0
					target_q_func = agent.call_Q(next_game_state)

					# TODO: Update indexing
					loss_weights = np.zeros(5)
					loss_weights[4] = 1
					target = reward + gamma * max(target_q_func[1], target_q_func[-1])
					updated_q_func = updated_q_func = agent.call_Q(state_vector=game_state, isTraining=True, action_taken=1, target=target, loss_weights=loss_weights)

				# Update the state of the game once complete, return to player turn while loop
				game_state = np.copy(next_game_state)

			elif complete_pass_action = False:  # choose to pass the turn, get target from next player's actions
				enemy_game_state = np.copy(next_game_state)  # Create a copy for simulated portion
				looking_ahead = True
			elif complete_pass_action = True:  # execute the pass_turn action
				game_state = np.copy(next_game_state)
				enemy_game_state = game_state  # Create a reference for actual game
			else:
				print("Game has missed condition, exiting")
				exit()

		if winner == 0:
			print("Player wins")
			break

	return

def attack(game_state, from_territory, to_territory):
	"""
	Function to determine the results of an attack
	:param game_state: the armies in each territory
	:param from_territory: the index of the territory attacking
	:param to_territory: the index of the territory defending
	"""

	if game_state[0, from_territory] < 0:
		enemy_territory = from_territory
	elif game_state[0, to_territory] < 0:
		enemy_territory = to_territory
	else:
		return game_state


	from_armies = abs(game_state[0,from_territory])
	to_armies = abs(game_state[0, to_territory])

	determine_attack = np.random.uniform()
	# new_game_state = np.zeros(len(game_state[0]))
	new_game_state = np.copy(game_state[0])
	if from_armies > 3: 
		if to_armies > 1: # Three-Two
			if determine_attack < (2890/7776):
				new_game_state[from_territory] = from_armies
				new_game_state[to_territory] = to_armies - 2
			elif determine_attack < (5165/7776):
				new_game_state[from_territory] = from_armies - 2
				new_game_state[to_territory] = to_armies
			else:
				new_game_state[from_territory] = from_armies - 1
				new_game_state[to_territory] = to_armies - 1
		elif to_armies == 1: # Three-One
			if determine_attack < (855/1296):
				new_game_state[from_territory] = from_armies
				new_game_state[to_territory] = to_armies - 1
			else:
				new_game_state[from_territory] = from_armies - 1
				new_game_state[to_territory] = to_armies
		else:
			return game_state

	elif from_armies == 3:  # Two-Two
		if to_armies > 1:
			if determine_attack < (295/1296):
				new_game_state[from_territory] = from_armies
				new_game_state[to_territory] = to_armies - 2
			elif determine_attack < (876/1296):
				new_game_state[from_territory] = from_armies - 2
				new_game_state[to_territory] = to_armies
			else:
				new_game_state[from_territory] = from_armies - 1
				new_game_state[to_territory] = to_armies - 1
		elif to_armies == 1: # Two-One
			if determine_attack < (125/216):
				new_game_state[from_territory] = from_armies
				new_game_state[to_territory] = to_armies - 1
			else:
				new_game_state[from_territory] = from_armies - 1
				new_game_state[to_territory] = to_armies
		else:
			return game_state

	elif from_armies == 2: 
		if to_armies > 1:  # One-Two
			if determine_attack < (55/216):
				new_game_state[from_territory] = from_armies
				new_game_state[to_territory] = to_armies - 1
			else:
				new_game_state[from_territory] = from_armies - 1
				new_game_state[to_territory] = to_armies

		elif to_armies == 1: # One-One
			if determine_attack < (15/36):
				new_game_state[from_territory] = from_armies
				new_game_state[to_territory] = to_armies - 1
			else:
				new_game_state[from_territory] = from_armies - 1
				new_game_state[to_territory] = to_armies
		else:
			return game_state

	elif from_armies == 1:  # No possible attack	
		return game_state

	new_game_state[enemy_territory] = -new_game_state[enemy_territory]
	new_game_state = np.reshape(new_game_state, (1, -1))

	return new_game_state



if __name__ == '__main__':
	main(sys.argv)