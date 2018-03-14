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
	parser.add_argument('--verbose',dest='verbose',type=bool, default=True)
	return parser.parse_args()

def main(args):
	"""
	Function to train the simplest type of attack network
	:param args: string command line arguments (none currently)
	:return : none
	"""

	args = parse_arguments()
	verbose = args.verbose

	# Simplest graph possible
	state_vector = np.zeros(2)

	T = len(state_vector)
	state_vector = np.reshape(state_vector, (1, -1))

	######### Hyperparameters  ########
	model_instance = '0-2'
	checkpoint_number = -1
	LEARNING_RATE = 0.0001
	GAMMA = 0.9
	# 0.2 for training, 0.1 for testing
	EPSILON = 0.2
	perform_update = True

	MAX_ARMIES = 12

	agent = linear_attack_net.LinearAttackNet(T, model_instance, checkpoint_number, LEARNING_RATE)
	opponent = max_success.MaxSuccess(T)

	print("model_instance: {}\nLEARNING_RATE: {}\nGAMMA: {}\nEPSILON: {}\nT: {}"
			   .format(model_instance, LEARNING_RATE, GAMMA, EPSILON, T))

	game_state = np.random.random_integers(1,MAX_ARMIES,size=(2))
	enemy_territory = np.random.random_integers(0,1)
	agent_territory = abs(1-enemy_territory)
	game_state[enemy_territory] = -game_state[enemy_territory]
	game_state = np.reshape(game_state,(1,-1))

	whose_turn = np.random.random_integers(0,1)
	winner = -1

	if verbose:
		print("Enemy territory is {}".format(enemy_territory))
		print("Agent territory is {}".format(agent_territory))
		if whose_turn == 1:
			print("Enemy begins")
		else:
			print("Agent begins")


	# print(game_state)

	# Initially set as a reference
	target_game_state = game_state 

	if verbose:
		print("Game state starts at: {}".format(game_state))
		print("Enemy_view starts at: {}".format(enemy_view(game_state)))

	# Set to prevent reference before assignment
	looking_ahead = False
	complete_pass_action = False
	enemy_starts = True
	agent_starts = True

	agent_wins = 0
	enemy_wins = 0


	for game in range(1000):
		while(winner == -1):

			# Opponent strategy
			while whose_turn == 1:
				# Enemy acts the same regardless of real game action or simulated for reward
				if enemy_starts:
					if verbose:
						if looking_ahead == False:
							print("Real game: enemy starts turn")
						else:
							print("Target fetch: enemy starts turn")
					if looking_ahead:
						if target_game_state[0, enemy_territory] > -MAX_ARMIES:
							target_game_state[0, enemy_territory] -= 1
						if verbose:
							print("   Enemy sees: {}\nTrue state is: {}".format(target_game_state, game_state))
					else:
						if game_state[0, enemy_territory] > -MAX_ARMIES:
							game_state[0, enemy_territory] -= 1
						if verbose:
							print("Game state is: {}".format(game_state))
					enemy_starts = False
		

				opponent_q = opponent.call_Q(enemy_view(target_game_state))
				opponent_action = np.argmax(opponent_q)
				# Attack action, valid only if enemy has more than 1 army
				if looking_ahead:
					if verbose:
						print("Target fetch, enemy action is: ")
					if (not opponent_action == 4) and target_game_state[0, enemy_territory] < -1 and (not target_game_state[0, agent_territory] == 0):  # attack action
						if verbose:
							print("\tEnemy attacks")
						target_game_state = attack(target_game_state, enemy_territory, agent_territory)
						if verbose:
							print("\tEnemy sees {}".format(target_game_state))

					else:
						if verbose:
							print("\tEnemy ends turn during fetch")
						whose_turn = 0
						agent_starts = True
				else:
					opponent_q = opponent.call_Q(enemy_view(game_state))
					opponent_action = np.argmax(opponent_q)

					if verbose:
						print("Real game: enemy action is: ")
					if (not opponent_action == 4) and game_state[0, enemy_territory] < -1 and (not game_state[0, agent_territory] == 0):
						if verbose:
							print("\tEnemy attacks")
						game_state = attack(game_state, enemy_territory, agent_territory)
						if verbose:
							print("\tNew state is {}".format(game_state))
					if game_state[0, agent_territory] == 0:  # Only true for game state, not target_game_state
						winner = 1
						break
					else:
						if verbose:
							print("\tEnemy ends real turn")
						whose_turn = 0
						agent_starts = True


			if winner == 1:
				if verbose:
					enemy_wins += 1
					print("Enemy wins")
				break

			# Player strategy
			while whose_turn == 0:
				
				if (not looking_ahead) and agent_starts:
					if verbose:
						print("Agent starting turn")
					if game_state[0, agent_territory] < MAX_ARMIES:
						game_state[0, agent_territory] += 1
					agent_action = epsilon_greedy(agent.call_Q(game_state), EPSILON)
					agent_starts = False

				# if looking_ahead:
				elif looking_ahead:
					if verbose:
						print("Target fetch: enemy has returned control")
					if target_game_state[0, agent_territory] < MAX_ARMIES:
						target_game_state[0, agent_territory] += 1
					agent_starts = False

					if target_game_state[0, enemy_territory] == 0:  # This shouldn't be possible
						print("WARNING: Enemy lost on own turn during copy game")
						exit()
					else:
						if verbose:
							print("Updating function approximator with state after opponent's turn")
						if target_game_state[0, agent_territory] > 0:
							reward = 0  # We know that the current action is pass (i.e. -1)
							target_q_func = agent.call_Q(target_game_state) # Run without update
							
							# TODO: Update indexing for improved state space
							loss_weights = np.zeros([1, 5])
							loss_weights[0][4] = 1
							target = np.zeros(5)
							target[-1] = reward + GAMMA * max(target_q_func[0][0][1], target_q_func[0][0][-1])  # max value
							target = np.reshape(target, (1, -1))
							updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=4, target=target, loss_weights=loss_weights)
						
						else:  # If agent lost in the enemy's game
							reward = -1

							# TODO: Update indexing for improved state space
							loss_weights = np.zeros([1, 5])
							loss_weights[0][4] = 1
							target = np.zeros(5)
							target[-1] = reward
							target = np.reshape(target, (1, -1))
							updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=4, target=target, loss_weights=loss_weights)


						# Go back to real game once next state has been updated
						if verbose:
							print("Returning to real game")
						looking_ahead = False
						complete_pass_action = True

				else:
					if verbose:
						print("Agent choosing next action, game state:{}".format(game_state))
					agent_action = epsilon_greedy(agent.call_Q(game_state), EPSILON)


				######### Remember - return is 3 dimensional list
				# print(action[0])
				# print(action[0][0][1])

				# TODO: Fix indexing for reasonable action space
				if agent_action == 1 and complete_pass_action == False:  # choose to attack

					if verbose:
						print("Agent chooses attack action")
					# Execute attack
					next_game_state = attack(game_state, agent_territory, enemy_territory)
					if verbose:
						print("Resulting in next state:{}".format(next_game_state))

					if next_game_state[0, enemy_territory] == 0:  # Win condition for simple env
						# terminal Q update
						reward = 1
						loss_weights = np.zeros([1, 5])
						loss_weights[0][1] = 1
						target = np.zeros(5)
						target[1] = reward
						target = np.reshape(target, (1, -1))
						updated_q_func = updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=1, target=target, loss_weights=loss_weights)

						# Set winner and break out of turn loop
						winner = 0
						break

					else:  # non-terminal q update
						reward = 0
						target_q_func = agent.call_Q(next_game_state)
						# print(target_q_func)

						# TODO: Update indexing
						loss_weights = np.zeros([1, 5])
						loss_weights[0][4] = 1
						target = np.zeros(5)
						target[1] = reward + GAMMA * max(target_q_func[0][0][1], target_q_func[0][0][-1])
						target = np.reshape(target, (1, -1))
						updated_q_func = updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=1, target=target, loss_weights=loss_weights)

					# Update the state of the game once complete, return to player turn while loop
					game_state = np.copy(next_game_state)
					agent_action = epsilon_greedy(agent.call_Q(game_state), EPSILON)

					if verbose:
						print("After agent attack, game is at: {}".format(game_state))

				elif complete_pass_action == False:  # choose to pass the turn, get target from next player's actions
					print("Agent chooses pass action, creating target fetch copy and passing turn")
					target_game_state = np.copy(game_state)  # Create a copy for simulated portion
					looking_ahead = True
					enemy_starts = True
					whose_turn = 1
				elif complete_pass_action == True:  # execute the pass_turn action
					print("Agent has completed target fetch, updating game state and passing turn")
					target_game_state = np.copy(game_state)  # Create a reference for actual game
					complete_pass_action = False
					looking_ahead = False
					enemy_starts = True
					whose_turn = 1
				else:
					print("Game has missed condition, exiting")
					exit()

			if winner == 0:
				if verbose:
					agent_wins += 1
					print("Agent wins")
				break

		# Restart the game
		looking_ahead = False
		complete_pass_action = False
		enemy_starts = True
		agent_starts = True

		game_state = np.random.random_integers(1,MAX_ARMIES,size=(2))
		enemy_territory = np.random.random_integers(0,1)
		agent_territory = abs(1-enemy_territory)
		game_state[enemy_territory] = -game_state[enemy_territory]
		game_state = np.reshape(game_state,(1,-1))

		whose_turn = np.random.random_integers(0,1)
		winner = -1

		target_game_state = game_state


	print("Training complete")
	print("Win count: Agent/Enemy: {}/{}".format(agent_wins, enemy_wins))

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


# TODO: Update so state space is reasonable
def epsilon_greedy(q_func, epsilon):
	"""
	Defines a policy which acts greedily except for epsilon exceptions
	:param q_func: q function returned by an attack network
	:param epsilon: the threshold value
	:return index: int the index of the corresponding action
	"""
	choice = np.random.uniform()

	short_q_func = np.array([q_func[0][0][4], q_func[0][0][1]])

	if choice < epsilon:
		action = np.argmax(short_q_func)
		if action == 0:
			action = 4
	else:
		action = np.argmin(short_q_func)
		if action == 0:
			action = 4
	return action


# TODO: Update so state space is correct
def enemy_view(game_state):
	"""
	Function to translate he game state as seen by the enemy
	:param game_state: the state vector of the game
	:return new_game_state: the reversed state vector
	"""
	new_game_state = np.copy(game_state)
	for state in range(len(game_state[0])):
		new_game_state[0, state] = -game_state[0,state]
	return new_game_state


if __name__ == '__main__':
	main(sys.argv)