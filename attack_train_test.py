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
from q_funcs.attack import random_attack
from q_funcs.attack import army_difference
from q_funcs.attack import three_layer_attack_net
from q_funcs.attack import leaky_relu_3_layer


def parse_arguments():
	parser = argparse.ArgumentParser(description='Agent Argument Parser')
	parser.add_argument('--train',dest='train',type=int)
	parser.add_argument('--verbose',dest='verbose_int',type=int, default=1)
	return parser.parse_args()

def main(args):
	"""
	Function to train the simplest type of attack network
	:param args: string command line arguments
	:param train: string 'train' or 'test'
	:param verbose: boolean default True
	:return : none
	"""

	args = parse_arguments()
	train = args.train
	verbose_int = args.verbose_int

	if verbose_int == 1:
		verbose = True
	else:
		verbose = False

	# Simplest graph possible
	T = 2
	state_vector = np.zeros(T)
	act_list = [[0,1],[-1]]

	state_vector = np.reshape(state_vector, (1, -1))

	######### Hyperparameters  ########
	if train == 1:
		if verbose:
			print("Beginning to train")
		model_instance = '0-2'
		checkpoint_number = -1
		LEARNING_RATE = 0.00001
		GAMMA = 0.95
		epsilon = 0.85
		perform_update = True
		NUM_GAMES = 200000
	elif train == 0:
		if verbose:
			print("Beginning to test")
		model_instance = '0-2'
		checkpoint_number = -1
		LEARNING_RATE = 0  # never used
		GAMMA = 0.9  # never used
		epsilon = 0.1 # Lower for testing, does not go lower than ENEMY_EPSILON
		perform_update = False
		NUM_GAMES = 1000
	else:
		print("Specify --train as 1 for training, 0 for testing")
		exit()


	MAX_ARMIES = 4
	ENEMY_EPSILON = 0.1  # Does not change for train/test

	# agent = max_success.MaxSuccess(T, act_list)
	# agent = army_difference.ArmyDifference(T, act_list)
	# agent = linear_attack_net.LinearAttackNet(T, act_list, model_instance, checkpoint_number, LEARNING_RATE)
	# agent = three_layer_attack_net.ThreeLayerAttackNet(T, act_list, model_instance, checkpoint_number, LEARNING_RATE)
	agent = leaky_relu_3_layer.LeakyRelu3Layer(T, act_list, model_instance, checkpoint_number, LEARNING_RATE)

	############ Opponent defined again at bottom randomly ################3
	# opponent = max_success.MaxSuccess(T, act_list)
	opponent = random_attack.RandomAttack(T, act_list)
	# opponent = army_difference.ArmyDifference(T, act_list)

	print("model_instance: {}\nLEARNING_RATE: {}\nGAMMA: {}\nepsilon: {}\nT: {}"
			   .format(model_instance, LEARNING_RATE, GAMMA, epsilon, T))

	# starting_armies = np.random.random_integers(1,MAX_ARMIES)
	starting_armies = MAX_ARMIES
	# game_state = np.random.random_integers(1,MAX_ARMIES,size=(2))
	game_state = np.array([starting_armies, starting_armies])
	enemy_territory = np.random.random_integers(0,1)
	# enemy_territory = 1
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
	enemy_starts = False
	agent_starts = False

	agent_wins = 0
	enemy_wins = 0


	for game in range(NUM_GAMES):
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
		
				if looking_ahead:
					opponent_q = opponent.call_Q(enemy_view(target_game_state))
					if target_game_state[0, enemy_territory] == -1:
						opponent_valid_mask = [0, 1]
					else:
						opponent_valid_mask = [1, 1]
					# print(np.multiply(opponent_valid_mask, opponent_q))
					opponent_action = epsilon_greedy_valid(opponent_q, opponent_valid_mask, ENEMY_EPSILON)
					# print(opponent_action)
					# print("Opponent chooses action: {}".format( opponent_action))

					# Attack action, valid only if enemy has more than 1 army
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
					if game_state[0, enemy_territory] == -1:
						opponent_valid_mask = [0, 1]
					else:
						opponent_valid_mask = [1, 1]
					# print(np.multiply(opponent_valid_mask, opponent_q))
					opponent_action = epsilon_greedy_valid(opponent_q, opponent_valid_mask, ENEMY_EPSILON)
					# print(opponent_action)
					# print("Opponent chooses action: {}".format( opponent_action))

					if verbose:
						print("Real game: enemy action is: ")
					if (not opponent_action == 1) and game_state[0, enemy_territory] < -1 and (not game_state[0, agent_territory] == 0):
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
					
			################# TODO: Determine standard shape for call_Q return #####

					agent_q = agent.call_Q(game_state)
					# agent_q = agent_big_q[0][0]
					if game_state[0, agent_territory] == 1:
						agent_valid_mask = [0, 1]
					else:
						agent_valid_mask = [1, 1]
					# print(np.multiply(agent_valid_mask, agent_q))
					agent_action = epsilon_greedy_valid(agent_q, agent_valid_mask, epsilon)
					# print(agent_action)
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
							
							loss_weights = np.zeros([1, len(act_list)])
							loss_weights[0][-1] = 1
							target = np.zeros(len(act_list))
							target[-1] = reward + GAMMA * max(target_q_func)  # max value
							target = np.reshape(target, (1, -1))
							updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=agent_action, target=target, loss_weights=loss_weights)
						
						else:  # If agent lost in the enemy's game
							reward = -1
							loss_weights = np.zeros([1, len(act_list)])
							loss_weights[0][-1] = 1
							target = np.zeros(len(act_list))
							target[-1] = reward
							target = np.reshape(target, (1, -1))
							updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=agent_action, target=target, loss_weights=loss_weights)


						# Go back to real game once next state has been updated
						if verbose:
							print("Returning to real game")
						looking_ahead = False
						complete_pass_action = True

				else:
					if verbose:
						print("Agent choosing next action, game state:{}".format(game_state))

				############## TODO: Determine standard return shape for call_Q ###3

					agent_q = agent.call_Q(game_state)
					# agent_q = agent_big_q[0][0]
					if game_state[0, agent_territory] == 1:
						agent_valid_mask = [0, 1]
					else:
						agent_valid_mask = [1, 1]
					# print(np.multiply(agent_valid_mask, agent_q))
					agent_action = epsilon_greedy_valid(agent_q, agent_valid_mask, epsilon)
					# print(agent_action)
				######### Remember - return is 3 dimensional list
				# print(action[0])
				# print(action[0][0][1])

				if (not agent_action == 1) and complete_pass_action == False:  # choose to attack

					if verbose:
						print("Agent chooses attack action")
					# Execute attack
					next_game_state = attack(game_state, agent_territory, enemy_territory)
					if verbose:
						print("Resulting in next state:{}".format(next_game_state))

					if next_game_state[0, enemy_territory] == 0:  # Win condition for simple env
						# terminal Q update
						reward = 1
						loss_weights = np.zeros([1, len(act_list)])
						loss_weights[0][agent_action] = 1
						target = np.zeros(len(act_list))
						target[agent_action] = reward
						target = np.reshape(target, (1, -1))
						updated_q_func = updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=agent_action, target=target, loss_weights=loss_weights)

						# Set winner and break out of turn loop
						winner = 0
						break

					else:  # non-terminal q update
						reward = 0
						target_q_func = agent.call_Q(next_game_state)
						loss_weights = np.zeros([1, len(act_list)])
						loss_weights[0][-1] = 1
						target = np.zeros(len(act_list))
						target[-1] = reward + GAMMA * max(target_q_func)
						target = np.reshape(target, (1, -1))
						updated_q_func = updated_q_func = agent.call_Q(state_vector=game_state, update=perform_update, action_taken=agent_action, target=target, loss_weights=loss_weights)

					# Update the state of the game once complete, return to player turn while loop
					game_state = np.copy(next_game_state)

			################## TODO: Determine standard shape for call_Q ########

					# agent_big_q = agent.call_Q(game_state)
					# agent_q = agent_big_q[0][0]
					# if game_state[0, agent_territory] == 1:
					# 	agent_valid_mask = [0, 1]
					# else:
					# 	agent_valid_mask = [1, 1]
					# # print(np.multiply(agent_valid_mask, agent_q))
					# agent_action = epsilon_greedy(np.multiply(agent_valid_mask, agent_q), EPSILON)
					# # print(agent_action)
					if verbose:
						print("After agent attack, game is at: {}".format(game_state))

				elif complete_pass_action == False:  # choose to pass the turn, get target from next player's actions
					if verbose:
						print("Agent chooses pass action, creating target fetch copy and passing turn")
					target_game_state = np.copy(game_state)  # Create a copy for simulated portion
					looking_ahead = True
					enemy_starts = True
					whose_turn = 1
				elif complete_pass_action == True:  # execute the pass_turn action
					if verbose:
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

		# Update epsilon
		if game == (NUM_GAMES % 1000) and epsilon >= ENEMY_EPSILON and train:
			epsilon -= 0.005

		# Choose next opponent randomly
		next_opponent = np.random.random_integers(0,3)
		if next_opponent == 0:
			opponent = max_success.MaxSuccess(T, act_list)
		elif next_opponent == 1:
			opponent = random_attack.RandomAttack(T, act_list)
		elif next_opponent == 2:
			opponent = army_difference.ArmyDifference(T, act_list)

		game_state = np.random.random_integers(1,MAX_ARMIES,size=(2))
		enemy_territory = np.random.random_integers(0,1)
		# enemy_territory = 1
		agent_territory = abs(1-enemy_territory)
		game_state[enemy_territory] = -game_state[enemy_territory]
		game_state = np.reshape(game_state,(1,-1))

		whose_turn = np.random.random_integers(0,1)
		winner = -1

		target_game_state = game_state


	if train:
		print("Training complete")
		print("Win count: Agent/Enemy: {}/{}".format(agent_wins, enemy_wins))
		agent.close()
	else:
		print("Testing complete")
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


def epsilon_greedy(q_func, epsilon):
	"""
	Defines a policy which acts greedily except for epsilon exceptions
	:param q_func: q function returned by an attack network
	:param epsilon: the threshold value
	:return index: int the index of the corresponding action
	"""

	eps_choices = len(q_func) - 1
	if eps_choices == 0:
		return -1

	choice = np.random.uniform()
	max_action = np.argmax(q_func)

	# print(choice)
	# print("Max action is {}".format(max_action))

	if choice > epsilon:
		return max_action
	else:
		eps_slice = epsilon/eps_choices
		for act_slice in range(eps_choices):
			# print(eps_slice*(1+act_slice))
			if choice < (eps_slice*(1+act_slice)):
				action = act_slice
				break


	if action >= max_action:  # Increment if past max_action
		action += 1

	return action


def epsilon_greedy_valid(q_func, valid_mask, epsilon):
	"""
	Returns an epsilon greedy action from a subset of function defined by mask
	Only chooses valid actions as specified by the mask
	:param q_func: float vector to return argmax in greedy case
	:param valid_mask: int vector of valid actions
	:param epsilon: probability under which to choose non-greedily
	:return arg: int choice
	"""
	nA = len(valid_mask)
	if not (len(q_func) == nA):
		print("Q function and mask different sizes")
		return -1
	eps_choices = np.sum(valid_mask) - 1

	valid_q_func = []
	valid_q_to_orig_q_map = []
	for ii in range(nA):
		if valid_mask[ii] == 1:
			valid_q_func.append(q_func[ii])
			valid_q_to_orig_q_map.append(ii)

	if len(valid_q_func) == 0:
		print("No valid actions")
		return -1

	# print(valid_q_func)
	# print(valid_q_to_orig_q_map)
	valid_action = epsilon_greedy(valid_q_func, epsilon)
	# print(valid_action)
	action = valid_q_to_orig_q_map[valid_action]

	return action


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