"""
This file contains the agent object for holding multiple q functions
"""

# import all existing policies
from q_funcs.allot import random_allot
from q_funcs.attack import linear_attack_net, max_success
from q_funcs.fortify import random_fortify

import sys, argparse

class Agent():
	"""
	This class holds three q_func objects
	One for each action type
	"""
	def __init__(self, territories, player_id, allot_q_func, attack_q_func, fortify_q_func, verbose=True):
		"""
		Constructor for agent
		:param territories: int number of territories on the board
		:param allot_policy: string q_func to enact for allot
		:param attack_policy: string q_func to enact for attack
		:param fortify_policy: string q_func to enact for fortify
		:return none:
		"""
		self.player_id = player_id

		if allot_q_func is "random_allot":
			self.allot_q_func = random_allot.RandomAllot(territories)
		else:
			print("No valid allot Q function specified")
			exit()
			
		if attack_q_func is "max_success":
			self.attack_q_func = max_success.MaxSuccess(territories)
		elif attack_q_func is "linear_attack_net":
			# TODO pass in arguments to this function
			self.attack_q_func = linear_attack_net.LinearAttackNet(territories)
		else:
			print("No valid attack Q function specified")
			exit()

		if fortify_q_func is "random_fortify":
			self.fortify_q_func = random_fortify.RandomFortify(territories)
		else:
			print("No valid attack Q function specified")
			exit()

		if verbose:
			print("Player {} successfully instantiated with Q functions:".format(player_id))
			print("\tallot: {}".format(allot_q_func))
			print("\tattack: {}".format(attack_q_func))
			print("\tfortify: {}".format(fortify_q_func))

		return


def parse_arguments():
	parser = argparse.ArgumentParser(description='Agent Argument Parser')
	parser.add_argument('--train',dest='train',type=bool,default=False)
	parser.add_argument('--territories',dest='territories',type=int, default=1)
	parser.add_argument('--player',dest='player_id',type=int,default=0)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	isTraining = args.train
	territories = args.territories
	player_id = args.player_id

	agent = Agent(territories, player_id, "random_allot", "linear_attack_net", "random_fortify")


if __name__ == '__main__':
	main(sys.argv)

