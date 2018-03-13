"""
This file contains the agent object for holding multiple policies
"""

# import all existing policies
from policies.allot import random_allot
from policies.attack import linear_attack_net, max_success
from policies.fortify import random_fortify

import sys, argparse


class Agent():
    """
    This class holds three policy objects
    One for each action type
    """
    def __init__(self, territories, player_id, allot_policy, attack_policy, fortify_policy, verbose=True):
        """
        Constructor for agent
        :param territories: int number of territories on the board
        :param allot_policy: string policy to enact for allot
        :param attack_policy: string policy to enact for attack
        :param fortify_policy: string policy to enact for fortify
        :return none:
        """
        self.player_id = player_id

        if allot_policy is "random_allot":
            self.allot_policy = random_allot.RandomAllot()
        else:
            print("No valid allot policy specified")
            exit()

        if attack_policy is "max_success":
            self.attack_policy = max_success.MaxSuccess()
        elif attack_policy is "linear_attack_net":
            # TODO pass in arguments to this function
            self.attack_policy = linear_attack_net.LinearAttackNet(territories)
        else:
            print("No valid attack policy specified")
            exit()

        if fortify_policy is "random_fortify":
            self.fortify_policy = random_fortify.RandomFortify()
        else:
            print("No valid attack policy specified")
            exit()

        if verbose:
            print("Player {} successfully instantiated with policies:".format(player_id))
            print("\tallot: {}".format(allot_policy))
            print("\tattack: {}".format(attack_policy))
            print("\tfortify: {}".format(fortify_policy))

        return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Agent Argument Parser')
    parser.add_argument('--train',dest='train',type=bool,default=False)
    parser.add_argument('--territories',dest='territories',type=int)
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

