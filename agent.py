"""
This file contains the agent object for holding multiple q functions
"""

# import all existing policies
from q_funcs.allot import random_allot, amass
from q_funcs.attack import linear_attack_net, max_success, three_layer_attack_net, random_attack, army_difference, two_layer_attack_net, manual_attack
from q_funcs.fortify import random_fortify, skip_fortify
from q_funcs.reinforce import reinforce_all

import sys, argparse

class Agent():
    """
    This class holds three q_func objects
    One for each action type
    """
    def __init__(self, player_id, territories, act_list, allot_q_func, attack_q_func, fortify_q_func, verbose=True):
        """
        Constructor for agent
        :param territories: int number of territories on the board
        :param allot_policy: string q_func to enact for allot
        :param attack_policy: string q_func to enact for attack
        :param fortify_policy: string q_func to enact for fortify
        :return none:
        """
        self.player_id = player_id
        self.territories = territories
        self.act_list = act_list

        if allot_q_func is "random_allot":
            self.allot_q_func = random_allot.RandomAllot(self.territories, self.act_list)
        elif allot_q_func is "amass":
            self.allot_q_func = amass.Amass(self.territories, self.act_list)
        else:
            print("No valid allot Q function specified")
            exit()

        if attack_q_func is "max_success":
            self.attack_q_func = max_success.MaxSuccess(self.territories, self.act_list)
        elif attack_q_func is "random_attack":
            self.attack_q_func = random_attack.RandomAttack(self.territories, self.act_list)
        elif attack_q_func is "linear_attack_net":
            self.attack_q_func = linear_attack_net.LinearAttackNet(self.territories, self.act_list, '0', verbose=verbose)
        elif attack_q_func is "two_layer_attack_net":
            self.attack_q_func = two_layer_attack_net.TwoLayerAttackNet(self.territories, self.act_list, '0', verbose=verbose)
        elif attack_q_func is "three_layer_attack_net":
            self.attack_q_func = three_layer_attack_net.ThreeLayerAttackNet(self.territories, self.act_list, '0-77', verbose=verbose)
        elif attack_q_func is "army_difference":
            self.attack_q_func = army_difference.ArmyDifference(self.territories, self.act_list)
        elif attack_q_func is "manual":
            self.attack_q_func = manual_attack.ManualAttack(self.territories, self.act_list)
        else:
            print("No valid attack Q function specified")
            exit()

        ############ TODO: Change default reinforce q function ###################
        reinforce_q_func = "reinforce_all"
        self.reinforce_q_func = reinforce_all.ReinforceAll(self.territories, self.act_list)

        if fortify_q_func is "random_fortify":
            self.fortify_q_func = random_fortify.RandomFortify(self.territories, self.act_list)
        if fortify_q_func is "skip_fortify":
            self.fortify_q_func = skip_fortify.SkipFortify(self.territories, self.act_list)
        else:
            print("No valid attack Q function specified")
            exit()

        if verbose:
            print("Player {} successfully instantiated with Q functions:".format(player_id))
            print("\tallot: {}".format(allot_q_func))
            print("\tattack: {}".format(attack_q_func))
            print("\treinforce: {}".format(reinforce_q_func))
            print("\tfortify: {}".format(fortify_q_func))

        return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Agent Argument Parser')
    parser.add_argument('--train',dest='train',type=bool,default=False)
    parser.add_argument('--territories',dest='territories',type=int, default=2)
    parser.add_argument('--player',dest='player_id',type=int,default=0)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    isTraining = args.train
    territories = args.territories
    player_id = args.player_id

    act_list = [[0,1],[-1]]

    agent = Agent(player_id, territories, act_list, "random_allot", "linear_attack_net", "random_fortify")


if __name__ == '__main__':
    main(sys.argv)

