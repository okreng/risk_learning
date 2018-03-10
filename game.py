from board import Board, Territory
from agent import Agent
import math
from random import shuffle
from transitions import Machine


class Game:
    def __init__(self, board="boards/Mini.yaml", agents=None, num_armies=100):
        self.board = Board(board)
        # change this to instantiate real agents
        self.agents = [Agent(), Agent()]
        self.num_armies = int(math.floor(num_armies/len(agents)))

        self.max_armies_per_territory = 30

        # gameplay flags
        self. distributed = False

    def __add_armies(self, territory, num_armies):
        """

        :param Territory territory: Territory to act on
        :param int num_armies: Armies to add

        :return int Num armies not added
        """
        num_over_limit = (territory.num_armies + num_armies) - self.max_armies_per_territory
        if num_over_limit <= 0:
            territory.num_armies += num_armies
            return 0
        else:
            territory.num_armies = self.max_armies_per_territory
            return num_over_limit

    def __remove_armies(self, territory, num_armies):
        """

        :param Territory territory:
        :param int num_armies:

        :return int Num armies not removed
        """
        num_under_zero = territory.num_armies - num_armies
        if num_under_zero > 0:
            territory.num_armies -= num_armies
            return 0
        else:
            territory.num_armies = 0
            return num_under_zero

    def distribute(self):
        """
        Distributes territories evenly and randomly amongst players
        """
        if not self.distributed:
            territories_list = self.board.territories.keys()
            shuffle(territories_list)
            territories_per_player = int(math.floor(len(territories_list) / len(self.agents)))
            for agent in self.agents:
                for _ in range(territories_per_player):
                    territory = self.board.territories[territories_list.pop()]
                    territory.owner = agent





