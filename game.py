from board import Board, Territory
from agent import Agent
import math
from random import shuffle
from transitions import Machine


class Game:
    def __init__(self, board="boards/Mini.yaml", agents=None, num_armies=100):
        self.board = Board(board)
        # change this to instantiate real agents
        self.agents = [Agent(), Agent()]  # type: [Agent]
        self.num_armies = int(math.floor(num_armies/len(agents)))

        self.agent_to_territories = {}  # type: dict(Agent, [Territory])
        self.max_armies_per_territory = 30

        # gameplay flags
        self. distributed = False

    def __allot(self):
        for agent in self.agents:
            valid_allotments = [(territory, self.max_armies_per_territory - territory.num_armies)
                                for territory in self.agent_to_territories[agent]]
            allotments = agent.get_allotments(valid_allotments)
            for territory, num_armies in allotments:
                territory.add_armies(num_armies)

    def __distribute(self):
        """
        Distributes territories evenly and randomly amongst players
        Asks players to allot armies for initial allotment and allots
        """
        if not self.distributed:
            territories_list = self.board.territories.keys()
            shuffle(territories_list)
            territories_per_player = int(math.floor(len(territories_list) / len(self.agents)))
            for agent in self.agents:
                for _ in range(territories_per_player):
                    territory = self.board.territories[territories_list.pop()]
                    territory.owner = agent
            self.__allot()







