from board import Board, Territory
from player import Player
import math
from random import shuffle, randint


class Game:
    def __init__(self, board="boards/Mini.yaml", agents=None, num_armies=100):
        self.board = Board(board)
        # change this to instantiate real agents
        self.player = [Player(), Player()]  # type: [Player]
        self.num_armies = int(math.floor(num_armies/len(agents)))

        self.agent_to_territories = {}  # type: dict(Player, [Territory])

        # gameplay flags
        self.distributed = False

    def __allot(self):
        for player in self.player:
            valid_allotments = [(territory, self.player.unallocated_armies)
                                for territory in self.agent_to_territories[player]]
            allotments = player.get_allotments(valid_allotments)
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
            territories_per_player = int(math.floor(len(territories_list) / len(self.player)))
            for player in self.player:
                for _ in range(territories_per_player):
                    territory = self.board.territories[territories_list.pop()]
                    territory.owner = player
            self.__allot()

    def __attack(self):
        """
        Executes attack round
        :return:
        """
        for player in self.player:
            # Player must have at least 2 armies in territory to attack
            #TODO: generate valid attacks
            valid_attacks = []
            attacks = player.get_attacks(valid_attacks)
            for territory_from, territory_to in attacks:  # type: Territory, Territory
                num_attacking = min(territory_from.num_armies - 1, 3)  # Leave one army behind in home
                num_defending = min(min(territory_to.num_armies, 2), num_attacking)
                attacking_dice = sorted([randint(1, 6) for _ in range(num_attacking)], reverse=True)
                defending_dice = sorted([randint(1, 6) for _ in range(num_defending)], reverse=True)

                for i in range(num_defending):
                    if attacking_dice[i] > defending_dice[i]:  # attacker is higher
                        territory_to.num_armies -= 1
                        # Check if territory is defeated
                        # If territory is defeated, switch owner,
                        if territory_to.num_armies <= 0:
                            territory_to.owner = player
                            territory_to.num_armies = num_attacking
                            territory_from.num_armies -= num_attacking
                    else:
                        territory_from.num_armies -= 1

    def __fortify(self):
        """
        Executes fortification round
        :return:
        """
        for player in self.player:
            #TODO generate valid fortifications
            valid_fortifications = []
            fortifications = player.get_fortifications(valid_fortifications)
            for territory_from, territory_to, num in fortifications:  # type: Territory, Territory, int
                territory_from -= num
                territory_to += num














