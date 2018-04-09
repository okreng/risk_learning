from board import Board, Territory
from players.manual_player import ManualPlayer
import math
from random import shuffle, randint
from enum import Enum  #, auto NOT SUPPORTED FOR PYTHON 3.5
import numpy as np

ARMIES_PER_TERRITORY = 0.3333333
MIN_ALLOT_ARMIES = 3

class GameStates(Enum):
    ALLOT = 0
    ATTACK = 1
    FORTIFY = 2
    # ALLOT = auto()
    # ATTACK = auto()
    # FORTIFY = auto()


class Game:
    def __init__(self, board="boards/Mini.yaml", num_armies=100, players=None):
        self.board = Board(board)
        # change this to instantiate real agents
        if players is None:
            self.players = [ManualPlayer(), ManualPlayer()]  # type: [Player]
        else:
            self.players = players
        # Set colors for graph drawing
        player_colors = ['r', 'g', 'c', 'm', 'y', 'b', 'w']
        for i, player in enumerate(self.players):
            player.color = player_colors[i]
            player.player_num = i + 1

        self.num_armies = int(math.floor(num_armies/len(self.players)))

        self.agent_to_territories = {}  # type: dict(Player, [Territory])

        # gameplay flags
        self.distributed = False

    def __distribute(self):
        """
        Distributes territories evenly and randomly amongst players
        """
        if not self.distributed:
            territories_list = list(self.board.territories.keys())
            shuffle(territories_list)
            territories_per_player = int(math.floor(len(territories_list) / len(self.players)))
            for player in self.players:
                for _ in range(territories_per_player):
                    territory = self.board.territories[territories_list.pop()]
                    territory.owner = player

    def __allot(self, player):
        """
        Gets allotment from player and allots as requested
        :param Player player:
        :return:
        """
        # valid_allotments = [(territory, player.unallocated_armies) 
        #                     for territory in self.board.get_player_territories(player)]
        player_terrs = [territory.u_id for territory in self.board.get_player_territories(player)]
        # allotments = player.get_allotments(valid_allotments, self.board.graph)
        valid_allotments = np.zeros(self.board.num_territories)
        valid_allotments[player_terrs] = 1
        allotment = player.get_allotment(valid_allotments, self.board)
        territory = self.board.territories_by_id[allotment]
        territory.add_armies(1)
        # for territory, num_armies in allotments:
        # territory.add_armies(num_armies)

    def __attack(self, player):
        """
        Gets attacks from player and modifies board based on attacks
        :param Player player:
        :return:
        """
        # Player must have at least 2 armies in territory to attack
        owned_territories = self.board.get_player_territories(player)
        valid_attacks = [(territory, neighbor)
                         for territory in owned_territories
                         for neighbor in territory.neighbors
                         if territory.num_armies >= 2]
        valid_attacks.append((None, None))
        attacks = player.get_attacks(valid_attacks, self.board.graph)

        for territory_from, territory_to in attacks:  # type: Territory, Territory
            if territory_from is None or territory_to is None:
                pass  # Literally pass
            else:
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

    def __fortify(self, player):
        """
        Gets fortifications from player and modifies board based on fortification
        :param Player player:
        :return:
        """
        owned_territories = self.board.get_player_territories(player)
        valid_fortifications = [(t_from, t_to, t_from.num_armies) for t_from in owned_territories
                                for t_to in owned_territories if t_from is not t_to]
        fortifications = player.get_fortifications(valid_fortifications, self.board.graph)
        for territory_from, territory_to, num in fortifications:  # type: Territory, Territory, int
            territory_from.num_armies -= num
            territory_to.num_armies += num

    def __check_end(self):
        """
        Checks if game has ended
        :return:  if there is more than one player who is alive
        """
        return len([player for player in self.players if player.alive]) < 1

    def get_state(self):
        """
        Gets current state of game
        :return:
        """
        state = self.board.graph
        return state

    def turn_start_armies(self, player):
        """
        Returns the number of armies a player has to allot
        args:
        player : the player whose turn is starting
        # TODO: add armies based on continent
        """
        territories = self.board.get_player_territories(player)
        return max(MIN_ALLOT_ARMIES, int(np.floor(len(territories)*ARMIES_PER_TERRITORY)))

    def play_game(self):
        """
        Plays through game without pause or ability to manually step
        :return:
        """
        while not self.__check_end():
            self.__distribute()
            for player in self.players:
                player.unallocated_armies = self.turn_start_armies(player)
                while player.unallocated_armies > 0:
                    self.__allot(player)
                    player.unallocated_armies -= 1
                self.__attack(player)
                self.__fortify(player)
                self.board.draw()














