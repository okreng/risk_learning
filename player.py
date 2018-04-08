from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self):
        """
        Initializes variables all subclasses of player should have
        """
        self.player_num = -1
        self.unallocated_armies = 0
        self.alive = True
        self.color = 'r'

    @abstractmethod
    def get_attacks(self, valid, graph):
        """ Decide what to attack
            :param valid: Valid choices
            :param graph: board
            :return (Territory, Territory): (territory to attack from, territory to attack)
        """
        return

    @abstractmethod
    def get_fortifications(self, valid, graph):
        """ Decide where to fortify
            :param valid: Valid choices
            :param graph: board
            :return (Territory, Territory, int): (territory_from, territory_to, num to add)
        """
        return

    @abstractmethod
    def get_allotments(self, valid, graph):
        """ Should return tuple of (territory, num_to_allot)
            :param valid: Valid choices
            :param graph: board
            :return (Territory, int)
        """
        return
