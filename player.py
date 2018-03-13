from abc import ABC, abstractmethod
from board import Territory


class Player(ABC):
    def __init__(self):
        """
        Initializes variables all subclasses of player should have
        """
        self.unallocated_armies = 0

    @abstractmethod
    def get_attacks(self, valid):
        """ Decide what to attack
        :return (Territory, Territory): (territory to attack from, territory to attack)
        """
        return

    @abstractmethod
    def get_fortifications(self, valid):
        """ Decide where to fortify
            :return (Territory, Territory, int): (territory_from, territory_to, num to add)
        """
        return

    @abstractmethod
    def get_allotments(self, valid):
        """ Should return tuple of (territory, num_to_allot)
            :return (Territory, int)
        """

        return
