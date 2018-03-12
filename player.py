from abc import ABC, abstractmethod
from board import Territory

class Player(ABC):

    @abstractmethod
    def get_attacks(self, valid):
        """ Decide what to attack
        :return (Territory, Territory): (territory to attack from, territory to attack)
        """
        return

    @abstractmethod
    def get_fortifications(self, valid):
        """ Decide where to fortify"""
        return

    @abstractmethod
    def get_allotments(self, valid):
        """ Should return tuple of (territory, num_to_allot)
            :return (Territory, int)
        """

        return
