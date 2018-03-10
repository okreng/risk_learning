from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def get_attacks(self, valid):
        """ Decide what to attack """
        return

    @abstractmethod
    def get_fortifications(self, valid):
        """ Decide where to fortify"""
        return

