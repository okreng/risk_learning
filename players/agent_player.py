from player import Player
import networkx as nx
from board import Territory
from abc import ABC, abstractmethod


class AgentPlayer(Player, ABC):

    @abstractmethod
    def get_attacks(self, valid, graph):
        return

    @abstractmethod
    def get_fortifications(self, valid, graph):
        return

    @abstractmethod
    def get_allotments(self, valid, graph):
        return

    def convert_state_to_plus_minus(self, graph):
        """
        Converts graph to list of armies on territory.
        Positive if territory is owned, negative if not
        :param nx.Graph graph:
        :return [int]: list of num_armies
        """
        # Sort to maintain order
        territories = sorted(graph.nodes())
        state = []
        for territory in territories:  # type: Territory
            state.append(territory.num_armies if territory.owner is self else -1 * territory.num_armies)
        return state


