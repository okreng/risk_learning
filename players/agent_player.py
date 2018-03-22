from player import Player
import networkx as nx
from board import Territory


class AgentPlayer(Player):
    def __init__(self):
        super().__init__()

    def get_attacks(self, valid, graph):
        pass

    def get_fortifications(self, valid, graph):
        pass

    def get_allotments(self, valid, graph):
        pass

    def convert_state_to_plus_minus(self, graph):
        """
        Converts graph to list of armies on territory.
        Positive if territory is owned, negative if not
        :param nx.Graph graph:
        :return:
        """
        # Sort to maintain order
        territories = sorted(graph.nodes())
        state = []
        for territory in territories:  # type: Territory
            state.append(territory.num_armies if territory.owner is self else -1 * territory.num_armies)


