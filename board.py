import networkx as nx
import yaml
from player import Player
import matplotlib.pyplot as plt
import matplotlib.colors


class Board:
    """
    Class that represents a board. Contains all available territories.
    """
    def __init__(self, board="boards/Mini.yaml"):
        self.graph = nx.Graph()  # Graph that holds territories and edges between them
        self.territories = {}  # Maps territory name to Territory
        self.territories_by_id = {}  # Maps u_id to Territory
        self.num_territories = 0

        if board is not None:
            self.parse_boardfile(board)

        self.continents = list(set([territory.continent for territory in self.territories.values()]))
        # print(self.continents)

    def parse_boardfile(self, boardfile):
        """
        Loads in a territory YAML and saves it to local vars.
        :param str boardfile: path to YAML file
        """
        self.graph = nx.Graph()
        self.territories = {}
        u_id = 0
        with open(boardfile) as f:
            board = yaml.load(f)
            for continent_name, continent_dict in board['continents'].items():
                for country_name, country_dict in continent_dict.items():
                    self.territories[country_name] = Territory(country_name, continent_name, u_id, country_dict['neighbors'])
                    self.territories_by_id[u_id] = self.territories[country_name]
                    u_id += 1
                    self.num_territories += 1

            for territory in self.territories.values():
                for neighbor in territory.neighbors:
                    self.graph.add_edge(territory, self.territories[neighbor])

    def get_player_territories(self, player):
        """
        Finds list of territories belonging to player
        :param Player player:
        :return [Player]: territories owned by player
        """
        return [t for t in self.territories.values() if t.owner is player]

    def draw(self, color="country"):
        """
        Draws current graph
        """

        colors = ['r', 'g', 'c', 'm', 'y', 'b', 'w']
        label_dict = {v: k + ':\n' + str(v.num_armies) for k, v in self.territories.items()}
        layout = nx.kamada_kawai_layout(self.graph, scale=50)
        node_color = []
        if color is "players":
            node_color = [node.owner.color if node.owner is not None else 'r'
                          for node in self.graph.nodes()]
        elif color is "country":
            continent_color_dict = {continent: colors[i] for i, continent in enumerate(self.continents)}
            node_color = [continent_color_dict[territory.continent] for territory in self.graph.nodes()]
        # Lower alpha channels for colors
        node_color = [matplotlib.colors.to_hex(matplotlib.colors.to_rgba(c, alpha=0.1), keep_alpha=True) for c in
                      node_color]
        print(node_color)

        nx.draw_networkx_nodes(self.graph, pos=layout, node_color=node_color, alpha=0.5)
        nx.draw_networkx_edges(self.graph, pos=layout)
        nx.draw_networkx_labels(self.graph, pos=layout, labels=label_dict)

        # plt.ion()
        plt.show()


class Territory:
    """
    Class representing a single territory.
    """
    def __init__(self, name, continent, u_id, neighbors=None):
        self.name = name
        self.neighbors = neighbors  # type: [Territory]
        self.continent = continent  # type: str
        self.num_armies = 0         # number of armies contained
        self.owner = None           # type: Player
        self.u_id = u_id            # type: int

    def add_armies(self, num_armies):
        """
        Adds armies to territories. Adding conditions should be checked here
        :param int num_armies: Armies to add

        :return int Num armies not added
        """
        self.num_armies = num_armies
        return 0

    def remove_armies(self, num_armies):
        """
        Removes armies from territories. Removal conditions should be checked here.
        :param Territory territory:
        :param int num_armies:

        :return int Num armies not removed
        """
        num_under_zero = self.num_armies - num_armies
        if num_under_zero > 0:
            self.num_armies -= num_armies
            return 0
        else:
            self.num_armies = 0
            return num_under_zero

