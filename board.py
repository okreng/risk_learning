import networkx as nx
import yaml


class Board:
    def __init__(self, board=None):
        self.graph = nx.Graph()
        self.territories = {}
        if board is not None:
            self.parse_boardfile(board)

    def parse_boardfile(self, boardfile):
        with open(boardfile) as f:
            board = yaml.load(f)
            for continent_name, continent_dict in board['continents'].items():
                for country_name, country_dict in continent_dict.items():
                    self.territories[country_name] = Territory(country_name, continent_name, country_dict['neighbors'])

            for territory in self.territories.values():
                for neighbor in territory.neighbors:
                    self.graph.add_edge(territory, self.territories[neighbor])

    def draw(self):
        inv_dict = {v: k for k, v in self.territories.items()}
        layout = nx.kamada_kawai_layout(self.graph)
        nx.draw(self.graph, pos=layout, labels=inv_dict, with_labels=True)


class Territory:
    def __init__(self, name, continent, neighbors=None):
        self.name = name
        self.neighbors = neighbors
        self.continent = continent
        self.num_armies = 0
        self.owner = None
        self.max_armies_per_territory = 30

    def add_armies(self, num_armies):
        """

        :param Territory territory: Territory to act on
        :param int num_armies: Armies to add

        :return int Num armies not added
        """
        num_over_limit = (self.num_armies + num_armies) - self.max_armies_per_territory
        if num_over_limit <= 0:
            self.num_armies += num_armies
            return 0
        else:
            self.num_armies = self.max_armies_per_territory
            return num_over_limit

    def remove_armies(self, num_armies):
        """

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

