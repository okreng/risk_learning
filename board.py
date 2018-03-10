import networkx as nx
import yaml

class Board:
    def __init__(self, board=None):
        self.graph = nx.Graph()


        if board is not None:
            self.parse_boardfile(board)

    def parse_boardfile(self, boardfile):
        with open(boardfile) as f:
            board = yaml.load(f)
            territories = {}
            for continent_name, continent_dict in board['continents'].items():
                for country_name, country_dict in continent_dict.items():
                    territories[country_name] = Territory(country_name, continent_name, country_dict['neighbors'])

            for territory in territories.values():
                for neighbor in territory.neighbors:
                    self.graph.add_edge(territory, territories[neighbor])

    def draw(self):
        dic = dict(zip(list(self.graph.nodes), [node.name for node in self.graph.nodes]))
        layout = nx.random_layout(self.graph)
        nx.draw(self.graph, pos=layout, labels=dic, with_labels=True)





class Territory:
    def __init__(self, name, continent, neighbors=None):
        self.name = name
        self.neighbors = neighbors


