from board import Board
from agent import Agent
import math
from transitions import Machine

class Game:
    def __init__(self, board="boards/Mini.yaml", agents=None, num_armies=100):
        self.board = Board(board)
        # change this to instantiate real agents
        self.agents = [Agent(), Agent()]
        self.num_armies = int(math.floor(num_armies/len(agents)))



