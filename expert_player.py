from player import Player
from player import AgentPlayer
import networkx as nx
from board import Territory
from abc import ABC, abstractmethod
from q_funcs.attack import max_success
from q_funcs.allot import amass
from q_funcs.fortify import random_fortify
import utils


class ExpertPlayer(AgentPlayer):

    def __init__(self, T, act_list, epsilon=0):
        self.epsilon = epsilon
        self.T = T
        self.act_list = act_list
        self.attack_q_func = max_success.MaxSuccess(T, act_list)
        self.allot_q_func = amass.Amass(T, act_list)
        self.fortify_q_func = random_fortify.RandomFortify(T, act_list)

    def get_attacks(self, valid, graph, verbose):
        state = self.convert_state_to_plus_minus(graph)
        q_table = self.attack_q_func.call_Q(state)
        choice = utils.epsilon_greedy_valid(q_table, self.epsilon)
        return choice

    def get_fortifications(self, valid, graph, verbose):
        state = self.convert_state_to_plus_minus(graph)
        q_table = self.fortify_q_func.call_Q(state)
        choice = utils.epsilon_greedy_valid(q_table, valid, self.epsilon)
        return choice

    def get_allotments(self, valid, graph, verbose):
        state = self.convert_state_to_plus_minus(graph)
        q_table = self.allot_q_func.call_Q(state)
        choice = utils.choose_by_weight(q_table)
        return choice

