"""
This file provides the environment for a player to interact with the Risk game board
# TODO: smart importing
"""

import numpy as np
# from OLD import risk_graph as rg
import risk_graph as rg
from enum import Enum

ActionType = Enum('ActionType','ALLOT ATTACK REINFORCE FORTIFY GAMEOVER')
MIN_ARMIES_PER_TURN = 1
INITIAL_PLACEMENT_ARMIES = 2

class RiskGame():
    """
    This class provides the functions for a player to interact with a Risk board game
    """
    def __init__(self, board, num_players=2, verbose=False):
        """
        The constructor for the risk environment
        :param board: the string of the .risk file to be loaded
        """
        self.player_turn = -1
        self.unallocated_armies = -1
        self.reinforce_edge = None
        self.action_type = ActionType.GAMEOVER
        self.num_players = num_players
        self.graph = rg.RiskGraph(board, verbose)

        # Create board_state as reference to other items in the board
        self.player_list = []

        self.winner = -1
        for player_id in range(self.num_players):
            self.add_player(player_id)
            # self.activate_player(player_id) # Maybe do this here
            if verbose:
                print("Created inactive player: {}".format(player_id))

        global MAX_ARMIES_PER_PLAYER
        MAX_ARMIES_PER_PLAYER = 42

        return

    def random_start(self, verbose=False):
        """
        This function shuffles the territories and distributes them amongst the players
        :param verbose: bool whether to talk about who has what initialization
        :return none:
        """

        self.active_players = 0
        for player in self.player_list:
            self.activate_player(player.get_id())
            self.active_players += 1

        self.player_placement_order = np.random.permutation(len(self.player_list))
        self.terr_placement_order = np.random.permutation(self.graph.total_territories)
        for terr_id in self.terr_placement_order:
            player_choice = self.player_placement_order[terr_id%self.num_players]
            self.assign_territory(terr_id, player_choice)
            if verbose:
                print("Territory {} assigned to player {}".format(terr_id, player_choice))

            # TOOD - determine how many armies to put on each territory

        self.player_turn_order = np.random.permutation(len(self.player_list)).tolist()

        terr_armies = 0
        player_armies = 0
        for terr in self.graph.territory_list:
            terr_armies += terr.get_armies()
            if terr.player_id == -1:
                print("Not all territories assigned")
                exit()
        for player in self.player_list:
            player_armies += player.get_total_armies()
            if player.isActive == False:
                print("Not all players active")
                exit()
            if player.total_armies < 1:
                print("Player {} has no armies".format(player.player_id))
                exit()
            if player.total_territories < 1:
                print("Player {} has no territories".format(player.player_id))
                exit()

        # As per official rules:
        self.player_turn = self.player_placement_order[0]

        self.placement_phase = True
        self.reinforce_from = None
        self.calculate_initial_allotment()
        self.action_type = ActionType.ALLOT
        self.winner = -1



        if verbose:
            print("Player {} starts".format(self.player_turn))

        return self.game_state(), True

    def game_state(self):
        """
        arguments:
        none
        returns:
        int player id: whose turn it is
        enum ActionType: what action they must take
        bool active: Whether the game is still being played
        """
        return self.board_state(), self.player_turn, self.action_type, self.unallocated_armies, self.reinforce_edge, self.winner

    def board_state(self):
        """
        arguments:
        none
        returns:
        the full state of the board as a tuple of playerid, armies
        """
        self.graph_state = [(terr.player_id, terr.armies) for terr in self.graph.territory_list]
        return self.graph_state


    def assign_territory(self, terr_id, player_id, losing_player_id=-1):
        """
        Assigns a territory in the game to a new player
        NOTE - This changes the number of armies, but only happens at beginning of game
        :param terr_id: int the unique id of the territory
        :param player_id: int the unique id of the player
        :return : none
        """

        terr = self.graph.get_terr_by_id(terr_id)
        player = self.get_player_from_id(player_id)

        if not (losing_player_id == -1):
            losing_player = self.get_player_from_id(losing_player_id)
            losing_player.subtract_territory()
            losing_player.remove_territory(terr_id)
        else:
            player.add_armies(1)  # Only add when initially assigning
            terr.set_armies(1)

        # Player effects
        player.add_territory()
        player.territory_list.append(terr_id)

        # Territory effects
        # if terr.player_id == player_id:
        #     print("Assigning territory to current occupant")
        terr.player_id = player_id
        return

    def set_player_id_to_terr(self, player_id, terr_id):
        """
        This function sets a graph to a given player ID
        :params player_id: the unique ID of a player
        :params terr_id: the unique ID of a territory
        """
        self.graph.get_terr_by_terr_id(self.graph, terr_id).set_player_id(player_id)
        return

    def add_player(self, player_id):
        """
        This function adds a player to the game
        :param player_id: the unique ID of the player
        :return : No return value
        """
        new_player = Player(player_id)
        self.player_list.append(new_player)
        return

    def activate_player(self, player_id, deactivate=False):
        """
        This function sets a player's isActive tag to True
        :param player_id: int the unique ID of the player
        """
        player = self.get_player_from_id(player_id)
        if deactivate:
            player.isActive = False
        else:
            player.isActive = True
        player.total_armies = 0
        player.total_territories = 0
        return

    def remove_armies(self, terr_id, player_id, num_armies):
        """
        remove armies from the territory and the player
        :return: lost if the territory was lost
        """
        lost = False # default
        self.get_player_from_id(player_id).lose_armies(num_armies)
        lost = self.graph.get_terr_by_id(terr_id).remove_armies(num_armies)

        return lost


    def check_loss(self):
        """
        This function checks if any players have lost
        If all but one player has lost, the game is set to inactive
        In this case, the winner is returned
        :param: none
        :return: -1 if no winner
        :return: player_id if a player has won
        """
        for player in self.player_list:
            if player.isActive:
                if (player.total_armies == 0):
                    player.lose()
                    self.player_turn_order.remove(player.player_id)
                    self.active_players -= 1
                else:
                    winning = player.player_id

        if self.active_players == 1:
            self.winner = winning
            return self.winner

        return -1


    def get_player_from_id(self, player_id):
        """
        This function returns a player object when its unique ID is called
        :param player_id: the unique ID of the player
        :return player: Player the player specified by the unique ID
        """
        if player_id >= self.num_players:
            print("Attempted to fetch nonexistent player")
            exit()
        return self.player_list[player_id]

    def calculate_initial_allotment(self):
        """
        Calculates the number of armies to provide at the start of the game
        These are share among all players
        """
        # TODO: base this on game logic
        self.unallocated_armies = INITIAL_PLACEMENT_ARMIES
        return


    def calculate_allotment(self):
        """
        Calculates the number of armies the active player can place
        """
        # TODO: Add based on actual game logic
        self.unallocated_armies = max(MIN_ARMIES_PER_TURN, np.floor(0.33333 * self.get_player_from_id(self.player_turn).total_territories))

    def allot(self, terr_id):
        """
        Function to add an army to a player's territory
        returns:
        game_state
        """
        if (self.graph.get_player_id_by_terr_id(terr_id) != self.player_turn):
            print("ALLOT ERROR: Player cannot allot to another player's territory")
            return self.game_state(), False
        else:
            if self.unallocated_armies >= 1:
                self.graph.get_terr_by_id(terr_id).add_armies(1)
                self.get_player_from_id(self.player_turn).add_armies(1)
                self.unallocated_armies -= 1
                if (self.placement_phase):
                    self.advance_turn()
            else:
                print("ALLOT ERROR: Player has no armies to allot")
                return self.game_state(), False

            if self.unallocated_armies == 0:
                self.unallocated_armies = -1  # So that it cannot be misinterpreted
                if (self.placement_phase):
                    self.placement_phase = False
                self.action_type = ActionType.ATTACK
        return self.game_state(), True


    def attack(self, edge_id):
        """
        Function to determine the results of an attack
        :param edge_id: edge upon which to attack
        :return board_state(): the state of the board after the attack
        """

        if edge_id == len(self.graph.edge_list)-1:
            self.action_type = ActionType.FORTIFY
            return self.game_state(), True

        attacker = self.player_turn
        nodes = self.graph.edge_list[edge_id]
        player_1 = self.graph.get_player_id_by_terr_id(nodes[0])
        player_2 = self.graph.get_player_id_by_terr_id(nodes[1])

        if (player_1 == player_2):
            print("ATTACK ERROR: Same player owns both edge territories")
            return self.game_state(), False

        if attacker == player_1:
            defender = self.graph.get_player_id_by_terr_id(nodes[1])
            from_id = nodes[0]
            to_id = nodes[1]
        elif attacker == player_2:
            defender = self.graph.get_player_id_by_terr_id(nodes[0])
            from_id = nodes[1]
            to_id = nodes[0]
        else:
            print("ATTACK ERROR: Attacking player does not own either edge territory")
            return self.game_state(), False


        from_armies = self.graph.get_armies_by_terr_id(from_id)
        to_armies = self.graph.get_armies_by_terr_id(to_id)

        from_terr = self.graph.get_terr_by_id(from_id)
        to_terr = self.graph.get_terr_by_id(to_id)

        determine_attack = np.random.uniform()
        result = np.zeros(2) # The number of armies [from, to] lose, respectively
        if from_armies > 3:
            if to_armies > 1: # Three-Two
                if determine_attack < (2890/7776):
                    result[1] = 2
                elif determine_attack < (5165/7776):
                    result[0] = 2
                else:
                    result[0] = 1
                    result[1] = 1
            elif to_armies == 1: # Three-One
                if determine_attack < (855/1296):
                    result[1] = 1
                else:
                    result[0] = 1
            else:
                return self.game_state(), False

        elif from_armies == 3:  # Two-Two
            if to_armies > 1:
                if determine_attack < (295/1296):
                    result[1] = 2
                elif determine_attack < (876/1296):
                    result[0] = 2
                else:
                    result[0] = 1
                    result[1] = 1
            elif to_armies == 1: # Two-One
                if determine_attack < (125/216):
                    result[1] = 1
                else:
                    result[0] = 1
            else:
                return self.game_state(), False

        elif from_armies == 2:
            if to_armies > 1:  # One-Two
                if determine_attack < (55/216):
                    result[1] = 1
                else:
                    result[0] = 1

            elif to_armies == 1: # One-One
                if determine_attack < (15/36):
                    result[1] = 1
                else:
                    result[0] = 1
            else:
                return self.game_state(), False

        elif from_armies == 1:  # No possible attack
            print("ATTACK ERROR: player {} attempting to attack with 1 army".format(attacker))
            return self.game_state(), False

        if (self.remove_armies(from_id, attacker, result[0])):
            print("ATTACK ERROR: Attacking player {} lost a territory while attacking")
            return self.game_state(), False

        if (self.remove_armies(to_id, defender, result[1])):
            self.assign_territory(to_id, attacker, defender)
            self.reinforce_from = from_id
            if not (self.check_loss() == -1):
                self.action_type = ActionType.GAMEOVER
                return self.game_state(), True
            self.action_type = ActionType.REINFORCE
            self.reinforce_edge = edge_id

        return self.game_state(), True

    def reinforce(self, terr_1_armies):
        """
        Action to reinforce a certain edge after defeating an enemy
        Calls fortify
        :param terr_1_armies: How many armies end on the first territory on the edge
        returns: game_state
        """
        return self.fortify(self.reinforce_edge, terr_1_armies, True)


    def fortify(self, edge_id, terr_1_armies=1, reinforce=False):
        """
        Fortifies armies along the edge specified
        :param edge_id: The edge to fortify
        :param terr_1_armies: How many armies end on the first territory on the edge
        returns: game_state
        """
        if edge_id == len(self.graph.edge_list)-1:
            # This is the no fortify action
            self.advance_turn()
            return self.game_state(), False

        nodes = self.graph.edge_list[edge_id]
        if (not (self.graph.get_terr_by_id(nodes[0]).player_id == self.graph.get_terr_by_id(nodes[1]).player_id)):
            print("FORTIFY ERROR: Both territories not owned by the same player")
            return self.game_state(), False

        if (not (self.graph.get_player_id_by_terr_id(nodes[0])  == self.player_turn)):
            print("FORTIFY ERROR: Player does not own the territories")
            return self.game_state(), False

        if self.reinforce_from:
            terr_1 = self.reinforce_from
            if (not(terr_1 == nodes[0])) and (not(terr_1 == nodes[1])):
                print("FORTIFY ERROR: Edge id does not correspond to reinforce_from territory")
                return self.game_state(), False
            if (terr_1 == nodes[0]):
                terr_2 = nodes[1]
            else:
                terr_2 = nodes[0]

        else:
            terr_1 = self.graph.get_terr_by_id(nodes[0])
            terr_2 = self.graph.get_terr_by_id(nodes[1])
        total_armies = terr_1.get_armies() + terr_2.get_armies()

        terr_2_armies = (total_armies - terr_1_armies)
        if terr_2_armies < 1:
            print("FORTIFY ERROR: Total armies between territories {} is not great enough".format(nodes))
            return self.game_state(), False

        terr_1.set_armies(terr_1_armies)
        terr_2.set_armies(terr_2_armies)

        if not reinforce:
            self.advance_turn()
        else:
            self.reinforce_edge = None
            self.reinforce_from = None
            self.action_type = ActionType.ATTACK
        return self.game_state(), True

    def advance_turn(self):
        """
        Moves the turn to the next player, changes action type to allot, calculates allotment
        :returns: game_state
        """
        index = self.player_turn_order[self.player_turn]
        self.player_turn = self.player_turn_order[(index+1)%self.active_players]
        self.action_type = ActionType.ALLOT
        if not self.placement_phase:
            self.calculate_allotment()
        return self.game_state()

    def act(self, action, player_id, action_type, aux_action=1):
        """
        Function that determines which action to call and then calls it
        :param action: index for action vector to perform
        :param player_id: the player making the move
        :action_type : enum int the type of action being performed
        """
        if player_id != self.player_turn:
            print("ACTION ERROR: Wrong player attempting to move. It is player {}'s turn".format(self.player_turn))
            return self.game_state(), False

        if action_type != self.action_type:
            print("ACTION ERROR: Wrong action type. Player must perform {}".format(self.action_type))
            return self.game_state(), False

        if self.action_type == ActionType.ALLOT:
            return self.allot(action)
        elif self.action_type == ActionType.ATTACK:
            return self.attack(action)
        elif self.action_type == ActionType.REINFORCE:
            return self.reinforce(action)
        elif self.action_type == ActionType.FORTIFY:
            return self.fortify(action, aux_action)
        else:
            print("ACTION ERROR: Action type cannot be interpreted")
            return self.game_state(), False

class Player():
    """
    This class defines a player for the Risk board game
    """
    def __init__(self, player_id, name="unassigned", isAgent=False, policy="unassigned"):
        """
        The constructor for a player
        :param int: unique of the player
        :param name: string name of the player
        :param isAgent: boolean whether the player is a learning agent
        :param isActive: boolean whether the player is still in the game
        :param policy: string policy taken by the player
        :return : no return value
        """
        self.player_id = player_id
        self.name = name
        self.isAgent = isAgent
        self.isActive = False
        self.territory_list = []

        # Note: armies or territories less than 1 corresponds to defeat once game is initialized
        self.total_armies = 0
        self.total_territories = 0

        # TODO: Add policies to correspond to agent and bot actions
        if not isAgent:
            self.policy_name = policy
        else:
            self.policy_name = policy

    def set_player_attributes(self, name, isAgent, policy="unassigned"):
        """
        Defines player attributes
        :param name: string name of the player
        :param isAgent: boolean whether player is being trained
        :param policy: string which policy the player executes
        """
        self.name = name
        self.isAgent = isAgent
        self.policy = policy

    def print_player_details(self):
        """
        Prints the player's member variables
        :param : none
        :return : none
        """
        if (self.isAgent):
            print("Player {}, {}, is an agent following {} policy".format(self.player_id, self.name, self.policy))
        else:
            print("Player {}, {}, is a bot following {} policy".format(self.player_id, self.name, self.policy))

    def add_armies(self, num_armies):
        """
        Add armies of the same player to a territory
        Will print out to the console if number of armies is over 30
        :param num_armies: the number of armies to add to the territory
        :return : No return value
        """
        self.total_armies += num_armies
        return

    def lose_armies(self, num_armies):
        """
        Lose armies
        """
        self.total_armies -= num_armies
        return

    def lose(self):
        """
        Deactivate player
        """
        self.isActive = False
        return

    def get_total_armies(self,):
        """
        Fetch total armies controlled by players
        :param none:
        :return self.total_armies: int
        """
        return self.total_armies

    def add_territory(self):
        """
        Add to the total of territories
        :param none:
        :return : No return value
        """
        self.total_territories += 1
        return

    def remove_territory(self, terr_id):
        """
        Remove a territory ID from the player's list
        """
        self.territory_list.remove(terr_id)
        return 

    def subtract_territory(self):
        """
        Remove from the total number of territories
        """
        self.total_territories -= 1
        return

    def get_id(self):
        """
        Fetch the ID of the player
        :param none:
        :return none:
        """
        return self.player_id