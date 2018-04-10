from player import Player
from board import Territory


class ManualPlayer(Player):
    def __init__(self):
        super().__init__()

    def get_attacks(self, valid, graph):
        print("Player {} Attacks:".format(self.player_num))
        for i, attack in enumerate(valid):  # type: int, (Territory, Territory)
            if None in attack:
                print("{}. From {} To {}".format(i, attack[0], attack[1]))
            else:
                print("{}. From {} To {}".format(i, attack[0].name, attack[1].name))
            print('\n')
        choice = min(max(int(input("Enter desired attack number:") or 0), len(valid)), 0)
        return [valid[choice]]

    def get_fortifications(self, valid, graph):
        print("Player {} Fortifications:".format(self.player_num))
        for i, fortification in enumerate(valid):  # type: int, (Territory, Territory, int)
            print("{}. From {} to {}".format(i, fortification[0].name, fortification[1].name))
            print('\n')
        choice = min(max(int(input("Enter desired fortification number:") or 0), len(valid)), 0)
        return [valid[choice]]

    def get_allotment(self, valid, board):
        valid_indices = []
        for index in range(len(valid)):
            if valid[index] == 1:
                valid_indices.append(index)
        print("Player {} Allotments.\nPlayer has {} armies to allot to territories: {}.".format(self.player_num, self.unallocated_armies, valid_indices))
        board.draw()
        choice = -1
        while True:
            try:
                choice = int(input("Which territory # will you allot an army to?:"))
                if (choice > -1) and (choice < len(valid)):
                    if valid[choice] == 1:
                        return choice
                    else:
                        print("Enter integer corresponding to a territory you own")
                else:
                    print("Enter integer corresponding to a territory you own")
            except:
                print("Enter an integer")

        # return choice
        # for i, allotment in enumerate(valid):  # type: int, (Territory, int)
        #     print("{}. {}".format(i, allotment[0].name))
        #     print('\n')
        # choice = min(max(int(input("Enter desired allotment number:") or 0), len(valid)), 0)
        # return [valid[choice]]

