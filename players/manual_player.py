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

    def get_allotments(self, valid, graph):
        print("Player {} Allotments".format(self.player_num))
        for i, allotment in enumerate(valid):  # type: int, (Territory, int)
            print("{}. {}".format(i, allotment[0].name))
            print('\n')
        choice = min(max(int(input("Enter desired allotment number:") or 0), len(valid)), 0)
        return [valid[choice]]
