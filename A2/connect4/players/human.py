import sys
from typing import Tuple, Dict
import numpy as np
from connect4.utils import get_valid_actions, Integer


def get_input() -> str:
    print('Enter your move: ')
    inp = sys.stdin.readline()
    inp = inp.replace('\n', '')
    return inp


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    @staticmethod
    def get_action(inp: str) -> Tuple[int, bool]:
        if inp[-1] == 'P':
            action = int(inp[:-1]), True
        else:
            action = int(inp), False
        return action

    def get_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state returns the next action
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        valid_actions = get_valid_actions(self.player_number, state)
        action = self.get_action(get_input())
        if action not in valid_actions:
            print('Invalid Move: Choose from: {}'.format(valid_actions))
            print('Turning to other player')
            # action = self.get_action(get_input())
        return action
