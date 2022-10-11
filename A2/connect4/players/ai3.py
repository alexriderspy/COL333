from cmath import inf
import copy
import random

import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer


class AIPlayer:
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        self.depth = 4
        # Do the rest of your implementation here

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        raise NotImplementedError('Whoops I don\'t know what to do')

    def perform_action(self, player_num, action, state):
        tmp_state = copy.deepcopy(state)
        board, num_popouts = tmp_state
        column, is_popout = action
        if not is_popout:
            for row in range(1, board.shape[0]):
                update_row = -1
                if board[row, column] > 0 and board[row - 1, column] == 0:
                    update_row = row - 1
                elif row == board.shape[0] - 1 and board[row, column] == 0:
                    update_row = row
                if update_row >= 0:
                    board[update_row, column] = player_num
                    break
        else:
            for r in range(board.shape[0] - 1, 0, -1):
                board[r, column] = board[r - 1, column]
            board[0, column] = 0
            num_popouts[player_num].decrement()
        return board, num_popouts

        
    def value(self,i,state,depth):
        if i==2:
            if depth >= self.depth or len(get_valid_actions(2 if self.player_number == 1 else 1,state)) == 0:
                return get_pts(self.player_number,state[0])-get_pts(2 if self.player_number == 1 else 1,state[0])
            return self.exp_val(state,depth)
        else:
            if depth >= self.depth or len(get_valid_actions(self.player_number,state)) == 0:
                return get_pts(self.player_number,state[0])-get_pts(2 if self.player_number == 1 else 1,state[0])
            return self.max_val(state,depth)

    def exp_val(self,state,depth):
        v = 0
        valid_actions = get_valid_actions(2 if self.player_number == 1 else 1,state)
        for action in valid_actions:
            st = self.perform_action(2 if self.player_number == 1 else 1,action,state)
            v += self.value(1,st,depth+1)
        v = v/len(valid_actions)
        return v

    def max_val(self,state,depth):
        v = -inf
        valid_actions = get_valid_actions(self.player_number,state)
        for action in valid_actions:
            st = self.perform_action(self.player_number,action,state)
            v = max(v,self.value(2,st,depth+1))
        return v

    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        valid_actions = get_valid_actions(self.player_number,state)
        if len(valid_actions) >= 8:
            self.depth = 3
        elif len(valid_actions) >= 5:
            self.depth = 5
        else:
            self.depth = 6
        action_best = None
        value_of_best_action = -inf
        for action in valid_actions:
            st = self.perform_action(self.player_number, action, state)
            val = self.value(2,st,1)
            if val > value_of_best_action:
                value_of_best_action = val
                action_best = action
        return action_best
