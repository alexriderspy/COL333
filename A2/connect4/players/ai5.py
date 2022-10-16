from cmath import inf
import random
import numpy as np
import copy
from typing import List, Tuple, Dict, Union
from connect4.utils import get_valid_actions, get_pts, Integer


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

    def minimax(self, i, state, depth, alpha, beta):
        if i==2:
            if depth >= self.depth or len(get_valid_actions(2 if self.player_number == 1 else 1,state)) == 0:
                return get_pts(self.player_number,state[0])-get_pts(2 if self.player_number == 1 else 1,state[0])
            return self.min_val(state,depth,alpha,beta)
        else:
            if depth >= self.depth or len(get_valid_actions(self.player_number,state)) == 0:
                return get_pts(self.player_number,state[0])-get_pts(2 if self.player_number == 1 else 1,state[0])
            return self.max_val(state,depth,alpha,beta)

    def max_val(self,state,depth,alpha,beta):
        maxEval = -inf
        valid_moves = get_valid_actions(self.player_number,state)
        for move in valid_moves:
            child_state = self.perform_action(self.player_number, move, state)
            maxEval = max(maxEval, self.minimax(2,child_state, depth+1, alpha, beta))
            if maxEval >= beta:
                return maxEval
            alpha = max(alpha, maxEval)
        return maxEval

    def min_val(self,state,depth,alpha,beta):
        minEval = inf
        valid_moves = get_valid_actions(self.player_number,state)
        for move in valid_moves:
            child_state = self.perform_action(self.player_number, move, state)
            minEval = min(minEval, self.minimax(2,child_state, depth+1, alpha, beta))
            if minEval <= alpha:
                return minEval
            beta = min(beta, minEval)
        return minEval

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        num_popout[1].get_int() # gets the popout moves remaining for the 1st player
        num_popout[1].decrement() # decrement the popout moves left for the 1st player
        num_popout[1].increment() # increment the popout moves left for the 1st player"""
        valid_actions = get_valid_actions(self.player_number,state)
        if len(valid_actions) >= 8:
            self.depth = 3
        elif len(valid_actions) >= 4:
            self.depth = 5
        else:
            self.depth = 6
        alpha = -inf
        beta = inf
        action_best = None

        value_of_best_action = -inf
        for action in valid_actions:
            st = self.perform_action(self.player_number, action, state)
            val = self.minimax(2,st,1,alpha,beta)
            if val > value_of_best_action:
                value_of_best_action = val
                action_best = action
        return action_best


    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        raise NotImplementedError('Whoops I don\'t know what to do')


#higher scoring if dot is placed in center column
