from cmath import inf
import random
import numpy as np
import copy
from typing import List, Tuple, Dict, Union
from connect4.utils import get_valid_actions, Integer
import time

class AIPlayer:
    win_pts = [0, 0, 2, 500, 2000]

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
        self.best_move = None
        # Do the rest of your implementation here

    def get_score(self, player_number: int, row: Union[np.array, List[int]]):
        score = 0
        n = len(row)
        j = 0
        while j < n:
            if row[j] == player_number:
                count = 0
                while j < n and row[j] == player_number:
                    count += 1
                    j += 1
                #Finding continuous dots of each player
                k = len(self.win_pts) - 1
                score += self.win_pts[count % k] + (count // k) * self.win_pts[k]
            else:
                j += 1
        return score


    def get_diagonals_primary(self, board: np.array) -> List[int]:
        m, n = board.shape
        for k in range(n + m - 1):
            diag = []
            for j in range(max(0, k - m + 1), min(n, k + 1)):
                i = k - j
                diag.append(board[i, j])
            yield diag


    def get_diagonals_secondary(self, board: np.array) -> List[int]:
        m, n = board.shape
        for k in range(n + m - 1):
            diag = []
            for x in range(max(0, k - m + 1), min(n, k + 1)):
                j = n - 1 - x
                i = k - x
                diag.append(board[i][j])
            yield diag

    def get_pts(self, player_number: int, board: np.array) -> int:
        score = 0
        m, n = board.shape
        
        for i in range(m):
            score += self.get_score(player_number, board[i])
        
        for j in range(n):
            score += self.get_score(player_number, board[:, j])
        
        for diag in self.get_diagonals_primary(board):
            score += self.get_score(player_number, diag)
        
        for diag in self.get_diagonals_secondary(board):
            score += self.get_score(player_number, diag)
        return score

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

    def eval(self,state):
        return self.get_pts(self.player_number,state[0])-self.get_pts(2 if self.player_number == 1 else 1,state[0])

    def minimax(self, i, state, depth, alpha, beta):
        if i==2:
            if depth >= self.depth or len(get_valid_actions(2 if self.player_number == 1 else 1,state)) == 0:
                return self.eval(state)
            return self.min_val(state,depth,alpha,beta)
        else:
            if depth >= self.depth or len(get_valid_actions(self.player_number,state)) == 0:
                return self.eval(state)
            return self.max_val(state,depth,alpha,beta)

    def max_val(self,state,depth,alpha,beta):
        maxEval = -inf
        valid_moves = get_valid_actions(self.player_number,state)
        valid_moves = self.order_valid_actions(valid_moves)

        for move in valid_moves:
            child_state = self.perform_action(self.player_number, move, state)
            maxEval = max(maxEval, self.minimax(2,child_state, depth+1, alpha, beta))
            if maxEval >= beta:
                return maxEval
            alpha = max(alpha, maxEval)
        return maxEval

    def min_val(self,state,depth,alpha,beta):
        minEval = inf
        valid_moves = get_valid_actions(2 if self.player_number == 1 else 1,state)
        valid_moves = self.order_valid_actions(valid_moves)

        for move in valid_moves:
            child_state = self.perform_action(2 if self.player_number == 1 else 1, move, state)
            minEval = min(minEval, self.minimax(1,child_state, depth+1, alpha, beta))
            if minEval <= alpha:
                return minEval
            beta = min(beta, minEval)
        return minEval

    def order_valid_actions(self,valid_actions):
        tmp = sorted(valid_actions)
        no_popouts = list(filter((lambda b : b[1] == False),tmp))
        popouts = list(filter((lambda b : b[1] == True),tmp))
        l = len(no_popouts)
        valid_actions = popouts + no_popouts[l//2:] + no_popouts[0:l//2]
        return valid_actions

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        valid_actions = get_valid_actions(self.player_number,state)
        #order valid actions
        
        valid_actions = self.order_valid_actions(valid_actions)
        if self.player_number == 2:
            if len(valid_actions) >= 8:
                self.depth = 3
            elif len(valid_actions) >= 4:
                self.depth = 5
            else:
                self.depth = 6
        else:
            if len(valid_actions) >= 8:
                self.depth = 3
            elif len(valid_actions) >= 4:
                self.depth = 5
            else:
                self.depth = 7
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
