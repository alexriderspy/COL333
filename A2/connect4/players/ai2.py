from cmath import inf
import random
import numpy as np
import copy
from typing import List, Tuple, Dict, Union
from connect4.utils import get_valid_actions, Integer


class AIPlayer:
    win_pts = [0, 2, 8, 18, 1000]
    best_move = (0,False)
     
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
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
        # score in rows
        for i in range(m):
            score += self.get_score(player_number, board[i])
        # score in columns
        for j in range(n):
            score += self.get_score(player_number, board[:, j])
        # scores in diagonals_primary
        for diag in self.get_diagonals_primary(board):
            score += self.get_score(player_number, diag)
        # scores in diagonals_secondary
        for diag in self.get_diagonals_secondary(board):
            score += self.get_score(player_number, diag)
        return score

    def eval(self, board):
        score = self.get_pts(1, board)
        score-= self.get_pts(2, board)
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

    def minimax(self, state, player_number, depth, alpha, beta, get_child):
        board = state[0]
        if depth==0:
            return self.eval(board)
        else:
            valid_moves = get_valid_actions(self.player_number, state)
            evals = []
            if player_number==1:
                maxEval = -inf
                for move in valid_moves:
                    child_state = self.perform_action(1, move, state)
                    eva = self.minimax(child_state, 2, depth-1, alpha, beta, False)
                    evals.append(eva)
                    maxEval = max(maxEval, eva)
                    alpha = max(alpha, maxEval)
                    if beta<=alpha:
                        break

                if get_child:
                    for i in range(len(evals)):
                        if evals[i]==maxEval:
                            self.best_move = valid_moves[i]

                return maxEval

            else:
                minEval = inf
                for move in valid_moves:
                    child_state = self.perform_action(2, move, state)
                    eva = self.minimax(child_state, 1, depth-1, alpha, beta, False)
                    evals.append(eva)
                    minEval = min(minEval, eva)
                    beta = min(beta, eva)
                    if beta<=alpha:
                        break

                if get_child:
                    for i in range(len(evals)):
                        if evals[i]==minEval:
                            self.best_move = valid_moves[i]

                return minEval

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        num_popout[1].get_int() # gets the popout moves remaining for the 1st player
        num_popout[1].decrement() # decrement the popout moves left for the 1st player
        num_popout[1].increment() # increment the popout moves left for the 1st player"""
        depth = 4
        alpha = -inf
        beta = inf
        val = self.minimax(state, self.player_number, depth, alpha, beta, True)
        print(val)
        print(self.best_move)
        return self.best_move


    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        raise NotImplementedError('Whoops I don\'t know what to do')


#higher scoring if dot is placed in center column
