from cmath import inf
import random
import numpy as np
import copy
from typing import List, Tuple, Dict, Union
from connect4.utils import Integer, get_diagonals_primary, get_diagonals_secondary
import time
import functools

class AIPlayer:
    win_pts = [0, 2, 8, 50, 1000]
    best_move = (0,False)
    start_popping = False
    memo = dict()
    transposition_table_pn = dict()
    transposition_table_not_pn = dict()

    terminate = False
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

    def get_valid_actions(self, player_number: int, state: Tuple[np.array, Dict[int, Integer]], local_start_popping) -> List[Tuple[int, bool]]:
        """
        :return: All the valid actions for player (with player_number) for the provided current state of board
        """
        valid_moves = []
        board, temp = state
        pop_out_left = temp[player_number].get_int()
        n = board.shape[1]
        # Adding fill move
        for col in range(n):
            if 0 in board[:, col]:
                valid_moves.append((col, False))
        # Adding popout move
        if pop_out_left > 0 and (self.start_popping or local_start_popping):
            for col in range(n):
                if col % 2 == player_number - 1:
                    # First player is allowed only even columns and second player is allowed only odd columns
                    column = list(board[:, col])
                    ones = column.count(1)
                    twos = column.count(2)
                    all_ones = ones>0 and twos==0
                    all_twos = ones==0 and twos>0
                    all_same = all_ones or all_twos
                    if not(all_same):
                        valid_moves.append((col, True))
        return valid_moves

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

    def evaluate_window(self, window, window_size,player_num):
        score = 0
        if window_size == 4:
            if window.count(player_num) == 4:                               #max 4 consec possible, currently 4
                score += 100000
            elif window.count(player_num) == 3 and window.count(0) == 1:    #max 4 consec possible, currently 3
                score += 8000
            elif window.count(player_num) == 2 and window.count(0) == 2:    #max 4 consec possible, currently 2
                score += 800
            elif window.count(player_num) == 1 and window.count(0) == 3:    #max 4 consec possible, currently 1
                score += 400
        elif window_size == 3:
            if window.count(player_num) == 3 and window.count(0) == 0:    #max 3 consec possible, currently 3 
                score += 30
            elif window.count(player_num) == 2 and window.count(0) == 1:    #max 3 consec possible, currently 2
                score += 25
            elif window.count(player_num) == 1 and window.count(0) == 2:    #max 3 consec possible, currently 1
                score += 20
        elif window_size == 2:
            if window.count(player_num) == 2 and window.count(0) == 0:    #max 2 consec possible, currently 2
                score += 10
            elif window.count(player_num) == 1 and window.count(0) == 1:    #max 2 consec possible, currently 1
                score += 10

        return score

    def evaluate_box(self, box, player_num):
        if box.count(player_num) == len(box):
            return 10*len(box)
        elif box.count(player_num) == len(box)-1:
            return 5*len(box)
        else:
            return 0   

    #recursively divides the window from i to j into pieces of 4 and rest
    def calc_window (self,i,j,window,player_num):
        if (i,j) in self.memo:
            return self.memo[(i,j)]
        if j<i :
            return 0
        if j-i+1 <= 4:
            self.memo[(i,j)] = self.evaluate_window(window[i:j+1],j-i+1,player_num)
            return self.memo[(i,j)]
        ans = -inf
        for k in range(i,j-2):
            ans = max(ans, self.calc_window(k,k+3,window,player_num) + self.calc_window(i,k-1,window,player_num) + self.calc_window(k+4,j,window,player_num))
        self.memo[(i,j)] = ans
        return self.memo[(i,j)]

    def evaluate_shape(self, s, p):
            if s[0:3]==[p,p,p] and s[4]==p and s[6]==p:     #Seven_1
                return 50
            elif s[0:3]==[p,p,p] and s[4]==p and s[8]==p:     #Seven_2
                return 50
            elif s[6:9]==[p,p,p] and s[4]==p and s[0]==p:     #Seven_3
                return 50
            elif s[6:9]==[p,p,p] and s[4]==p and s[2]==p:     #Seven_4
                return 50
            elif s[3:6]==[p,p,p] and s[1]==p and s[7]==p:     #Plus
                return 50
            else:
                return 0

    def score_position(self, board, player_num):
        score = 0
        rows, cols = board.shape
        ## Score columns for distance from centre
        middle = cols//2

        arr = list(board[:, middle])
        for i in range(len(arr)):
            if arr[i]==player_num:
                score+=500*(i+1)

        k = 50
        for j in range(middle):
            arr = list(board[:, j])
            for i in range(len(arr)):
                if arr[i]==player_num:
                    score+=k*(i+1)*(j+1)
        
        for j in range(middle+1, cols):
            arr = list(board[:, j])
            for i in range(len(arr)):
                if arr[i]==player_num:
                    score+=k*(i+1)*(cols-j)

        ## Score Horizontal
        for r in range(rows):
            row_array = list(board[r,:])
            self.memo = dict()
            score += self.calc_window(0,cols-1,row_array,player_num)
        ## Score Vertical
        for c in range(cols):
            col_array = list(board[:,c])
            self.memo = dict()
            score += self.calc_window(0,rows-1,col_array,player_num)

        ## Score posiive sloped diagonal
        for diag in get_diagonals_primary(board):
            self.memo = dict()
            score += self.calc_window(0,len(diag)-1,diag, player_num)

        for diag in get_diagonals_secondary(board):
            self.memo = dict()
            score += self.calc_window(0,len(diag)-1,diag, player_num)

        ## Score 2x2 Boxes
        for r in range(1, rows):
            row_array1 = list(board[r-1,:])
            row_array2 = list(board[r,:])
            for c in range(cols-1):
                box = row_array1[c:c+2] + row_array2[c:c+2] 
                score += self.evaluate_box(box, player_num)

        ## Score 3x3 Boxes and sevens and pluses
        for r in range(2, rows):
            row_array1 = list(board[r-2,:])
            row_array2 = list(board[r-1,:])
            row_array3 = list(board[r,:])
            for c in range(cols-2):
                box = row_array1[c:c+3] + row_array2[c:c+3] + row_array3[c:c+3] 
                score += self.evaluate_box(box, player_num)
                score += self.evaluate_shape(box, player_num)

        return score

    def eval(self, board):
        score = self.score_position(board, self.player_number) - self.score_position(board, 2 if self.player_number == 1 else 1)
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


    def minimax(self, state, player, depth, alpha, beta):
        def func_pn(a,b):
            x_str = (''.join(''.join(str(x) for x in y) for y in a[0]),str(a[1]))
            y_str = (''.join(''.join(str(x) for x in y) for y in b[0]),str(b[1]))
            if x_str not in self.transposition_table_not_pn or y_str not in self.transposition_table_not_pn:
                return self.eval(b[0]) - self.eval(a[0])
            else:
                return self.transposition_table_not_pn[y_str] - self.transposition_table_not_pn[x_str]
            #decreasing

        def func_not_pn(a,b):
            x_str = (''.join(''.join(str(x) for x in y) for y in a[0]),str(a[1]))
            y_str = (''.join(''.join(str(x) for x in y) for y in b[0]),str(b[1]))
            if x_str not in self.transposition_table_pn or y_str not in self.transposition_table_pn:
                return self.eval(a[0]) - self.eval(b[0])
            else:
                return self.transposition_table_pn[x_str] - self.transposition_table_pn[y_str]
            #decreasing

        board = state[0]
        curr = time.time() - self.start_time
        if curr>=self.time*0.8:
            self.terminate = True
            return self.eval(board)
        elif depth==0:
            return self.eval(board)
        else:
            local_start_popping = False
            m,n = board.shape
            unique, counts = np.unique(board, return_counts=True)
            num = dict(zip(unique, counts))
            if 0 in num and num[0]<=(m*n)//2:
                local_start_popping = True

            #If in the state reached during minimax, the number of pieces crosses half, popping can occur
            if player==1:
                valid_moves = self.get_valid_actions(self.player_number, state, local_start_popping)

                if len(valid_moves)==0:
                    return self.eval(board)

                next_states = [self.perform_action(self.player_number, move, state) for move in valid_moves]

                sorted_states = sorted(next_states, key=functools.cmp_to_key(func_pn))


                maxEval = -inf
                for child_state in sorted_states:
                    if self.terminate==False:
                        str_board = ''.join(''.join(str(x) for x in y) for y in child_state[0])
                        popouts = str(child_state[1])
                        eva = self.minimax(child_state, 2, depth-1, alpha, beta)
                        self.transposition_table_not_pn[(str_board,popouts)] = eva
                        maxEval = max(maxEval, eva)
                        alpha = max(alpha, maxEval)
                        if beta<=alpha:
                            break
                    else:
                        break

                return maxEval

            else:
                valid_moves = self.get_valid_actions(2 if self.player_number==1 else 1, state, local_start_popping)
                
                if len(valid_moves)==0:
                    return self.eval(board)
                
                next_states = [self.perform_action(2 if self.player_number==1 else 1, move, state) for move in valid_moves]
                sorted_states = sorted(next_states, key=functools.cmp_to_key(func_not_pn))

                minEval = inf
                for child_state in sorted_states:
                    if self.terminate==False:
                        str_board = ''.join(''.join(str(x) for x in y) for y in child_state[0])
                        popouts = str(child_state[1])
                        eva = self.minimax(child_state, 1, depth-1, alpha, beta)
                        self.transposition_table_pn[(str_board,popouts)] = eva
                        minEval = min(minEval, eva)
                        beta = min(beta, minEval)
                        if beta<=alpha:
                            break
                    else:
                        break

                return minEval

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        
        self.start_time = time.time()
        alpha = -inf
        beta = inf
        board = state[0]
        m,n = board.shape
        unique, counts = np.unique(board, return_counts=True)
        num = dict(zip(unique, counts))
        if (0 in num and num[0]<=(m*n)//2) or (0 not in num and self.start_popping == False):
            self.start_popping = True
        action_best = None

        
        valid_actions = self.get_valid_actions(self.player_number,state,False)
        
        if self.time >= 0 and self.time <= 6:
            if len(valid_actions) >= 16:
                depth = 0
            elif len(valid_actions)  >= 12:
                depth = 1
            elif len(valid_actions) >= 7:
                depth = 2
            elif len(valid_actions) >= 4:
                depth = 3
            else:
                depth = 4
        elif self.time >=7 and self.time <= 8:
            if len(valid_actions) >= 19:
                depth = 1
            elif len(valid_actions)  >= 16:
                depth = 2
            elif len(valid_actions) >= 8:
                depth = 3
            elif len(valid_actions) >= 4:
                depth = 4
            else:
                depth = 6
        elif self.time >=9 and self.time <= 12:
            if len(valid_actions) >= 19:
                depth = 1
            elif len(valid_actions)  >= 16:
                depth = 2
            elif len(valid_actions) >= 8:
                depth = 4
            elif len(valid_actions) >= 4:
                depth = 5
            else:
                depth = 7
        elif self.time >=13 and self.time <= 16:
            if len(valid_actions) >= 19:
                depth = 2
            elif len(valid_actions)  >= 16:
                depth = 3
            elif len(valid_actions) >= 8:
                depth = 4
            elif len(valid_actions) >= 4:
                depth = 6
            else:
                depth = 8
        elif self.time >=17 and self.time <= 20:
            if len(valid_actions) >= 19:
                depth = 2
            elif len(valid_actions)  >= 16:
                depth = 3
            elif len(valid_actions) >= 8:
                depth = 5
            elif len(valid_actions) >= 4:
                depth = 7
            else:
                depth = 8
        if (0 in num and num[0]<=(m*n)//10) :
            depth = 0
        
        depth+=1
        
        while self.terminate == False:
            value_of_best_action = -inf
            for action in valid_actions:
                child_state = self.perform_action(self.player_number, action, state)

                str_board = ''.join(''.join(str(x) for x in y) for y in child_state[0])
                popouts = str(child_state[1])
                eva = self.minimax(child_state, 2, depth-1, alpha, beta)
                self.transposition_table_not_pn[(str_board,popouts)] = eva

                if eva > value_of_best_action:
                    value_of_best_action = eva
                    action_best = action
            depth += 1
            if depth > 25:
                break
        return action_best

    def expectimax(self, state, player, depth):
        board = state[0]
        curr = time.time() - self.start_time
        if curr>=self.time*0.8:
            self.terminate = True
            return self.eval(board)
        elif depth==0:
            return self.eval(board)
        else:
            local_start_popping = False
            m,n = board.shape
            unique, counts = np.unique(board, return_counts=True)
            num = dict(zip(unique, counts))
            if 0 in num and num[0]<=(m*n)//2:
                local_start_popping = True

            #If in the state reached during minimax, the number of pieces crosses half, popping can occur
            if player==1:
                valid_moves = self.get_valid_actions(self.player_number, state, local_start_popping)
                random.shuffle(valid_moves)
                if len(valid_moves)==0:
                    return self.eval(board)

                maxEval = -inf
                for move in valid_moves:
                    if self.terminate==False:
                        child_state = self.perform_action(self.player_number, move, state)
                        eva = self.expectimax(child_state, 2, depth-1)
                        maxEval = max(maxEval, eva)
                    else:
                        break

                return maxEval

            else:
                expEval = 0.0
                valid_moves = self.get_valid_actions(2 if self.player_number==1 else 1, state,local_start_popping)
                random.shuffle(valid_moves)
                if len(valid_moves)==0:
                    return self.eval(board)
                l = 0
                for move in valid_moves:
                    if self.terminate==False:
                        child_state = self.perform_action(2 if self.player_number==1 else 1, move, state)
                        eva = self.expectimax(child_state, 1, depth-1)
                        expEval +=  eva
                        l += 1
                    else:
                        break
                expEval /= l
                return expEval

    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:

        self.start_time = time.time()
        board = state[0]
        m,n = board.shape
        unique, counts = np.unique(board, return_counts=True)
        num = dict(zip(unique, counts))
        if (0 in num and num[0]<=(m*n)//2) or (0 not in num and self.start_popping == False):
            self.start_popping = True
        action_best = None

        valid_actions = self.get_valid_actions(self.player_number,state, False)
        random.shuffle(valid_actions)
        if self.time >= 0 and self.time <= 6:
            if len(valid_actions) >= 16:
                depth = 0
            elif len(valid_actions)  >= 12:
                depth = 1
            elif len(valid_actions) >= 7:
                depth = 2
            elif len(valid_actions) >= 4:
                depth = 3
            else:
                depth = 5
        elif self.time >=7 and self.time <= 8:
            if len(valid_actions) >= 19:
                depth = 1
            elif len(valid_actions)  >= 16:
                depth = 2
            elif len(valid_actions) >= 8:
                depth = 3
            elif len(valid_actions) >= 4:
                depth = 4
            else:
                depth = 6
        elif self.time >=9 and self.time <= 12:
            if len(valid_actions) >= 19:
                depth = 1
            elif len(valid_actions)  >= 16:
                depth = 2
            elif len(valid_actions) >= 8:
                depth = 4
            elif len(valid_actions) >= 4:
                depth = 5
            else:
                depth = 7
        elif self.time >=13 and self.time <= 16:
            if len(valid_actions) >= 19:
                depth = 2
            elif len(valid_actions)  >= 16:
                depth = 3
            elif len(valid_actions) >= 8:
                depth = 4
            elif len(valid_actions) >= 4:
                depth = 6
            else:
                depth = 8
        elif self.time >=17 and self.time <= 20:
            if len(valid_actions) >= 19:
                depth = 2
            elif len(valid_actions)  >= 16:
                depth = 3
            elif len(valid_actions) >= 8:
                depth = 5
            elif len(valid_actions) >= 4:
                depth = 7
            else:
                depth = 8
        
        depth += 1
        while self.terminate == False:
            value_of_best_action = -inf
            for action in valid_actions:
                child_state = self.perform_action(self.player_number, action, state)

                eva = self.expectimax(child_state, 2,depth-1)

                if eva > value_of_best_action:
                    value_of_best_action = eva
                    action_best = action
            depth += 1
            if depth > 25:
                break
        return action_best

#higher scoring if dot is placed in center column
#case 1 popouts were originally 6