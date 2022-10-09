# system libs
import argparse
import json
import multiprocessing as mp
import random
import sys
import tkinter as tk
from datetime import datetime

# 3rd party libs
from threading import Thread
from time import sleep
from typing import Tuple, Callable, Dict

import numpy as np

from connect4.utils import get_pts, get_valid_actions, Integer

# Local libs
from connect4.players.ai import AIPlayer
# from connect4.players.ai2 import AIPlayer as AIPlayer2
from connect4.players.random import RandomPlayer
from connect4.players.human import HumanPlayer


TimeLimitExceedAction = (1000, True)

# https://stackoverflow.com/a/37737985
def turn_worker(state: Tuple[np.array, Dict[int, Integer]], send_end,
                p_func: Callable[[Tuple[np.array, Dict[int, Integer]]], Tuple[int, bool]]):
    send_end.send(p_func(state))


class Game:
    def __init__(self, player1, player2, time: int, board_init: np.array, m: int, n: int, popout_moves: int):
        """
        :param player1:
        :param player2:
        :param time: Time in milliseconds
        :param m:
        :param n:
        :param popout_moves:
        """
        self.players = [player1, player2]
        self.colors = ['white', 'yellow', 'red']  # Extra white color added
        self.current_turn = 0
        self.m = m
        self.n = n
        # Giving both of the players same number of popout moves.
        self.state = board_init, {1: Integer(popout_moves), 2: Integer(popout_moves)}
        self.gui_board = []
        self.game_over = False
        self.ai_turn_limit = time

        board = self.state[0]
        # Log: Writing initial state of the board to log file
        # Explain the log file
        with open('logs.txt', 'w') as log_file:
            s = f'{m} {n}\n'
            for i in range(m):
                for j in range(n):
                    s += str(board[i][j]) + ' '
                s += '\n'
            s += f'{popout_moves}\n'
            log_file.write(s)
            print(s)

        # https://stackoverflow.com/a/38159672
        root = tk.Tk()
        root.title('Extended Connect 4')

        self.current = tk.Label(root, text="Current:")
        self.current.pack()

        self.player1_string = tk.Label(root, text=player1.player_string)
        self.player1_string.pack()

        self.player2_string = tk.Label(root, text=player2.player_string)
        self.player2_string.pack()

        height = m * 100
        width = n * 100
        self.c = tk.Canvas(root, height=height, width=width)
        self.c.pack()
        for j in range(n):
            column = []
            row = j * 100
            for i in range(m):
                col = i * 100
                c = board[i][j]
                column.append(self.c.create_oval(row, col, row + 100, col + 100, fill=self.colors[c]))
            self.gui_board.append(column)

        thread = Thread(target=self.threaded_function, args=(100000,))
        thread.start()
        root.mainloop()

    # def get_random_board(self):
    #     m = self.m
    #     n = self.n
    #     board = np.zeros([m, n]).astype(np.uint8)
    #     # for j in range(n):
    #     #     x = random.randrange(m)
    #     #     for i in range(x, m):
    #     #         p = random.randrange(2) + 1
    #     #         board[i][j] = p
    #     return board

    def threaded_function(self, arg):
        sleep(1)  # Wait for tkinter to setup
        for i in range(arg):
            self.make_move()
            # wait 0.01 sec in between
            sleep(0.01)
            if self.game_over:
                with open('logs.txt', 'w') as log_file:
                    s = 'Game Over\n'
                    s += f'Player 1 Score: {get_pts(1, self.state[0])}\n'
                    s += f'Player 2 Score: {get_pts(2, self.state[0])}\n'
                    log_file.write(s)
                    print(s)
                break

    def make_move(self):
        current_player = self.players[self.current_turn]
        valid_actions = get_valid_actions(current_player.player_number, self.state)

        if len(valid_actions) == 0:
            self.game_over = True

        if not self.game_over:
            if current_player.type == 'ai':
                if self.players[int(not self.current_turn)].type == 'random':
                    p_func = current_player.get_expectimax_move
                else:
                    p_func = current_player.get_intelligent_move
                try:
                    recv_end, send_end = mp.Pipe(False)
                    p = mp.Process(target=turn_worker, args=(self.state, send_end, p_func))
                    p.start()
                    if p.join(self.ai_turn_limit) is None and p.is_alive():
                        p.terminate()
                        raise Exception('Player Exceeded time limit')
                    action = recv_end.recv()
                except Exception as e:
                    uh_oh = 'Uh oh.... something is wrong with Player {}'
                    print(uh_oh.format(current_player.player_number))
                    print(e)
                    action = TimeLimitExceedAction
            else:
                action = current_player.get_move(self.state)

            if action == TimeLimitExceedAction:
                log_action = {'player': current_player.player_number, 'move': 'TLE', 'is_pop': 'TLE'}
            elif action not in valid_actions:
                # Invalid move by the player. Don't do anything
                log_action = {'player': current_player.player_number, 'move': 'invalid', 'is_pop': 'invalid'}
            else:
                move, is_popout = action
                self.update_board(int(move), current_player.player_number, is_popout=is_popout)
                log_action = {'player': current_player.player_number, 'move': int(move), 'is_pop': is_popout}

            # Log: Writing action to log file
            with open('logs.txt', 'a') as log_file:
                log_file.write(json.dumps(log_action) + '\n')

            self.current_turn = int(not self.current_turn)

            self.current.configure(text=f'cur: {self.players[self.current_turn].player_string}, ')
            self.player1_string.configure(text=f'player1: {get_pts(1, self.state[0])}, '
                                               f'popout left: {self.state[1][1].get_int()}')
            self.player2_string.configure(text=f' player2: {get_pts(2, self.state[0])}, '
                                               f'popout left: {self.state[1][2].get_int()}')

    def update_board(self, column: int, player_num: int, is_popout: bool = False):
        board, num_popouts = self.state
        if not is_popout:
            if 0 in board[:, column]:
                for row in range(1, board.shape[0]):
                    update_row = -1
                    if board[row, column] > 0 and board[row - 1, column] == 0:
                        update_row = row - 1
                    elif row == board.shape[0] - 1 and board[row, column] == 0:
                        update_row = row
                    if update_row >= 0:
                        board[update_row, column] = player_num
                        self.c.itemconfig(self.gui_board[column][update_row], fill=self.colors[self.current_turn + 1])
                        break
            else:
                err = 'Invalid move by player {}. Column {}'.format(player_num, column, is_popout)
                raise Exception(err)
        else:
            if 1 in board[:, column] or 2 in board[:, column]:
                for r in range(board.shape[0] - 1, 0, -1):
                    board[r, column] = board[r - 1, column]
                    self.c.itemconfig(self.gui_board[column][r], fill=self.colors[
                        board[r, column]])  # this needs to be tweaked
                board[0, column] = 0
                self.c.itemconfig(self.gui_board[column][0], fill=self.colors[0])
            else:
                err = 'Invalid move by player {}. Column {}'.format(player_num, column)
                raise Exception(err)
            num_popouts[player_num].decrement()


def get_start_board(file_pth: str) -> Tuple[int, np.array]:
    num_pop_outs = -1
    b = []
    with open(file_pth) as f:
        for line in f:
            line = line.strip()
            if num_pop_outs == -1:
                num_pop_outs = int(line)
            else:
                row = [int(ch) for ch in line.split(' ')]
                b.append(row)
    board = np.array(b, dtype=int)
    return num_pop_outs, board


def main(player1: str, player2: str, init_fine_name: str, time: int):
    def make_player(name, num):
        if name == 'ai':
            return AIPlayer(num, time)
        elif name == 'random':
            return RandomPlayer(num)
        elif name == 'human':
            return HumanPlayer(num)

    random.seed(datetime.now())
    # Both players should have the same number of columns for pop out movements, thus number of columns should be even
    num_pop_outs, board = get_start_board(init_fine_name)
    m, n = board.shape
    Game(make_player(player1, 1), make_player(player2, 2), time, board, m, n, num_pop_outs)


if __name__ == '__main__':
    player_types = ['ai', 'random', 'human']
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', choices=player_types)
    parser.add_argument('player2', choices=player_types)
    parser.add_argument("start_file", help="Initial state of the game (Num popout moves, Board)")
    # Giving Max of 60 seconds to AI agent in each step
    parser.add_argument('--time',
                        type=int,
                        default=20,
                        help='Time to wait for a move in seconds (int)')
    args = parser.parse_args()
    main(args.player1, args.player2, args.start_file, args.time)
