from typing import List, Tuple, Dict, Union

import numpy as np

from connect4.config import debug_mode, win_pts


class Integer:
    """
    Used to manage num pop counts in state. Normal python integer cannot be passed by reference.
    Thus, we created this integer class
    """

    def __init__(self, i: int):
        self._i = i
        self._initial = i

    def decrement(self) -> None:
        assert not debug_mode or self._i > 0
        self._i -= 1

    def increment(self) -> None:
        assert not debug_mode or self._i <= self._initial
        self._i += 1

    def get_int(self) -> int:
        return self._i


def get_valid_actions(player_number: int, state: Tuple[np.array, Dict[int, Integer]]) -> List[Tuple[int, bool]]:
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
    if pop_out_left > 0:
        for col in range(n):
            if col % 2 == player_number - 1:
                # First player is allowed only even columns and second player is allowed only odd columns
                if board[:, col].any():
                    valid_moves.append((col, True))
    return valid_moves


def get_row_score(player_number: int, row: Union[np.array, List[int]]):
    score = 0
    n = len(row)
    j = 0
    while j < n:
        if row[j] == player_number:
            count = 0
            while j < n and row[j] == player_number:
                count += 1
                j += 1
            k = len(win_pts) - 1
            score += win_pts[count % k] + (count // k) * win_pts[k]
        else:
            j += 1
    return score


def get_diagonals_primary(board: np.array) -> List[int]:
    m, n = board.shape
    for k in range(n + m - 1):
        diag = []
        for j in range(max(0, k - m + 1), min(n, k + 1)):
            i = k - j
            diag.append(board[i, j])
        yield diag


def get_diagonals_secondary(board: np.array) -> List[int]:
    m, n = board.shape
    for k in range(n + m - 1):
        diag = []
        for x in range(max(0, k - m + 1), min(n, k + 1)):
            j = n - 1 - x
            i = k - x
            diag.append(board[i][j])
        yield diag


def get_pts(player_number: int, board: np.array) -> int:
    """
    :return: Returns the total score of player (with player number) on the board
    """
    score = 0
    m, n = board.shape
    # score in rows
    for i in range(m):
        score += get_row_score(player_number, board[i])
    # score in columns
    for j in range(n):
        score += get_row_score(player_number, board[:, j])
    # scores in diagonals_primary
    for diag in get_diagonals_primary(board):
        score += get_row_score(player_number, diag)
    # scores in diagonals_secondary
    for diag in get_diagonals_secondary(board):
        score += get_row_score(player_number, diag)
    return score
