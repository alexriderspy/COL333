from typing import List, Tuple, Dict, Union
import numpy as np

win_pts = [0,0,2,5,20]

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

board = np.array([1,1,1,1,1,1,1,1])
print(get_row_score(1,board))