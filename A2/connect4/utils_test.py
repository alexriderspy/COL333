import unittest
import numpy as np
from connect4.utils import get_row_score, get_pts, get_diagonals_primary, get_diagonals_secondary


class UtilsTest(unittest.TestCase):

    # Returns True or False.
    def test_get_row_score(self):
        row = [1, 1, 0, 0, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1]
        self.assertEqual(get_row_score(1, row), 9)
        self.assertEqual(get_row_score(2, row), 5)
        row = np.array([0, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1])
        self.assertEqual(get_row_score(1, row), 27)

    def test_get_pts_rows(self):
        mat = np.array([[0, 1, 1, 0, 1], [0, 1, 0, 2, 2], [0, 1, 1, 0, 0]])
        self.assertEqual(get_pts(1, mat), 13)

        mat2 = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]])
        self.assertEqual(get_pts(1, mat2), 48)

    def test_get_diagonals_primary(self):
        # [1, 2, 3],
        # [4, 5, 6],
        # [7, 8, 9],
        # [1, 3, 4]
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 4]])
        diagonals_expected = [[1], [4, 2], [7, 5, 3], [1, 8, 6], [3, 9], [4]]
        for i, diag in enumerate(get_diagonals_primary(mat)):
            self.assertEqual(diag, diagonals_expected[i])

        # [1, 2, 3],
        # [4, 5, 6]
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        diagonals_expected = [[1], [4, 2], [5, 3], [6]]
        for i, diag in enumerate(get_diagonals_primary(mat)):
            self.assertEqual(diag, diagonals_expected[i])

    def test_get_diagonals_secondary(self):
        # [1, 2, 3],
        # [4, 5, 6],
        # [7, 8, 9],
        # [1, 3, 4]
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 4]])
        diagonals_expected = [[3], [6, 2], [9, 5, 1], [4, 8, 4], [3, 7], [1]]
        for i, diag in enumerate(get_diagonals_secondary(mat)):
            self.assertEqual(diag, diagonals_expected[i])

        # [1, 2, 3],
        # [4, 5, 6]
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        diagonals_expected = [[3], [6, 2], [5, 1], [4]]
        for i, diag in enumerate(get_diagonals_secondary(mat)):
            self.assertEqual(diag, diagonals_expected[i])


if __name__ == '__main__':
    unittest.main()
