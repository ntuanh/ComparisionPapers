from __future__ import annotations

from typing import List, Tuple

# 1 . use class Hungarian
# 2 . use

class Hungarian:
    def __init__(self, cost_matrix: List[List[float]]):
        if not cost_matrix or not cost_matrix[0]:
            raise ValueError("cost_matrix must be non-empty")
        if any(len(row) != len(cost_matrix[0]) for row in cost_matrix):
            raise ValueError("cost_matrix must be rectangular")
        if len(cost_matrix) != len(cost_matrix[0]):
            raise ValueError("Hungarian implementation expects a square matrix")

        self.original_cost = [row[:] for row in cost_matrix]
        self.C = [row[:] for row in cost_matrix]
        self.size = len(cost_matrix)

    def subtract_each_row(self) -> None:
        for i in range(self.size):
            min_val = min(self.C[i])
            for j in range(self.size):
                self.C[i][j] -= min_val

    def subtract_each_column(self) -> None:
        for j in range(self.size):
            min_val = min(self.C[i][j] for i in range(self.size))
            for i in range(self.size):
                self.C[i][j] -= min_val

    def solve(self) -> Tuple[List[Tuple[int, int]], float]:
        self.subtract_each_row()
        self.subtract_each_column()

        starred = [[False] * self.size for _ in range(self.size)]
        primed = [[False] * self.size for _ in range(self.size)]
        row_covered = [False] * self.size
        col_covered = [False] * self.size

        for i in range(self.size):
            for j in range(self.size):
                if self.C[i][j] == 0 and not row_covered[i] and not col_covered[j]:
                    starred[i][j] = True
                    row_covered[i] = True
                    col_covered[j] = True

        row_covered = [False] * self.size
        col_covered = [False] * self.size

        def cover_columns_with_stars() -> None:
            for j in range(self.size):
                if any(starred[i][j] for i in range(self.size)):
                    col_covered[j] = True

        def find_uncovered_zero() -> Tuple[int, int]:
            for i in range(self.size):
                if row_covered[i]:
                    continue
                for j in range(self.size):
                    if self.C[i][j] == 0 and not col_covered[j]:
                        return i, j
            return -1, -1

        def find_star_in_row(row: int) -> int:
            for j in range(self.size):
                if starred[row][j]:
                    return j
            return -1

        def find_star_in_col(col: int) -> int:
            for i in range(self.size):
                if starred[i][col]:
                    return i
            return -1

        def find_prime_in_row(row: int) -> int:
            for j in range(self.size):
                if primed[row][j]:
                    return j
            return -1

        def clear_primes() -> None:
            for i in range(self.size):
                for j in range(self.size):
                    primed[i][j] = False

        cover_columns_with_stars()

        while sum(col_covered) < self.size:
            row, col = find_uncovered_zero()
            while row == -1:
                min_uncovered = None
                for i in range(self.size):
                    if row_covered[i]:
                        continue
                    for j in range(self.size):
                        if col_covered[j]:
                            continue
                        value = self.C[i][j]
                        if min_uncovered is None or value < min_uncovered:
                            min_uncovered = value

                if min_uncovered is None:
                    break

                for i in range(self.size):
                    for j in range(self.size):
                        if row_covered[i] and col_covered[j]:
                            self.C[i][j] += min_uncovered
                        elif not row_covered[i] and not col_covered[j]:
                            self.C[i][j] -= min_uncovered

                row, col = find_uncovered_zero()

            if row == -1:
                break

            primed[row][col] = True
            star_col = find_star_in_row(row)
            if star_col == -1:
                path = [(row, col)]
                while True:
                    star_row = find_star_in_col(path[-1][1])
                    if star_row == -1:
                        break
                    path.append((star_row, path[-1][1]))
                    prime_col = find_prime_in_row(path[-1][0])
                    path.append((path[-1][0], prime_col))

                for r, c in path:
                    if starred[r][c]:
                        starred[r][c] = False
                    else:
                        starred[r][c] = True

                row_covered = [False] * self.size
                col_covered = [False] * self.size
                clear_primes()
                cover_columns_with_stars()
            else:
                row_covered[row] = True
                col_covered[star_col] = False

        assignment = []
        for i in range(self.size):
            for j in range(self.size):
                if starred[i][j]:
                    assignment.append((i, j))
                    break

        cost = sum(self.original_cost[i][j] for i, j in assignment)
        return assignment, cost

    def run(self) -> None:
        assignment, cost = self.solve()
        print("assignment:", assignment)
        print("cost:", cost)


if __name__ == "__main__":
    cost_matrix = [
        [1500, 4000, 4500],
        [2000, 6000, 3500],
        [2000, 4000, 2500]
    ]
    best_matching = Hungarian(cost_matrix)
    best_matching.run()

# import numpy as np
# from scipy.optimize import linear_sum_assignment
#
# # Cost matrix (n x n)
# C = np.array([
#     [1500, 4000, 4500],
#     [2000, 6000, 3500],
#     [2000, 4000, 2500]
# ])
#
# row_ind, col_ind = linear_sum_assignment(C)
#
# total_cost = C[row_ind, col_ind].sum()
#
# print("Assignments:")
# for r, c in zip(row_ind, col_ind):
#     print(f"Worker {r} -> Job {c}, cost = {C[r, c]}")
#
# print("Total cost:", total_cost)