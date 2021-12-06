#!/usr/bin/python3
# Simplex method
# Author:  HongXin
# 2016.11.17

import numpy as np


def xlpsol(c, A, b):
    """
    Solve linear programming problem with the follow format:
    min     c^Tx
    s.t.    Ax <= b
            x >= 0
    (c^T means transpose of the vector c)
    :return: x - optimal solution, opt - optimal objective value
    """
    (B, T) = __init(c, A, b)
    (m, n) = T.shape
    opt = -T[0, 0]  # -T[0, 0] is exactly the optimal value!
    v_c = T[0, 1:]
    v_b = T[1:, 0]
    v_A = T[1:,1:]

    while True:
        if all(T[0, 1:] >= 0):  # c >= 0
            # just get optimal solution by manipulating index and value
            x = map(lambda t: T[B.index(t) + 1, 0] if t in B else 0,
                    range(0, n - 1))
            return x, opt
        else:
            # choose fist element of v_c smaller than 0
            e = next(x for x in v_c if x < 0)
            delta = map(lambda i: v_b[i]/v_A[i, e] , range(0, m-1))


def __init(c, A, b):
    """
    0   c   0
    b   A   I
    """
    # transfer to vector and matrix
    (c, A, b) = map(lambda t: np.array(t), [c, A, b])
    [m, n] = A.shape
    if m != b.size:
        print('The size of b must equal with the row of A!')
        exit(1)
    if n != c.size:
        print('The size of c must equal with the column of A!')
        exit(1)
    part_1 = np.vstack((0, b.reshape(b.size, 1)))
    part_2 = np.vstack((c, A))
    part_3 = np.vstack((np.zeros(m), np.identity(m)))
    return range(n, n + m), np.hstack((np.hstack((part_1, part_2)), part_3))


def __pivot():
    pass


if __name__ == '__main__':
    c = [-1, -14, -6]
    A = [[1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 3, 1]]
    b = [4, 2, 3, 6]
    [x, opt] = xlpsol(c, A, b)