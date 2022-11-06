#%%
import numpy as np
import gurobipy as gbp  # Python3.8.12
from typing import List, Tuple
W = [[5, 3, 0, 0],
     [0, 1, 8, 4]]

V = [[0, 6],
     [6, 0]]

a = [0, 4, 6, 10] # x-axis of old
b = [2, 0, 8, 4]  # y-axis of old
# for x-coordinate


def solve_axis(n: int, m: int,
                vector:List) -> Tuple[gbp.Model, List[gbp.Var]]:
    """Multi-facility Location Problem (Rectilinear)
    With Goal-Programming Technique, change variables

    Interaction between new factories
    qjk: amount by which yj is to the right of xk
    pjk: amount by which xj is to the left of xk

    Interaction between new factory and exisiting factories
    (x: new f x-axis, a: old f x-axis)
    rji: amount by which xj is to the right of ai
    sji: amount by which yj is to the left of ai

    Args:
        n (int): # of new factories to be set up
        m (int): # of exisiting factories
    Returns:
        gbp.Model: the optimized model
        X: X-axis
    """

    model = gbp.Model('Rectilinear MF:x-axis')
    X = [0 for _ in range(n)]

    P = [[0 for _ in range(n)] for _ in range(n)]
    Q = [[0 for _ in range(n)] for _ in range(n)]
    R = [[0 for _ in range(m)] for _ in range(n)]
    S = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        X[i] = model.addVar(vtype=gbp.GRB.CONTINUOUS, name=f'x{i}')

    # ========== modeling the interaction between new factories ============
    for j in range(n):
        for k in range(j+1, n):
            P[j][k] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'P_{j}_{k}')
            Q[j][k] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'Q_{j}_{k}')
            model.addConstr(X[j] + P[j][k] - X[k] - Q[j][k] ==
                            0, name='inter-newf interactions')

    # ========== modeling the interaction between new and old ===========
    for j in range(n):
        for i in range(m):
            R[j][i] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'R_{j}_{i}')
            S[j][i] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'S_{j}_{i}')
            model.addConstr(X[j] - R[j][i] + S[j][i] == vector[i],
                            name='inter-new and old interactions')

    # =========== Objective ================
    model.setObjective(
        gbp.quicksum(
            V[j][k]*(P[j][k] + Q[j][k]) for j in range(n) for k in range(j+1, n)
        )
        + gbp.quicksum(
            W[j][i]*(R[j][i] + S[j][i]) for j in range(n) for i in range(m)
        )
        ,gbp.GRB.MINIMIZE
    )

    model.optimize()
    return model, X

# %%
Xmodel, X = solve_axis(n = 2, m = 3, vector = a)
Ymodel, Y = solve_axis(n = 2, m = 3, vector = b)
# https://www.desmos.com/calculator




