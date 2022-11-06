#%%
import multiprocessing
import time
import numpy as np
import gurobipy as gbp  # Python3.8.12
from typing import List, Tuple, Dict
from utils import timer


data = {
    (16, 8): 4,
    (9, 9): 5,
    (18, 3): 3
}


# (long side, short side)
J = []
for d in data:
    for _ in range(data[d]):
        J.append(d)


toydata = {
    (16, 8): 2,
    (9, 9): 2,
    (18, 3): 2

}
toyJ = []
for d in data:
    for _ in range(toydata[d]):
        toyJ.append(d)


def solveCuttingStock_linear(J: List[Tuple[int, int]], m: int, RUNTIMELIMIT: int):
    """Cutting-Stock Problem Solver

    Args:
        J (List[Tuple[int,int]]): the set of rectangles represented by (p,q) to be placed
    Notations:
    X' (|J|), xi': distance between rec i's center and (0,0) along the x-axis
    Y' (|J|), yi': distance between rec i's center and (0,0) along the y-axis
    S (|J|),  si: orientation indicator of rec i
            si = 1: rectangle i is placed with long side parallel to x-axis (取pi)
            si = 0: rectangle i is placed with long side parallel to y-axis (取qi)

    U (|J| x |J|)
            應該是用C(|J|, 2)的概念，就是只有三角矩陣（例： i < k，上三角）
            uik = 1: rec i is to the right of rec k                i k
            uik = 0: rec i is to the left of rec k                 k i
    V (|J| x |J|)
            同上。
            vik = 1: rec i is to the top of rec k
            vik = 0: rec i is to the bottom of rec k
    Objective: (approx) ln X + ln Y
    """
    model = gbp.Model('Cutting Stock (piece-wise linear)')

    # ======= VARIABLES ==========
    # variables: center of each placed rectangle i (xi', yi')
    Xprime = [None for _ in range(len(J))]
    Yprime = [None for _ in range(len(J))]
    # variables: big-M like upper bound
    xupper = yupper = sum(max(p, q) for p, q in J)
    xlower = ylower = 1  # ln 0: undefined, ln 1 = 0

    # Backpack Size
    X = model.addVar(lb=xlower, ub=xupper, vtype=gbp.GRB.CONTINUOUS, name='X')
    Y = model.addVar(lb=ylower, ub=yupper, vtype=gbp.GRB.CONTINUOUS, name='Y')

    S = [None for _ in range(len(J))]
    U = [[None for _ in range(len(J))] for _ in range(len(J))]
    V = [[None for _ in range(len(J))] for _ in range(len(J))]

    # Variables
    for i in range(len(J)):
        Xprime[i] = model.addVar(lb=xlower, ub=xupper,
                                 vtype=gbp.GRB.CONTINUOUS, name=f'x{i}\'')
        Yprime[i] = model.addVar(lb=ylower, ub=yupper,
                                 vtype=gbp.GRB.CONTINUOUS, name=f'y{i}\'')
        S[i] = model.addVar(vtype=gbp.GRB.BINARY, name=f's{i}')
        for k in range(len(J)):
            U[i][k] = model.addVar(vtype=gbp.GRB.BINARY, name=f'u_{i}_{k}')
            V[i][k] = model.addVar(vtype=gbp.GRB.BINARY, name=f'v_{i}_{k}')

    # ======= CONSTRAINTS ==========
    # Constraints: Non-overlapping conditions
    # see slides ORA_07_Facility_Bin-Packing_2022_ncku.pdf p.46 for the formula indices
    for i in range(len(J)):
        for k in range(i+1, len(J)):
            pi, qi = J[i]
            pk, qk = J[k]
            RHS_1 = 1/2 * (pi * S[i] + qi * (1 - S[i]) +
                           pk * S[k] + qk * (1 - S[k]))
            RHS_2 = 1/2 * (pi * (1 - S[i]) + qi *
                           S[i] + pk * (1 - S[k]) + qk * S[k])
            model.addConstr((Xprime[i] - Xprime[k]) + U[i][k] * xupper + V[i][k] * xupper
                            >= RHS_1, name='(2)')
            model.addConstr((Xprime[k] - Xprime[i]) + (1 - U[i][k]) * xupper + V[i][k] * xupper
                            >= RHS_1, name='(3)')
            model.addConstr((
                (Yprime[i] - Yprime[k]) + U[i][k] *
                yupper + (1 - V[i][k]) * yupper
                >= RHS_2), name='(4)')
            model.addConstr((
                (Yprime[k] - Yprime[i]) + (1 - U[i][k]) *
                yupper + (1 - V[i][k]) * yupper
                >= RHS_2), name='(5)')
    # Constraints: Within the enveloping rectangle
    for i in range(len(J)):
        pi, qi = J[i]
        # rectangles need to stay at First Quadrant
        model.addConstr(X >= Xprime[i] + 1/2 *
                        (pi * S[i] + qi * (1 - S[i])), name='(6)')  # "xupper >= X" is specified in addVar()
        model.addConstr(Y >= Yprime[i] + 1/2 *
                        (pi * (1 - S[i]) + qi * S[i]), name='(7)')
        model.addConstr(Xprime[i] - 1/2 * (pi * S[i] +
                        qi * (1 - S[i])) >= 0, name='(8)')
        model.addConstr(Yprime[i] - 1/2 *
                        (pi * (1 - S[i]) + qi * S[i]) >= 0, name='(9)')

    # ======= OBJECTIVE ==========
    # piecewise linearization technique
    # 1) take log: min xy -> min (lnx + lny)
    # 2) linearize: lnx ~= lna1 + s1(x - a1) + sum_(j=2)^(m-1)(s{j} - s{j-2})/2 * (|x - aj| + x - aj)
    # linearizing X

    interval = (xupper - xlower) / (m-1)
    # linear schedule approximation
    # setting up a series of breakpoints (a)
    a = [xlower + interval * i for i in range(m)]

    u = [None for _ in range(m)]
    w = [None for _ in range(m)]
    for j in range(m):
        u[j] = model.addVar(vtype=gbp.GRB.BINARY, name=f'u{j}')
        w[j] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'w{j}')
        # see slides ORA_07_Facility_Bin-Packing_2022_ncku.pdf p.50 for the formula indices

        # Gurobi addConstr only accepts <=, >=, == to appear once at a time
        # so every formula of form "a <= b <= c" is split into two constraints
        model.addConstr(-a[m-1] * u[j] <= X - a[j], name=f'obj (i)-1')
        model.addConstr(X - a[j] <= a[m-1] * (1 - u[j]), name=f'obj (i)-2')

        model.addConstr(-a[m-1] * u[j] <= w[j], name=f'obj (ii)-1')
        model.addConstr(w[j] <= a[m-1] * u[j], name=f'obj (ii)-2')

        model.addConstr(a[m-1] * (u[j] - 1) + X <= w[j], name=f'obj (iii)-1')
        model.addConstr(w[j] <= a[m-1] * (1 - u[j]) + X, name=f'obj (iii)-2')
    for j in range(1, m):
        model.addConstr(u[j] >= u[j-1], name=f'obj (iv)')

    # linearizing Y
    _u = [None for _ in range(m)]
    _w = [None for _ in range(m)]
    for j in range(m):
        _u[j] = model.addVar(vtype=gbp.GRB.BINARY, name=f'_u{j} (y)')
        _w[j] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'_w{j} (y)')
        # see slides ORA_07_Facility_Bin-Packing_2022_ncku.pdf p.50 for the formula indices

        # Gurobi addConstr only accepts <=, >=, == to appear once at a time
        # so every formula of form "a <= b <= c" is split into two constraints
        model.addConstr(-a[m-1] * _u[j] <= Y - a[j], name=f'obj (i)-1 (y)')
        model.addConstr(Y - a[j] <= a[m-1] * (1 - _u[j]),
                        name=f'obj (i)-2 (y)')

        model.addConstr(-a[m-1] * _u[j] <= _w[j], name=f'obj (ii)-1 (y)')
        model.addConstr(_w[j] <= a[m-1] * _u[j], name=f'obj (ii)-2 (y)')

        model.addConstr(a[m-1] * (_u[j] - 1) + Y <=
                        _w[j], name=f'obj (iii)-1 (y)')
        model.addConstr(_w[j] <= a[m-1] * (1 - _u[j]) +
                        Y, name=f'obj (iii)-2 (y)')

    for j in range(1, m):
        model.addConstr(_u[j] >= _u[j-1], name=f'obj (iv) (y)')

    # slopes
    s = [(np.log(a[j+1]) - np.log(a[j])) / (a[j+1] - a[j]) for j in range(m-1)]
    print(f'line segments slope: {s}')
    print(f'len(s): {len(s)}')
    # ~lnx + ~lny
    model.setObjective(
        np.log(a[0]) + s[0] * (X - a[0]) + gbp.quicksum((s[j] - s[j-1])
                                                        * (a[j] * u[j] + X - a[j] - w[j]) for j in range(1, m-1)) +
        np.log(a[0]) + s[0] * (Y - a[0]) + gbp.quicksum((s[j] - s[j-1])
                                                        * (a[j] * _u[j] + Y - a[j] - _w[j]) for j in range(1, m-1)),
        gbp.GRB.MINIMIZE)

    model.setParam('TimeLimit', RUNTIMELIMIT)
    model.optimize()

    return model, X, Y, Xprime, Yprime


# %%

def solveCuttingStock(J: List[Tuple[int, int]], RUNTIMELIMIT:int):
    """Cutting-Stock Problem Solver

    Args:
        J (List[Tuple[int,int]]): the set of rectangles represented by (p,q) to be placed
    Notations:
    X' (|J|), xi': distance between rec i's center and (0,0) along the x-axis
    Y' (|J|), yi': distance between rec i's center and (0,0) along the y-axis
    S (|J|),  si: orientation indicator of rec i
            si = 1: rectangle i is placed with long side parallel to x-axis (取pi)
            si = 0: rectangle i is placed with long side parallel to y-axis (取qi)

    U (|J| x |J|)
            應該是用C(|J|, 2)的概念，就是只有三角矩陣（例： i < k，上三角）
            uik = 1: rec i is to the right of rec k                i k
            uik = 0: rec i is to the left of rec k                 k i
    V (|J| x |J|)
            同上。
            vik = 1: rec i is to the top of rec k
            vik = 0: rec i is to the bottom of rec k
    Objective; X * Y
    """
    model = gbp.Model('Cutting Stock')

    model.setParam('NonConvex', 2)

    # ======= VARIABLES ==========
    # variables: center of each placed rectangle i (xi', yi')
    Xprime = [None for _ in range(len(J))]
    Yprime = [None for _ in range(len(J))]
    # variables: big-M like upper bound
    xupper = yupper = sum(max(p, q) for p, q in J)
    xlower = ylower = 1  # ln 0: undefined, ln 1 = 0

    # Backpack Size
    X = model.addVar(lb=xlower, ub=xupper, vtype=gbp.GRB.CONTINUOUS, name='X')
    Y = model.addVar(lb=ylower, ub=yupper, vtype=gbp.GRB.CONTINUOUS, name='Y')

    S = [None for _ in range(len(J))]
    U = [[None for _ in range(len(J))] for _ in range(len(J))]
    V = [[None for _ in range(len(J))] for _ in range(len(J))]

    # Variables
    for i in range(len(J)):
        Xprime[i] = model.addVar(lb=xlower, ub=xupper,
                                 vtype=gbp.GRB.CONTINUOUS, name=f'x{i}\'')
        Yprime[i] = model.addVar(lb=ylower, ub=yupper,
                                 vtype=gbp.GRB.CONTINUOUS, name=f'y{i}\'')
        S[i] = model.addVar(vtype=gbp.GRB.BINARY, name=f's{i}')
        for k in range(len(J)):
            U[i][k] = model.addVar(vtype=gbp.GRB.BINARY, name=f'u_{i}_{k}')
            V[i][k] = model.addVar(vtype=gbp.GRB.BINARY, name=f'v_{i}_{k}')

    # ======= CONSTRAINTS ==========
    # Constraints: Non-overlapping conditions
    # see slides ORA_07_Facility_Bin-Packing_2022_ncku.pdf p.46 for the formula indices
    for i in range(len(J)):
        for k in range(i+1, len(J)):
            pi, qi = J[i]
            pk, qk = J[k]
            RHS_1 = 1/2 * (pi * S[i] + qi * (1 - S[i]) +
                           pk * S[k] + qk * (1 - S[k]))
            RHS_2 = 1/2 * (pi * (1 - S[i]) + qi *
                           S[i] + pk * (1 - S[k]) + qk * S[k])
            model.addConstr((Xprime[i] - Xprime[k]) + U[i][k] * xupper + V[i][k] * xupper
                            >= RHS_1, name='(2)')
            model.addConstr((Xprime[k] - Xprime[i]) + (1 - U[i][k]) * xupper + V[i][k] * xupper
                            >= RHS_1, name='(3)')
            model.addConstr((
                (Yprime[i] - Yprime[k]) + U[i][k] *
                yupper + (1 - V[i][k]) * yupper
                >= RHS_2), name='(4)')
            model.addConstr((
                (Yprime[k] - Yprime[i]) + (1 - U[i][k]) *
                yupper + (1 - V[i][k]) * yupper
                >= RHS_2), name='(5)')
    # Constraints: Within the enveloping rectangle
    for i in range(len(J)):
        pi, qi = J[i]
        # rectangles need to stay at First Quadrant
        model.addConstr(X >= Xprime[i] + 1/2 *
                        (pi * S[i] + qi * (1 - S[i])), name='(6)')  # "xupper >= X" is specified in addVar()
        model.addConstr(Y >= Yprime[i] + 1/2 *
                        (pi * (1 - S[i]) + qi * S[i]), name='(7)')
        model.addConstr(Xprime[i] - 1/2 * (pi * S[i] +
                        qi * (1 - S[i])) >= 0, name='(8)')
        model.addConstr(Yprime[i] - 1/2 *
                        (pi * (1 - S[i]) + qi * S[i]) >= 0, name='(9)')

    # ======= OBJECTIVE ==========
    # directly using the non-linear term
    model.setObjective(X * Y, gbp.GRB.MINIMIZE)
    model.setParam('TimeLimit', RUNTIMELIMIT)
    model.optimize()
    return model, X, Y, Xprime, Yprime



# %%
def show_results(key, *args):
    model, X, Y, Xprime, Yprime = args
    print(f' ========== {key} ===========')
    print(f'model obj: {model.objVal}')
    print(f'(sub) optimal backpack size: {X.x} * {Y.x} = {X.x * Y.x}')
    print(f'stock center:')
    Jnum = len(Xprime)
    for i in range(Jnum):
        print(f'\t {i}th stock center: {round(Xprime[i].x, 2)}, {round(Yprime[i].x)}')

#%%
runtimelimit = 1 * 60
model, X, Y, Xprime, Yprime = solveCuttingStock(J, runtimelimit)
linmodel, linX, linY, linXprime, linYprime = solveCuttingStock_linear(J, m = 20, RUNTIMELIMIT = runtimelimit)
#%%
show_results('non-linear', model, X, Y, Xprime, Yprime)
show_results('linear', linmodel, linX, linY, linXprime, linYprime)
#%%
# if __name__ == '__main__':
#     # https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce
#     manager = multiprocessing.Manager()
#     return_dict = manager.dict()
#     waitsec = 20
#     p = multiprocessing.Process(
#         target=solveCuttingStock, name="CuttingStock (non-linear)", args=(J, return_dict))
#     p.start()
#     time.sleep(waitsec)
#     p.terminate()
#     p2 = multiprocessing.Process(
#         target=solveCuttingStock_linear, name="CuttingStock (linear)", args=(J, 20, return_dict))
#     p2.start()
#     time.sleep(waitsec)
#     p2.terminate()
#     show_results(return_dict)


# %%
