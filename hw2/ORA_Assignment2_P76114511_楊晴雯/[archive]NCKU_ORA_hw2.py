#%%
from gurobipy import *  # Python3.8.12
import numpy as np

# =========== variables ===========
Yields = {
    'low': [2, 2.4, 16],
    'medium': [2.5, 3, 20],
    'high': [3, 3.6, 24]
}
Prob = [1/3, 1/3, 1/3]
N = 3
GrowCost = [150, 230, 260]
CattleNeed = [200, 240, 0]
Sell = [170, 150]
Buy = [238, 210]
THRESH = 6000
BELOW, ABOVE = 36, 10


# ========= EV solution =========
def solve(Y):
    model = Model("Agriculture")
    X = [0 for _ in range(N)]
    Fx = [0 for _ in range(N)]

    for i in range(N):
        X[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'x{i+1}')
        Fx[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'fx{i+1}')
    # === constraints ===
    model.addConstr(quicksum(X[i] for i in range(N)) <= 500, name='total area')
    # cr. to 小傑老師
    # let Fx[i] be the profit function of each crop
    # Fx[i] = min(Fx[i]a, Fx[i]b) # piece-wise linear
    # then with linearization, when MAX Fx[i] term, let
    #   Fx[i] <= Fx[i]a
    #   Fx[i] <= Fx[i]b

    for i in range(N):
        if i < 2:
            model.addConstr(Fx[i] <= ((Y[i]*X[i] - CattleNeed[i])
                            * Sell[i] - GrowCost[i]*X[i]), name=f'f{i+1}a')
            model.addConstr(Fx[i] <= ((CattleNeed[i] - Y[i]*X[i])
                            * (-Buy[i]) - GrowCost[i]*X[i]), name=f'f{i+1}b')
        else:
            model.addConstr(Fx[i] <= (Y[i]*X[i]*BELOW -
                            GrowCost[i]*X[i]), name=f'f{i+1}a')
            model.addConstr(Fx[i] <= ((Y[i]*X[i] - THRESH)*ABOVE +
                            THRESH * BELOW - GrowCost[i]*X[i]), name=f'f{i+1}b')
    model.setObjective(quicksum(Fx[j] for j in range(N)), GRB.MAXIMIZE)
    model.optimize()

    for i in range(N):
        print(f'Crop Growing Acres: x{i+1}: {X[i].x}')
        print(f'Crop Profits: fx{i+1}: {Fx[i].x}')
    return model.objVal, [x.x for x in X], Fx


# ========== (b) ===========
#%%
# find EV solution
Y = [sum(v[i] for v in Yields.values())/3 for i in range(3)]
EVobj, EVX, EVFx = solve(Y)
print('===================')
print(EVX)
print(EVobj)

#%%

# find Wait-n-See solution
WSobj = 0
for i, (scenario, yields) in enumerate(Yields.items()):
    print(f'Scenario {scenario}:')
    WSobj += Prob[i] * solve(yields)[0]
print(f'WS objective: {WSobj}')


#%%

# ========== (c) RP ===========


def solve_stochastic(Yields):
    W = 3
    model = Model("Agriculture")
    X = [0 for _ in range(N)]
    Fx = [[0 for _ in range(M)] for _ in range(N)]
    stochastic_obj = 0
    for i in range(N):
        X[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'x{i+1}')
        for j in range(W):
            Fx[j][i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'fxw{i+1}')
    # === constraints ===
    model.addConstr(quicksum(X[i] for i in range(N)) <= 500, name='total area')

    for j, sc in enumerate(['low', 'medium', 'high']):
        # outer: scenario
        for i in range(N):
            # inner: crop
            if i < 2:
                model.addConstr(Fx[j][i] <= ((Yields[sc][i]*X[i] - CattleNeed[i])
                                * Sell[i] - GrowCost[i]*X[i]), name=f'f{sc},{i+1}a')
                model.addConstr(Fx[j][i] <= ((CattleNeed[i] - Yields[sc][i]*X[i])
                                * (-Buy[i]) - GrowCost[i]*X[i]), name=f'f{sc},{i+1}b')
            else:
                model.addConstr(Fx[j][i] <= (Yields[sc][i]*X[i]*BELOW -
                                GrowCost[i]*X[i]), name=f'f{sc},{i+1}a')
                model.addConstr(Fx[j][i] <= ((Yields[sc][i]*X[i] - THRESH)*ABOVE +
                                THRESH * BELOW - GrowCost[i]*X[i]), name=f'f{sc},{i+1}b')

        scProfits = quicksum(Fx[j][k] for k in range(N))
        stochastic_obj += Prob[i]*scProfits
    model.setObjective(stochastic_obj, GRB.MAXIMIZE)
    model.optimize()
    for i in range(N):
        print(f'Crop Growing Acres: x{i+1}: {X[i].x}')
        for j in range(W):
            print(f'Crop Profits: fx{j},{i+1}: {Fx[j][i].x}')
    return model.objVal, X, Fx


print('RP objective:')
RPobj, X, Fx = solve_stochastic(Yields)

#%%
# =========== (e) ===============
# For maximization problem:
# EVPI = RPObj - WSObj
# EEV = RPObj - getEEV()
scenarios = ['low', 'medium', 'high']
y1 = [Yields[w][0] for w in scenarios]
y2 = [Yields[w][1] for w in scenarios]
y3 = [Yields[w][2] for w in scenarios]


def getEEV(y1, y2, y3, x1, x2, x3):
    Fx = {}
    for w in range(3):
        # w=0: low
        # w=1: medium
        # w=2: high
        fx1 = min(170*(y1[w]-200) - 150 * x1, (-238)*(200-y1[w]*x1) - 150*x1)
        fx2 = min(150*(y2[w]*x2-240) - 230 * x2, -210*(240-y2[w]*x2) - 230*x2)
        fx3 = min(36*y3[w]*x3 - 260*x3, 10*(y3[w]*x3-6000) + 36*6000 - 260*x3)
        Fx[w] = fx1 + fx2 + fx3
    return Prob[w] * Fx[w]


getEEV(y1, y2, y3, *EVX)

# ============== (g) ====================
#%%


def SPN(
        Y,  # Y[scenario][crop]  # dimension = 30 * 3
        C: int = 3,
        W: int = 30):
    """ LOWER_BOUND
    SSA for continuoius stochastic programming
    Caclulate SPN 1 time.
    Averaging all omega's objective and then minimize (optimize)
    Args:
        C (int): crop size, here is fixed at 3
        W (int): scenario count, sample size, here is fixed at 30
    """
    model = Model("LB")
    SPNobj = 0
    X = [0 for _ in range(C)]
    Fx = [0 for _ in range(C)]  # F[scenraio][crop]
    model.addConstr(quicksum(X[i] for i in range(C)) <= 500, name='total area')
    for i in range(C):
        # for each scenario
        X[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'x{i+1}')
        Fx[w][i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'fx{i+1}')

    for w in range(W):
        # outer: scenario
        for i in range(C):
            # inner: crop
            if i < 2:
                model.addConstr(Fx[w][i] <= ((Y[w][i]*X[i] - CattleNeed[i])
                                * Sell[i] - GrowCost[i]*X[i]), name=f'f{w},{i+1}a')
                model.addConstr(Fx[w][i] <= ((CattleNeed[i] - Y[w][i]*X[i])
                                * (-Buy[i]) - GrowCost[i]*X[i]), name=f'f{i+1}b')
            else:
                model.addConstr(Fx[w][i] <= (Y[w][i]*X[i]*BELOW -
                                GrowCost[i]*X[i]), name=f'f{i+1}a')
                model.addConstr(Fx[w][i] <= ((Y[w][i]*X[i] - THRESH)*ABOVE +
                                THRESH * BELOW - GrowCost[i]*X[i]), name=f'f{i+1}b')
        obj_per_w = sum(Fx[w][k] for k in range(C))
        SPNobj += obj_per_w
    SPNobj /= W
    model.setObjective(SPNobj, GRB.MAXIMIZE)
    model.optimize()
    XNstar = [X[i].x for i in range(C)]
    return SPNobj, XNstar



def getY(N: int):
    """Get a set of y-rated yields for the 3 crops
    Args:
        N (int): sample size (to approximate a continuous distribution)
    """
    mu, sigma = 1, 0.1  # mean and standard deviation
    standard_Y = [2.5, 3, 20]
    C = len(standard_Y)
    Y = []
    for i in range(N):
        yield_rate = np.random.normal(mu, sigma, 1000)
        curr_y = [standard_Y[j] * yield_rate[j] for j in range(C)]
        Y.append(curr_y)
    return Y


def plugintoOBJ(Y, X):
    yw1, yw2, yw3 = Y
    x1, x2, x3 = X
    fx1 = min(170*(yw1*x1-200), (-238)*(200-yw1*x1)) - 150 * x1
    fx2 = min(150*(yw2*x2-240), -210*(240-yw2*x2)) - 230 * x2
    fx3 = min(36*yw3*x3, 10*(yw3*x3-6000) + 36*6000) - 260 * x3
    return fx1 + fx2 + fx3



# LOWER BOUND
# UPPER BOUND

import gurobipy as gp
# with gp.Env(empty=True) as env:
#     env.setParam('OutputFlag', 0)
#     env.start()
#     with gp.Model(env=env) as m:
M = 30
i = 0
zjns = []  # LB
ztns = []
while i < M:
    # sample from continuous scenario distribution
    # a discrete set of 30 scenarios (to approx.)
    sample_omegas = getY(30)
    try:
        i += 1
        # Averaging all omega's objective and then minimize (optimize)
        zjn, Xstar = SPN(sample_omegas)
        zjns.append(zjn)
        batch_ztnhats = []  # UB
        # for every scenario yw in sample omegas,
        # plug into the RP function with the optimized Xstar to get the batch's optimal
        ztn = np.mean([plugintoOBJ(yw, Xstar) for yw in sample_omegas])
        ztns.append(ztn)
    except:
        pass
# calculate LB confidence interval
Lmn, Lmn_var = np.mean(zjns), np.var(zjns)
Unt, Unt_var = np.mean(ztns), np.var(ztns)


# %%
