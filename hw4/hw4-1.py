
#%%
import gurobipy as gbp
from typing import List
import matplotlib.pyplot as plt

# 1. Try to find an efficient set of product mixes
def solve(ObjWeights:List[float]):
    model = gbp.Model('Efficient Set of Product Mixes')
    # shape (plant#, product production num#) = (3, 2)
    X = [[0 for _ in range(2)] for _ in range(3)]
    for i in range(3):
        for j in range(2):
            X[i][j] = model.addVar(vtype=gbp.GRB.INTEGER, lb = 0, name='X'+str(i)+str(j))
    model.addConstr(X[0][0] * 1  <= 4, 'plant 1 cacacity')
    model.addConstr(X[1][1] * 2 <= 12, 'plant 2 cacacity')
    model.addConstr(X[2][0] * 3 + X[2][1] * 2 <= 18, 'plant 3 capacity')
    model.addConstr(X[0][1] <= 0, 'plant 1: not able to produce product 2')
    model.addConstr(X[1][0] <= 0, 'plant 2: not able to produce product 1')
    # Objective 1: Profit for 2 products
    # Objective 2: Pollution Penalty for 2 products
    Objs = [3*gbp.quicksum(X[j][0] for j in range(3)) + 5*gbp.quicksum(X[j][1] for j in range(3)),
            - 2*gbp.quicksum(X[j][0] for j in range(3)) - 4*gbp.quicksum(X[j][1] for j in range(3))]
    model.setObjective(
        ObjWeights[0] * (3*gbp.quicksum(X[j][0] for j in range(3)) + 5*gbp.quicksum(X[j][1] for j in range(3)))
         + ObjWeights[1] * (- 2*gbp.quicksum(X[j][0] for j in range(3)) - 4*gbp.quicksum(X[j][1] for j in range(3))) ,
         gbp.GRB.MAXIMIZE)
    model.optimize()
    return model, X, model.objVal
#%%
"""
1(a) Graphical Solutions by the concept of dominance
Efficient set = Pareto Frontier (線段上的點互相簿dominate)
https://www.mathworks.com/help/optim/ug/generate-and-plot-a-pareto-front.html
- set many (n) weight vectors and solve the problem for n times
- find the optimal 2 objective and plot them, 1 axis is obj 1, the other is obj 2
"""
points = []
Xmap = {}
Pnum = 30
with open('hw4/hw4-1a.txt', 'w') as f:
    for w in range(Pnum):
        wratio = w/Pnum
        wprime = 1 - wratio
        model, X, _ = solve(ObjWeights=[wratio, wprime])
        ObjVal1 = 3*sum(X[j][0].x for j in range(3)) + 5*sum(X[j][1].x for j in range(3))
        ObjVal2  = 2*sum(X[j][0].x for j in range(3)) + 4*sum(X[j][1].x for j in range(3))
        print(f'===========================')
        print(f'{wratio} * {ObjVal1} - {wprime} * {ObjVal2} = {model.objVal}', file=f)
        print(f'{wratio * ObjVal1} - {wprime* ObjVal2} = {model.objVal}', file=f)

        for i in range(3):
            for j in range(2):
                print(f'\t Plant {j} produces {X[i][j].x} product {i}.', file=f)
        x = wratio * ObjVal1
        y = wprime * ObjVal2
        points.append((x, y))
plt.scatter([p[0] for p in points],
             [p[1] for p in points], c ="yellow",
            linewidths = 1,
            marker ="^",
            edgecolor ="red")

plt.xlabel("(Max) ObjVal1: Profit")
plt.ylabel("(Min) ObjVal2: Pollution Penalty")
plt.title("4-1(a) Pareto Frontier")




"""

1(b)
Weighting method
1. find all extreme points
2. find the 2 objective values for each extreme point
3. plot the (2/3 * obj1, 1/3 * obj2) points
"""

# %%
wpoints = [
    (0,0), (8, 2.67), (20, 8), (28, 10.67),
    (12,4), (20,10), (32, 12), (40, 14.67),
    (30, 12), (38, 14.67), (50, 20), (58, 20)
]
plt.scatter([p[0] for p in wpoints],[p[1] for p in wpoints], c ="green", linewidths = 1,
            marker ="^",
            edgecolor ="cyan")
plt.xlabel("(Max) ObjVal1: Profit (w:2/3")
plt.ylabel("(Min) ObjVal2: Pollution Penalty (w:1/3)")
# %%
