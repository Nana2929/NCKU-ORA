"""
=================
HW4-3: DEA model
Dataset: NCKU departments
Description: Given 3 inputs, 3 outputs,
             determine the Efficiency with DEA (dual; input-oriented)
Author: Ching Wen Yang
=================

"""
#%%
import pandas as pd
import numpy as np
import gurobipy as gbp  # Python3.8.12
from typing import List, Tuple
filepath = '/Users/yangqingwen/Desktop/成大111-1課程/ORA/hw4/ORA_Assignment_04_DEA.xlsx'
rawdata = pd.read_excel(filepath)
rawdata.head()
rawdata = rawdata[rawdata.Department.isna() == False]
rawdata.reset_index(drop=True, inplace=True)


# map back the school
schools = ['Liberal Arts']*4
schools += ['Science']*6
schools += ['Engineering']*18
schools += ['Management']*5
schools += ['Medicine']*5
schools += ['Social Sciences']*3
rawdata['School'] = schools

# %%
# Preparing dataset
# DepCount: k
DepCount = len(rawdata.Department.unique())
InpNum = 3
OupNum = 3
X = np.zeros((DepCount, InpNum))
Y = np.zeros((DepCount, InpNum))
DecodeDepId = {}
for depid, depinfo in rawdata.iterrows():
    # filling in input data
    DecodeDepId[depid] = depinfo.Department
    X[depid, 0] = depinfo['Personnel']
    X[depid, 1] = depinfo['Expenses (unit:1000)']
    X[depid, 2] = depinfo['Space']
    # filling in output data
    Y[depid, 0] = depinfo['Teaching']
    Y[depid, 1] = depinfo['Publications']
    Y[depid, 2] = depinfo['Grants (unit:1000)']

#%%
# Building model
# We need a set of lambdak

def CRS_solve(r:int):
    """The DUAL formulation of CRS-DEA model
       illustrated in ORA_10_DEA_2022_ncku.pdf p.28
        This is the input-oriented version (weighing inputs)
        Measuring overall efficiency
    Args:
        r (int): which department among the k deps to be evaluated

    Returns:
        model: the optimized Gurobi model
        L: lambdak, the weights of the k departments
        thetar: the overall efficiency of the r-th department
                compare to: VRS evaluates the technical efficiency
    """
    model = gbp.Model('CRS-DEA(input-oriented)')
    L = [0 for _ in range(DepCount)]
    for i in range(DepCount):
        L[i] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'l{i}')
    thetar = model.addVar(lb=0, ub=1, vtype=gbp.GRB.CONTINUOUS, name=f'theta{r}')
    for i in range(InpNum):
        model.addConstr(gbp.quicksum(X[k, i] * L[k] for k in range(DepCount)) <= thetar*X[r,i], name=f'input_constr_{i}')
    for j in range(OupNum):
        model.addConstr(gbp.quicksum(Y[k, j] * L[k] for k in range(DepCount)) >= Y[r,j], name=f'output_constr_{j}')
    model.setObjective(thetar, gbp.GRB.MINIMIZE)
    model.optimize()
    return model, L, thetar

def VRS_solve(r:int):

    """DUAL formulation of VRS-DEA model
        illustrated in ORA_10_DEA_2022_ncku.pdf p.28
        This is the input-oriented version (weighing inputs)
        Measuring technical efficiency
    Args:
        r (int): department index among the k departments to be evaluated
    Returns:
        model: the optimized Gurobi model
        L: lambdak, the weights of the k departments
        thetar: the result; tech efficiency of the r-th department

    """
    model = gbp.Model('VRS-DEA(input-oriented)')
    L = [0 for _ in range(DepCount)]
    for i in range(DepCount):
        L[i] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'l{i}')
    thetar = model.addVar(lb=0, ub=1, vtype=gbp.GRB.CONTINUOUS, name=f'theta{r}')
    for i in range(InpNum):
        model.addConstr(gbp.quicksum(X[k, i] * L[k] for k in range(DepCount)) <= thetar*X[r,i], name=f'input_constr_{i}')
    for j in range(OupNum):
        model.addConstr(gbp.quicksum(Y[k, j] * L[k] for k in range(DepCount)) >= Y[r,j], name=f'output_constr_{j}')
    # == the "variable" returns to scale constraint ==
    # Convex combination of lambdas
    model.addConstr(gbp.quicksum(L[k] for k in range(DepCount)) == 1)

    model.setObjective(thetar, gbp.GRB.MINIMIZE)
    model.optimize()
    return model, L, thetar




# for each department, calculate its overallefficency
overalleff = []
techeff = []
for r, depinfo in rawdata.iterrows():
    crs_model, cL, cthetar = CRS_solve(r)
    print(f'Overall efficiency of {r}:{DecodeDepId[r]}: {cthetar.x}',
    file=open('./hw4/hw4-3_OE.txt', 'a'))
    vrs_model, vL, vthetar = VRS_solve(r)

    overalleff.append(cthetar.x)
    techeff.append(vthetar.x)


# %%
rawdata['overall_eff'] = overalleff
rawdata['tech_eff'] = techeff
rawdata['scale_eff'] = rawdata['overall_eff'] / rawdata['tech_eff']


rawdata.to_csv('./hw4/hw4-3_efficiency.csv', index=False)





# %%
rawdata['overalleff'].describe()
rawdata['overalleff'].plot(kind='hist', bins=20)
# %%
rawdata[rawdata['overalleff'] == 1] # 13 departments
# Foreign LanguagesMathematics Environmental Electrical Eng.
# %%
# 附上表格
print(rawdata.groupby(['School'])['scale_eff'].mean().sort_values(ascending=False))
# %%
