import numpy as np
# from Model_PROPOSED import Model_PROPOSED
from Glob_Vars import Glob_Vars
from Model_PROPOSED import Model_PROPOSED


def Objfun(Soln):
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln

        learnper = round(Glob_Vars.Data.shape[0] * 0.75)
        train_data = Glob_Vars.Data[learnper:, :]
        train_target = Glob_Vars.Target[learnper:,:]
        test_data = Glob_Vars.Data[:learnper, :]
        test_target = Glob_Vars.Target[:learnper, :]
        Eval = Model_PROPOSED(train_data, train_target, test_data, test_target, sol.astype('int'))
        Fitn[i] = 1 / (Eval[4])

    return Fitn

