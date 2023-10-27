import numpy as np
from pymoo.core.problem import ElementwiseProblem

import modact.problems as pb


class PymopProblem(ElementwiseProblem):

    def __init__(self, function, **kwargs):
        
        if isinstance(function, pb.Problem):
            self.fct = function
        else:
            self.fct = pb.get_problem(function)
        lb, ub = self.fct.bounds()
        n_var = len(lb)
        n_obj = len(self.fct.weights)
        n_constr = len(self.fct.c_weights)
        xl = lb
        xu = ub

        self.weights = np.array(self.fct.weights)
        self.c_weights = np.array(self.fct.c_weights)
        # f, g = self.fct(xl)
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl,
                         xu=xu, elementwise_evaluation=True, type_var=np.double,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f, g = self.fct(x)
        out["F"] = np.sum(np.abs((np.array(f)-np.array([6.08e-01,1.117e+02]))/np.array([6.08e-01,-1.117e+02])))+np.maximum(np.array(g)*self.c_weights,0).sum()
        # out["G"] = np.array(g)*self.c_weights
