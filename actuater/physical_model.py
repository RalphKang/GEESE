import numpy as np
import modact.problems as pb



def Error_function(actual_observation, reconstructed_observation, constraint_np, dif_coef):
    reconstructed_observation_norm = reconstructed_observation / actual_observation
    error_est_perf = np.log(np.abs(reconstructed_observation_norm - np.ones_like(reconstructed_observation_norm)) + 1)
    error_est_constraint = np.maximum(constraint_np, 0)
    error_est=np.concatenate((error_est_perf,error_est_constraint))
    error_sum = np.mean(np.concatenate((error_est_perf, error_est_constraint)))
    return error_est * dif_coef, error_sum * dif_coef


class modact_forward_model:
    def __init__(self, function):
        self.fct = pb.get_problem(function)
        self.lb, self.ub = self.fct.bounds()
        self.feasible_domain=np.array([self.lb,self.ub])

    def generate_sample(self, state):
        performance, constraint = self.fct(state)
        performance_np = np.array(performance)
        constraint_np = 0-np.asarray(constraint)  # need to minimize
        return performance_np, constraint_np


def physics_evaluation_module(state_norm, data_norm_store, forward_physical_model):
    """
    this function is used to calculate error according to the given state
    ---------------------------------------------------------------------
    INPUT:
    state_norm: normalized state (to feasible domain)
    data_norm_store: the normalization_data_store class in data_operation.py, which contains all necessary and basic information
        of the problem
    forward_physical_model: the forward physical model created here
    ---------------------------------------------------------------
    OUTPUT:
    as name shows, no need further explain
    """
    state_norm_clip = np.clip(state_norm, 0, 1)
    state_est = data_norm_store.state_unnorm_feasible(state_norm_clip)  # state estimation: normed-->un-normed
    rebuilt_observation, rebuild_constraint = forward_physical_model.generate_sample(state_est)  # state estimation-->rebuilt observation
    error_group, error_sum = Error_function(actual_observation=data_norm_store.orig_observation,
                                            reconstructed_observation=rebuilt_observation,
                                            constraint_np=rebuild_constraint,
                                            dif_coef=data_norm_store.error_zoomer)
    return rebuilt_observation, state_est, error_group, error_sum
