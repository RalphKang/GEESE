import numpy as np

"""
% x -----> ps X D where 'ps': number of population and 'D': Dimension of
% the problem.
% f -----> Objective Function Value.
% g -----> Inequality Consstraints Value; ps X ng where 'ng': number of
% inequality constraints.
% h -----> Equality Constraints Value; ps X nh where 'nh': number of
% equality constraints.
% prob_k -> Index of problem."""


# xmin35 = -0 * np.ones((1, 30))
# xmax35 = +90 * np.ones((1, 30))
# x = np.ones((3, 30)) * 45.

def Error_function(actual_observation, reconstructed_observation, state_norm, dif_coef):
    reconstructed_observation_norm = reconstructed_observation / actual_observation
    error_est_perf = np.log(np.abs(reconstructed_observation_norm - np.ones_like(reconstructed_observation_norm)) + 1)
    state_norm_clip = np.clip(state_norm, 0, 1)
    state_boundary_loss = np.maximum(state_norm - 1, 0) + np.maximum(-state_norm, 0)
    error_est_constraint = np.maximum(state_norm_clip[0:29] - state_norm_clip[1:30] + 1e-6, 0)
    # error_sum = np.mean(np.concatenate((error_est_perf, error_est_constraint))) + np.mean(state_boundary_loss)
    error_sum = 0.5*np.mean(error_est_perf) + 10 * np.mean(error_est_constraint) + 0.1 * np.mean(state_boundary_loss)

    return error_est_perf , error_sum * dif_coef


class invert_forward_model:
    def __init__(self, ):

        self.m = 0.32
        self.s = np.array(
            [1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1])
        self.k = np.array(
            [5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83,
             85, 91,
             95, 97])

    def generate_sample(self, state):
        # %% objective function
        observation = np.zeros(2)
        su = 0
        for j in range(0, 30):
            su2 = 0
            for l in range(0, 30):
                su2 = su2 + self.s[l] * np.cos(self.k[j] * state[l] * np.pi / 180)
            su = su + su2 ** 2 / self.k[j] ** 4
        observation[0] = su ** 0.5 / (np.sum(1. / self.k ** 4)) ** 0.5
        # %%
        observation[1] = (np.sum(self.s * np.cos(state * np.pi / 180)) - self.m) ** 2
        return observation
        # %% constraints
        # g = np.zeros((ps, D - 1))
        # for i in range(0, D - 1):
        #     g[:, i] = x[:, i] - x[:, i + 1] + 1e-6


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
    rebuilt_observation = forward_physical_model.generate_sample(state_est)  # state estimation-->rebuilt observation
    error_group, error_sum = Error_function(actual_observation=data_norm_store.orig_observation,
                                            reconstructed_observation=rebuilt_observation,
                                            state_norm=state_norm,
                                            dif_coef=data_norm_store.error_zoomer)
    return rebuilt_observation, state_est, error_group, error_sum
