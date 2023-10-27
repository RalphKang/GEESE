import numpy as np
from matmos import ISA
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

"""
huracan is the open source gasturbine model
github: https://github.com/alopezrivera/huracan
"""
from huracan.components import inlet, fan, compressor, combustion_chamber, turbine, nozzle
from huracan.engine import shaft
from huracan.thermo.fluids import gas, fuel


def Error_function(actual_observation, reconstructed_observation, estimated_state, dif_coef):
    reconstructed_observation_norm = reconstructed_observation / actual_observation
    error_est = np.log(np.abs(reconstructed_observation_norm - np.ones_like(reconstructed_observation_norm)) + 1)

    estimated_state_level = np.squeeze(estimated_state)
    error_regularization_1 = np.maximum(estimated_state_level - np.ones_like(estimated_state_level),
                                      np.zeros_like(estimated_state_level)) + \
                           np.maximum(np.zeros_like(estimated_state_level) - estimated_state_level,
                                      np.zeros_like(estimated_state_level))
    error_regularization_2 = np.std(estimated_state_level)
    error_sum = np.mean(error_est) + np.mean(error_regularization_1) * 0.1+ error_regularization_2 * 0.1
    return error_est * dif_coef, error_sum * dif_coef


class gasturbine_forward_model:
    def __init__(self, h, mach, mf, eta_inlet, pi_burner, eta_nozzle):
        self.a = ISA(h)
        self.mach = mach
        self.t = self.a.t
        self.p = self.a.p
        self.mf = mf
        self.eta_in = eta_inlet
        self.pi_burner = pi_burner
        self.eta_nozzle = eta_nozzle

    def generate_sample(self, state):
        bpr = state[0]
        pi_fan = state[1]
        pi_c1 = state[2]
        pi_c2 = state[3]
        T_tb = state[4]
        eta_fan = state[5]
        eta_c1 = state[6]
        eta_c2 = state[7]
        eta_burner = state[8]
        eta_t1 = state[9]
        eta_t2 = state[10]

        # mf = self.mf*(bpr+1)
        mf = self.mf
        f = fuel(LHV=43e6)
        g = gas(mf=mf,
                cp=lambda T: 1150 if T > 1000 else 1000,
                k=lambda T: 1.33 if T > 1000 else 1.4,
                m=self.mach, t_0=self.t, p_0=self.p)

        i = inlet(PI=self.eta_in)
        fn = fan(eta=eta_fan, PI=pi_fan)
        c1 = compressor(eta=eta_c1, PI=pi_c1)
        # ic = intercooler       (eta=0.95,  Q_out=15e6)
        c2 = compressor(eta=eta_c2, PI=pi_c2)
        cc = combustion_chamber(fuel=f, eta=eta_burner, PI=self.pi_burner, t01=T_tb)
        t1 = turbine(eta=eta_t1)
        t2 = turbine(eta=eta_t2)
        # t3 = turbine           (eta=0.97)
        nc = nozzle(eta=self.eta_nozzle)
        nf = nozzle(eta=self.eta_nozzle)

        shaft1 = shaft(fn, c1, t2, eta=0.995)
        shaft2 = shaft(c2, t1, eta=0.995)
        stream = g - i - fn
        s1core, s1bypass = stream * (bpr / (bpr + 1))

        s1core - c1 - c2 - cc - t1 - t2 - nc
        s1bypass - nf
        stream.run(log=False)

        fuel_consumption = stream.fmf()
        thrust = stream.thrust_total() / 1000  # kN

        # stream.efficiency_total()
        TSFC = fuel_consumption / thrust * 1000  # g/KN/s
        if np.logical_or(np.isinf(thrust), np.isnan(thrust)):
            print("thrust cannot be calculated")
        if thrust < 0:
            print("thrust is negative")
        state_guess = [thrust, TSFC]
        return np.array(state_guess)


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
                                            estimated_state=state_norm, dif_coef=data_norm_store.error_zoomer)
    return rebuilt_observation, state_est, error_group, error_sum
