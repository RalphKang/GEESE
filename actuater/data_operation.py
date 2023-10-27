import collections
import pickle
import random

import numpy as np
import pandas as pd
import torch
import pandas as pd
from physical_model import Error_function


class normalization_data_store:
    def __init__(self, feasible4state, observation, state_size, error_zoomer):
        """
        this class is used to store all feasible boundaries, normalization boundaries and original observation,
        meanwhile, the given functions include all transformations needed for using these boundaries
        ---------------------------------------------------
        INPUT:
        norm_dir4state: the dir to store state normalization boundaries
        norm_dir4observation: the dir to store observation normalization boundaries
        feasible4state: the feasible domain for state observations
        observation: the original or actual observation
        """
        self.orig_observation = observation
        self.feasible_domain = feasible4state
        self.state_size = state_size
        self.error_zoomer = error_zoomer

    def obse_trans_phy2state_est(self, observation):
        """
        used to normalize spectrum obtained from physical model to the spectrum format needed for state estimator
        ---------------------------------------------------------
        INPUT:
        Observation: here is the original spectrum format from physical model: numpy[spectrum_dim], unnormalized
        self: in fact, used the the normalization data inside the self.
        ---------------------------------------------------------
        OUTPUT:
        data4net: the needed spectrum format for state estimator: torch[1,1,spectrum_dim], normalized, on CPU, not GPU
        """
        data_norm = observation / self.orig_observation
        data4net = torch.from_numpy(np.expand_dims(np.expand_dims(data_norm, axis=0), axis=0)).float()
        return data4net

    def state_unnorm_feasible(self, normed_state):
        """
        transform normalized state to unnormalized ones by feasible domain
        ------------------------------------
        INPUT:
        self: used the state feasible domain inside
        normed_state: the normalized state
        OUTPUT:
        un-normalized state
        """
        return normed_state * (self.feasible_domain[1] - self.feasible_domain[0]) + self.feasible_domain[0]

    def state_norm_feasible(self, est_state):
        """
        transform the absolute state to normalized ones by feasible domain
        ------------------------------------
        INPUT:
        self: used the state feasible domain inside
        est_state: the actual state values, numpy
        OUTPUT:
        normalized state, numpy
        """
        return (est_state - self.feasible_domain[0]) / (self.feasible_domain[1] - self.feasible_domain[0])

    def state_numpy2dict(self, est_state):
        """
                this function is used to transform state from numpy shape to dictionary type, as bayes optimization package
                needs the input to be dictionary
                ------------------------------------
                INPUT:
                self:none
                est_state: the state given by bayes optimization, numpy type
                OUTPUT:
                state: transformed to dictionary type
                """
        return dict(enumerate(est_state.flatten(), 1))


class ReplayBuffer:
    """
    This class is used to construct the replay buffer
    """
    def __init__(self, prestore=False, buffer_dir=None):
        """
        This class is used to construct the replay buffer
        ------------------------------------------------
        INPUT:
        prestore: this is the pointer to mention whether needs to load a already existed buffer.
        buffer_dir: type:str, the dir to read alreay existed buffer.
        """
        self.buffer = collections.deque(maxlen=10000)
        if prestore:
            self.buffer = pickle.load(open(buffer_dir, 'rb'))
            print("buffer load")

    def put(self, transition):
        """
            used to store the data feeded
            ------------------------------------------------
            INPUT:
            transition: the data needs to store into buffer. type: cell, order: observation,state and error
            """
        self.buffer.append(transition)

    def sample(self, n, seed=5, seed_needed=False):
        """
                 randomly take sample from the buffer
                 ------------------------------------------------
                 INPUT:
                 n: the number of the samples
                 seed: the seed for random sampling
                 seed_needed: the pointer, to mention whether a seed is needed to lead sampling.
                 ------------------------------------------------
                 OUTPUT:
                 three torch tensors, which respectively corresponds the observation list, state list and error list
                 """
        if seed_needed:
            random.seed(seed)
        mini_batch = random.sample(self.buffer, n)
        observation_lst, state_lst, error_lst = [], [], []

        for transition in mini_batch:
            observation, state, error,_ = transition
            observation_lst.append(observation)
            state_lst.append([state])
            error_lst.append([error])

        return torch.tensor(observation_lst, dtype=torch.float), torch.tensor(state_lst, dtype=torch.float), \
               torch.tensor(error_lst, dtype=torch.float)

    def save(self, buffer_name):
        """
        used to save the buffer
        --------------------------------------
        INPUT:
        buffer_name: the dir to save the buffer.
        """
        pickle.dump(self.buffer, open(buffer_name, 'wb'))
        print("buffer saved ")

    def size(self):
        """
        return the size of the buffer
        """
        return len(self.buffer)

class ReplayBuffer_2:
    """
    This class is used to construct the replay buffer
    """
    def __init__(self, lowest_ratio=0.3, highest_ratio=0.3,prestore=False, buffer_dir=None):
        """
        This class is used to construct the replay buffer
        ------------------------------------------------
        INPUT:
        prestore: this is the pointer to mention whether needs to load a already existed buffer.
        buffer_dir: type:str, the dir to read alreay existed buffer.
        """
        self.buffer = collections.deque(maxlen=10000)
        self.error_list=[]
        self.lowest_ratio=lowest_ratio
        self.highest_ratio=highest_ratio
        if prestore:
            self.buffer = pickle.load(open(buffer_dir, 'rb'))
            print("buffer load")

    def put(self, transition):
        """
            used to store the data feeded
            ------------------------------------------------
            INPUT:
            transition: the data needs to store into buffer. type: cell, order: observation,state and error
            """
        self.buffer.append(transition)
        self.error_list.append(transition[3])

    def sample(self, n,seed=5,  seed_needed=False):
        """
                 randomly take sample from the buffer
                 ------------------------------------------------
                 INPUT:
                 n: the number of the samples
                 seed: the seed for random sampling
                 seed_needed: the pointer, to mention whether a seed is needed to lead sampling.
                 ------------------------------------------------
                 OUTPUT:
                 three torch tensors, which respectively corresponds the observation list, state list and error list
                 """
        if seed_needed:
            random.seed(seed)
        # get the index of several lowest error
        observation_lst, state_lst, error_lst = [], [], []
        if int(self.lowest_ratio*n)>0 or int(self.highest_ratio*n)>0:
            index_order=np.argsort(self.error_list)
            lowest_index=index_order[:int(self.lowest_ratio*n)]
            highest_index=index_order[int(self.highest_ratio*n):]
            index=np.concatenate((lowest_index,highest_index))
            mini_batch_max_min = [self.buffer[i] for i in index]
            remained_number=n-int(self.lowest_ratio*n)-int(self.highest_ratio*n)
            mini_batch_random = random.sample(self.buffer, remained_number)
            mini_batch=mini_batch_max_min+mini_batch_random
            for transition in mini_batch:
                observation, state, physical_error,_ = transition
                observation_lst.append(observation)
                state_lst.append([state])
                error_lst.append([physical_error])
        else:
            mini_batch = random.sample(self.buffer, n)
            for transition in mini_batch:
                observation, state, physical_error,_ = transition
                observation_lst.append(observation)
                state_lst.append([state])
                error_lst.append([physical_error])

        return torch.tensor(observation_lst, dtype=torch.float), torch.tensor(state_lst, dtype=torch.float), \
               torch.tensor(error_lst, dtype=torch.float)

    def save(self, buffer_name):
        """
        used to save the buffer
        --------------------------------------
        INPUT:
        buffer_name: the dir to save the buffer.
        """
        pickle.dump(self.buffer, open(buffer_name, 'wb'))
        print("buffer saved ")

    def size(self):
        """
        return the size of the buffer
        """
        return len(self.buffer)


def sample_from_buffer_v5(i, in_situ_memory, global_search_memory,
                          current_norm_state, current_observation, current_error, batch_size, partition=0.5):
    """
    this function is used to sample and stack data from several buffers, in order to provide samples for network to use
    ---------------------------------------------------------------
    INPUT:
    i: i is the epoch, which is the indicator of the sample ratio from each buffer
    in-situ_memory: the buffer stores the noised historical data
    global_search_memory: the buffer stores the random search data
    current_norm_state, current_observation, current_error: the current prediction of the network
    batch_size: type: int; how many samples needed to construct a buffer
    partition: type: int; the maximum ratio of samples from history and in-situ buffer, the default value of which
                is 1/4 of batch size
    -----------------------------------------------------------------
    OUTPUT:
    three torch tensors for the use of network, which respectively corresponds the observation list, state list and error list
    """
    sample_limit = int(batch_size *partition)
    if i <= sample_limit:
        observation_insitu, state_insitu, error_insitu = in_situ_memory.sample(i)
        observation_search, state_search, error_search = global_search_memory.sample(batch_size - i)
    else:  # fetch half from each one
        observation_insitu, state_insitu, error_insitu = in_situ_memory.sample(sample_limit)
        observation_search, state_search, error_search = global_search_memory.sample(
            batch_size - sample_limit)
        # print("")
    observation_set = torch.cat((observation_insitu, observation_search), dim=0)
    state_set = torch.cat((state_insitu,state_search), dim=0)
    error_set = torch.squeeze(torch.cat((error_insitu, error_search), dim=0))
    # error_set = torch.cat((error_insitu, error_search), dim=0)
    # # add the guide from current action
    current_state = torch.tensor(current_norm_state, dtype=torch.float).view(1, 1, -1)
    current_observation = torch.tensor(current_observation, dtype=torch.float).view(1, -1)
    current_error = torch.tensor(current_error, dtype=torch.float).view(1, -1)
    #
    observation_set = torch.cat((observation_set, current_observation))
    state_set = torch.cat((state_set, current_state))
    error_set = torch.cat((error_set, current_error))
    return observation_set, state_set, error_set

def sample_from_buffer_v7(i, in_situ_memory, global_search_memory,
                          current_norm_state, current_observation, current_error,
                          sample_norm_state, sample_observation, sample_error, batch_size, partition=0.5,
                          valid_flag=False,valid_flag_round=0):
    """
    this function is used to sample and stack data from several buffers, in order to provide samples for network to use
    ---------------------------------------------------------------
    INPUT:
    i: i is the epoch, which is the indicator of the sample ratio from each buffer
    in-situ_memory: the buffer stores the noised historical data
    global_search_memory: the buffer stores the random search data
    current_norm_state, current_observation, current_error: the current prediction of the network
    batch_size: type: int; how many samples needed to construct a buffer
    partition: type: int; the maximum ratio of samples from history and in-situ buffer, the default value of which
                is 1/4 of batch size
    -----------------------------------------------------------------
    OUTPUT:
    three torch tensors for the use of network, which respectively corresponds the observation list, state list and error list
    """
    sample_limit = int(batch_size *partition)
    current_sample_number=i-valid_flag_round
    if current_sample_number<= sample_limit:
        observation_insitu, state_insitu, error_insitu = in_situ_memory.sample(current_sample_number)
        observation_search, state_search, error_search = global_search_memory.sample(batch_size - current_sample_number)
    else:  # fetch half from each one
        observation_insitu, state_insitu, error_insitu = in_situ_memory.sample(sample_limit)
        observation_search, state_search, error_search = global_search_memory.sample(
            batch_size - sample_limit)
        # print("")
    observation_set = torch.cat((observation_insitu, observation_search), dim=0)
    state_set = torch.cat((state_insitu,state_search), dim=0)
    error_set = torch.squeeze(torch.cat((error_insitu, error_search), dim=0))
    # error_set = torch.cat((error_insitu, error_search), dim=0)
    # # add the guide from current action
    sample_state= torch.tensor(sample_norm_state, dtype=torch.float).view(1, 1, -1)
    sample_observation = torch.tensor(sample_observation, dtype=torch.float).view(1, -1)
    sample_error = torch.tensor(sample_error, dtype=torch.float).view(1, -1)
    if valid_flag>0:
        current_state = torch.tensor(current_norm_state, dtype=torch.float).view(1, 1, -1)
        current_observation = torch.tensor(current_observation, dtype=torch.float).view(1, -1)
        current_error = torch.tensor(current_error, dtype=torch.float).view(1, -1)
        observation_set = torch.cat((observation_set, current_observation, sample_observation))
        state_set = torch.cat((state_set, current_state, sample_state))
        error_set = torch.cat((error_set, current_error, sample_error))
    else:
        observation_set = torch.cat((observation_set, sample_observation))
        state_set = torch.cat((state_set, sample_state))
        error_set = torch.cat((error_set, sample_error))
    return observation_set, state_set, error_set

class pandas_recording:
    """
    this class is used to store the important outputs of the data
    """
    def __init__(self, column: list, save_dir: str, load_data: bool = False):
        """
        INPUT:
        column: the columns needed to create pandas dataframe, which are the name of the data needs to record
        save_dir: where to store the data
        load_data: pointer, whether load the already existed dataframe
        """
        if load_data:
            save_dir.recording = pd.read_csv(save_dir, index_col=0)
        else:
            self.recording = pd.DataFrame(columns=column)
        self.save_dir = save_dir

    def add_values(self, added_data: list):
        """
        this function is used to add and save dataframe
        ------------------------------------------------
        INPUT:
        self: the dataframe
        added_data: the data needs to record
        """
        self.recording.loc[len(self.recording)] = added_data
        self.recording.to_csv(self.save_dir)
