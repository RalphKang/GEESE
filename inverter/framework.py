import numpy as np

from data_operation import *
from network_management import load_state_estimator, optimization_network_config, error_estimator_training, \
    RAM_net_training_inference, sample_inference
from physical_model import invert_forward_model, physics_evaluation_module
import argparse


def main(index_test, ground_truth_state):
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iteration', type=int, default=1000, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--seed_size', type=int, default=256, help='number of seed size')
    parser.add_argument('--error_threshold', type=float, default=0.075, help='error threshold')
    parser.add_argument('--delay_update', type=int, default=40, help='delay')
    parser.add_argument('--error_zoomer', type=float, default=1.0, help='error zoomer')
    parser.add_argument('--error_estimator_number', type=int, default=4, help='number of error estimator')
    parser.add_argument('--feasible_domain_state', type=np.ndarray,
                        default=np.array([np.zeros(30),np.ones(30)*90]), help='feasible domain of state')
    parser.add_argument('--ground_truth_state', type=np.ndarray,
                        default=np.array([0]),
                        help='ground truth')
    parser.add_argument('--state_meaning', type=list,
                        default=["p1","p2","p3","p4",'p5',"p6","p7","p8","p9",'p10',
                                 "p11","p12","p13","p14","p15","p16","p17","p18","p19","p20",
                                 "p21","p22","p23","p24","p25","p26","p27","p28","p29","p30"],
                        help='state meaning')
    parser.add_argument('--record_path', type=str, default='output/best_state_recording.csv', help='record path')
    parser.add_argument('--observation_meaning', type=list, default=['cost', 'safety_factor'])
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()
    # args.ground_truth_state = np.loadtxt("modact/X_choose.txt")
    args.ground_truth_state = ground_truth_state
    optimization_recording = pandas_recording(
        column=args.state_meaning + args.observation_meaning + ['error'] + ['valid_flag'],
        save_dir='output/best_state_' + str(index_test) + '_recording.csv')
    state_size = len(args.ground_truth_state)
    # %% Obtain the test sample
    forward_physical_model = invert_forward_model()
    actual_observation= forward_physical_model.generate_sample(
        args.ground_truth_state)  # generate test observation
    data_norm_store = normalization_data_store(feasible4state=args.feasible_domain_state,
                                               observation=actual_observation,
                                               state_size=state_size,
                                               error_zoomer=args.error_zoomer)  # store all normalization boundaries, feasible boundaries

    """
    ===========================================================================================
                         1 pretrained model
    ===========================================================================================
    """
    state_estimator = load_state_estimator(input_size=len(args.observation_meaning),
                                           num_classes=state_size)  # load state estimator
    actual_observation4net = data_norm_store.obse_trans_phy2state_est(actual_observation)  # observation transform
    state_est_first_norm_tc = state_estimator(actual_observation4net.to(args.device))  # observation-->state estimation
    state_est_first_norm = state_est_first_norm_tc.detach().cpu().squeeze().numpy()  # state estimation: torch-->numpy
    observation_rebuilt_first, state_est_first, error_group_first, error_first = physics_evaluation_module(
        state_norm=state_est_first_norm,
        data_norm_store=data_norm_store,
        forward_physical_model=forward_physical_model)
    best_error = error_first
    best_state_norm = state_est_first_norm
    best_state = state_est_first
    best_observation = observation_rebuilt_first
    optimizaton_provide_mini = "pretrained model"
    i = 0
    valid_flag = 0
    optimization_recording_row_list = np.hstack(
        (best_state.flatten(), observation_rebuilt_first.flatten(), best_error, valid_flag)).tolist()
    optimization_recording.add_values(added_data=optimization_recording_row_list)
    if best_error < args.error_threshold:
        print("Optimization is done, best state: {}, best error:{}, the result is provided by {}".
              format(best_state, best_error, optimizaton_provide_mini))
    """
    ========================================================================================================================
                         2 OPTIMIZATION MODE
    ========================================================================================================================
    """
    if error_first > args.error_threshold:
        print("Inverse function mode does not work well, state estimation:{},error:{}".format(state_est_first,
                                                                                              error_first))

        """
        --------------- 2.1 Initialize networks, buffers, recordings and optimizer----------------------------------------------
        """

        RAM_net, error_estimator, state_optimizer, error_estimator_optimizer, mse_loss, seed_for_RAM = \
            optimization_network_config(LR_RAM=5.e-4, LR_error=1.e-4, LR_sample=1e-3, state_size=state_size,
                                        error_group_size=len(error_group_first),
                                        batch_size=args.seed_size,
                                        number_error_estimator=args.error_estimator_number)  # two physical errors
        random_walk_buffer = ReplayBuffer()
        in_situ_buffer = ReplayBuffer()
        gradient_recording = pandas_recording(
            column=args.state_meaning + args.observation_meaning + ['error', 'est_error', 'state_loss', 'error_loss'],
            save_dir='output/network_' + str(index_test) + '_recording.csv')
        random_recording = pandas_recording(column=args.state_meaning + args.observation_meaning + ['error'],
                                            save_dir='output/random_' + str(index_test) + '_recording.csv')
        valid_flag_round = 0
        query_number = 0
        for i in range(0, args.num_iteration):
            if i == 0:
                seed_random_state = np.random.rand(args.batch_size, state_size)
                RAM_error = np.inf
                for sample in range(0, args.batch_size):
                    random_observation, random_state, random_physcial_error, random_error = physics_evaluation_module(
                        state_norm=seed_random_state[sample, :],
                        data_norm_store=data_norm_store,
                        forward_physical_model=forward_physical_model)
                    random_walk_buffer.put(
                        (random_observation, seed_random_state[sample, :], random_physcial_error, random_error))

                    if random_error < RAM_error:
                        RAM_physical_error = random_physcial_error
                        # RAM_error = random_error
                        random_state_norm = est_state_norm = seed_random_state[sample, :]
                        est_state = random_state
                        rebuild_observation = random_observation
                    random_recording_row_list = np.hstack(
                        (random_state.flatten(), random_observation.flatten(), random_error)).tolist()
                    random_recording.add_values(random_recording_row_list)
                estimated_error = RAM_error

            else:
                if valid_flag > 0:
                    rebuild_observation, est_state, RAM_physical_error, RAM_error = physics_evaluation_module(
                        state_norm=est_state_norm,
                        data_norm_store=data_norm_store,
                        forward_physical_model=forward_physical_model)
                    in_situ_buffer.put((rebuild_observation, est_state_norm, RAM_physical_error, RAM_error))
                    query_number += 1
                random_state_norm = sample_inference(error_estimator=error_estimator,
                                                     error_estimator_number=args.error_estimator_number,
                                                     number_class=state_size,
                                                     seed_number=args.seed_size)

                random_observation, random_state, random_physcial_error, random_error = physics_evaluation_module(
                    state_norm=random_state_norm,
                    data_norm_store=data_norm_store,
                    forward_physical_model=forward_physical_model)
                random_walk_buffer.put((random_observation, random_state_norm, random_physcial_error, random_error))
                random_recording_row_list = np.hstack(
                    (random_state.flatten(), random_observation.flatten(), random_error)).tolist()
                random_recording.add_values(random_recording_row_list)
                query_number += 1
            # %%------------evaluation of the current state estimation----------------------------------------------
            if RAM_error < best_error:
                best_state_norm = est_state_norm
                best_state = data_norm_store.state_unnorm_feasible(best_state_norm)
                best_observation = rebuild_observation
                best_error = RAM_error
                optimizaton_provide_mini = "gradient_optimization"

            optimization_recording_row_list = np.hstack(
                (best_state.flatten(), best_observation.flatten(), best_error, valid_flag)).tolist()
            optimization_recording.add_values(optimization_recording_row_list)

            print(
                "current epoch: {},reconstructed_observation: {},actual_observation: {},current actual error: {}, current_pred_erro:{}, current best error: {}".
                    format(i, best_observation, actual_observation, RAM_error, estimated_error, best_error))
            if best_error < args.error_threshold or 2*i-valid_flag_round >= args.num_iteration-1:
                print("Optimization is done, best state: {}, best error:{}, the result is provided by {}".
                      format(best_state, best_error, optimizaton_provide_mini))
                break
            # %% update model parameters-----------------------------------------------------------------------------------
            # sample from buffer
            converge_model = 0
            for model_index in range(0, args.error_estimator_number):
                for epoch_round in range(0, 3):
                    loss_sum = 0
                    for batch_round in range(0, args.delay_update):
                        _, state_set, error_set = sample_from_buffer_v7(i=i, in_situ_memory=in_situ_buffer,
                                                                        global_search_memory=random_walk_buffer,
                                                                        current_norm_state=est_state_norm,
                                                                        current_observation=rebuild_observation,
                                                                        current_error=RAM_physical_error,
                                                                        sample_norm_state=random_state_norm,
                                                                        sample_observation=random_observation,
                                                                        sample_error=random_physcial_error,
                                                                        batch_size=args.batch_size, partition=0.3,
                                                                        valid_flag=valid_flag,
                                                                        valid_flag_round=valid_flag_round)
                        loss_error, maximum_epoch = error_estimator_training(state_set=state_set, error_set=error_set,
                                                                             error_estimator=error_estimator,
                                                                             error_estimator_optimizer=error_estimator_optimizer,
                                                                             loss_function=mse_loss,
                                                                             delay=1,
                                                                             estimator_index=model_index)
                        loss_sum = loss_sum + loss_error
                    average_loss = loss_sum / args.delay_update
                    if average_loss < 1e-3:
                        converge_model = converge_model + 1
                        break
            run_time = (round(converge_model / (args.error_estimator_number / 2)) + 1)*7
            print("the converge model is {}, the run time is {}".format(converge_model, run_time))
            loss_state, estimated_error, est_state_norm, valid_flag = RAM_net_training_inference(RAM_net=RAM_net,
                                                                                                 RAM_net_optimizer=state_optimizer,
                                                                                                 seed_for_RAM=seed_for_RAM,
                                                                                                 error_estimator=error_estimator,
                                                                                                 error_estimator_number=args.error_estimator_number,
                                                                                                 error_zoomer=args.error_zoomer,
                                                                                                 error_threshold=args.error_threshold,
                                                                                                 RAM_delay=run_time)

            gradient_recording_row_list = np.hstack((est_state.flatten(), rebuild_observation.flatten(), RAM_error,
                                                     estimated_error, loss_state, loss_error)).tolist()
            gradient_recording.add_values(gradient_recording_row_list)
            if valid_flag == 0:
                valid_flag_round += 1
            print("query number: {}" .format(query_number))
            if query_number >= args.num_iteration:
                print("The query number has reached the maximum number")
                break
if __name__ == '__main__':
    ground_truth = np.loadtxt("x_choose_inverter.txt")
    for index_test in range(1, 100):
        print("The {}th experiment is running".format(index_test))
        main(index_test, ground_truth[index_test])
