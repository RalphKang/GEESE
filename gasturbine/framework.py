from data_operation import *
from network_management import load_state_estimator, optimization_network_config, error_estimator_training, \
    RAM_net_training_inference, sample_inference
from physical_model import gasturbine_forward_model, physics_evaluation_module
import argparse


def main(index_test):
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iteration', type=int, default=500, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--error_threshold', type=float, default=0.075, help='error threshold')
    parser.add_argument('--delay_update', type=int, default=40, help='delay')
    parser.add_argument('--error_zoomer', type=float, default=1.0, help='error zoomer')
    parser.add_argument('--error_estimator_number', type=int, default=4, help='number of error estimator')
    parser.add_argument('--feasible_domain_state', type=np.ndarray,
                        default=np.array([[5.0, 1.3, 1.2, 8.0, 1300., 0.85, 0.82, 0.84, 0.95, 0.86, 0.87],
                                          [6.0, 2.5, 2.0, 15.0, 1800., 0.95, 0.92, 0.94, 0.995, 0.96, 0.97]]),
                        help='feasible domain')
    parser.add_argument('--ground_truth_state', type=np.ndarray,
                        default=np.array([5.313, 1.636, 2.84, 9.0, 1624., 0.864, 0.87, 0.915, 0.85, 0.985, 0.985]),
                        help='ground truth')
    parser.add_argument('--state_meaning', type=list,
                        default=['BPR', 'PR_fan', 'PR_LC', 'PR_HC', 'T4', 'eta_fan', 'eta_LC', 'eta_HC', 'eta_B',
                                 'eta_HT',
                                 'eta_LT'], help='state meaning')
    parser.add_argument('--record_path', type=str, default='output/best_state_recording.csv', help='record path')
    parser.add_argument('--observation_meaning', type=list, default=['thrust', 'TSFC'])
    args = parser.parse_args()

    global_search_step = int(args.batch_size)  # the number of initial global search steps

    optimization_recording = pandas_recording(
        column=args.state_meaning + args.observation_meaning + ['error'] + ['valid_flag'],
        save_dir='output/best_state_' + str(index_test) + '_recording.csv')
    state_size = len(args.ground_truth_state)
    # %% Obtain the test sample
    forward_physical_model = gasturbine_forward_model(h=0., mach=0.0, mf=361., eta_inlet=0.98, pi_burner=0.99,
                                                      eta_nozzle=0.985)
    actual_observation = forward_physical_model.generate_sample(args.ground_truth_state)  # generate test observation
    data_norm_store = normalization_data_store(feasible4state=args.feasible_domain_state,
                                               observation=actual_observation,
                                               state_size=state_size,
                                               error_zoomer=args.error_zoomer)  # store all normalization boundaries, feasible boundaries

    """
    ===========================================================================================
                         1 pretrained model
    ===========================================================================================
    """
    state_estimator = load_state_estimator()  # load state estimator
    actual_observation4net = data_norm_store.obse_trans_phy2state_est(actual_observation)  # observation transform
    state_est_first_norm_tc = state_estimator(actual_observation4net.to('cuda'))  # observation-->state estimation
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

        error_estimator, state_optimizer, error_estimator_optimizer, mse_loss, seed_norm_state_tc = \
            optimization_network_config(LR_RAM=1.e-2, LR_error=1.e-4, LR_sample=1e-3, error_group_size=2,
                                        batch_size=64,
                                        number_error_estimator=args.error_estimator_number)  # two physical errors
        random_walk_buffer = ReplayBuffer()
        in_situ_buffer = ReplayBuffer()
        gradient_recording = pandas_recording(
            column=args.state_meaning + args.observation_meaning + ['error', 'est_error', 'state_loss', 'error_loss'],
            save_dir='output/network_' + str(index_test) + '_recording.csv')
        random_recording = pandas_recording(column=args.state_meaning + args.observation_meaning + ['error'],
                                            save_dir='output/random_recording.csv')
        valid_flag_round = 0
        for i in range(0, args.num_iteration):
            if i == 0:
                est_state_norm_batch = seed_norm_state_tc.detach().cpu().numpy()
                RAM_error = np.inf
                for sample in range(0, args.batch_size):
                    random_observation, random_state, random_physcial_error, random_error = physics_evaluation_module(
                        state_norm=est_state_norm_batch[sample, :],
                        data_norm_store=data_norm_store,
                        forward_physical_model=forward_physical_model)
                    random_walk_buffer.put(
                        (random_observation, est_state_norm_batch[sample, :], random_physcial_error, random_error))

                    if random_error < RAM_error:
                        RAM_physical_error = random_physcial_error
                        # RAM_error = random_error
                        random_state_norm = est_state_norm = est_state_norm_batch[sample, :]
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

                random_state_norm = sample_inference(error_estimator=error_estimator,
                                                     error_estimator_number=args.error_estimator_number)

                random_observation, random_state, random_physcial_error, random_error = physics_evaluation_module(
                    state_norm=random_state_norm,
                    data_norm_store=data_norm_store,
                    forward_physical_model=forward_physical_model)
                random_walk_buffer.put((random_observation, random_state_norm, random_physcial_error, random_error))
                random_recording_row_list = np.hstack(
                    (random_state.flatten(), random_observation.flatten(), random_error)).tolist()
                random_recording.add_values(random_recording_row_list)

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
                "current epoch: {},current estimation: {}, reconstructed_observation: {},actual_observation: {},current actual error: {}, current_pred_erro:{}, current best error: {}".
                    format(i, est_state, best_observation, actual_observation, RAM_error, estimated_error, best_error))
            if best_error < args.error_threshold or i == args.num_iteration - 1:
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
                    if average_loss < 1e-4:
                        converge_model = converge_model + 1
                        break
            run_time=round(converge_model/(args.error_estimator_number/2))+1
            print("converge_model: {}, run_time: {}".format(converge_model, run_time))
            loss_state, estimated_error, est_state_norm, valid_flag = RAM_net_training_inference(
                seed_for_RAM=seed_norm_state_tc,
                RAM_net_optimizer=state_optimizer,
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


if __name__ == '__main__':
    for index_test in range(0, 100):
        print("The {}th experiment is running".format(index_test))
        main(index_test)
