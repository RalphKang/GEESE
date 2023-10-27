import torch

from network_alter.mlp_spec import *
from torch.optim import AdamW
import torch.nn.functional as F
from torch.autograd import Variable

def load_state_estimator(pretrained=False):
    """
    This module is used to load the trained/untrained state estimator
    """
    state_estimator = MLP_RAM(input_size=2, num_classes=11)
    state_estimator.to("cuda")
    if pretrained:
        model_save_dir = 'ml_model/MlP_RAM'
        state_estimator.load_state_dict(torch.load(model_save_dir))
    return state_estimator


def optimization_network_config(LR_RAM=1.5e-4, LR_error=1.5e-4, LR_sample=1.5e-4,error_group_size=2,batch_size=32,number_error_estimator=2):
    seed_state = Variable(torch.rand(batch_size, 11).to('cuda'), requires_grad=True)
    optimizer_RAM = AdamW([seed_state], lr=LR_RAM, weight_decay=0.)
    # according to the number of error estimator, we need to define the error estimator by the number of error estimator
    error_estimator = []
    parameter_group=[]
    for i in range(number_error_estimator):
        error_estimator.append(MLP_error_est(input_size=11, num_classes=error_group_size).to("cuda"))
        parameter_group+=list(error_estimator[i].parameters())
    error_estimator_optimizer = AdamW(parameter_group, lr=LR_error, weight_decay=0.)
    mse_loss = nn.MSELoss(reduction="mean")

    return error_estimator, optimizer_RAM, error_estimator_optimizer, mse_loss, seed_state


def online_update_2(error_net_1, error_net_2, RAM_net, seed_for_RAM, state_set, error_set, error_net_optimizer,
                    error_net_2_optimizer, RAM_net_optimizer, loss_error_net, delay):
    error_net_1.train()
    error_net_2.train()
    RAM_net.train()

    state_set = state_set.to("cuda")
    error_set = error_set.to("cuda")
    # norm_label_pred = RAM_net(test_sample)
    for i in range(delay):
        error_net_pred_1 = error_net_1(state_set)
        loss_diff = loss_error_net(error_net_pred_1, error_set)
        if loss_diff > 1e-4:
            error_net_optimizer.zero_grad()
            loss_diff.backward()
            error_net_optimizer.step()

        error_net_2_pred = error_net_2(state_set)
        loss_diff_2 = loss_error_net(error_net_2_pred, error_set)
        if loss_diff_2 > 1e-4:
            error_net_2_optimizer.zero_grad()
            loss_diff_2.backward()
            error_net_2_optimizer.step()
    current_state = RAM_net(seed_for_RAM)
    estimated_error = torch.mean(torch.maximum(error_net_1(current_state), error_net_2(current_state)),dim=1)
    loss_boundary = torch.mean(F.relu(current_state - torch.ones_like(current_state)) + \
                               F.relu(torch.zeros_like(current_state) - current_state),dim=1)
    loss_regularization_2=torch.std(current_state,dim=1)
    loss_state = torch.mean(estimated_error + loss_boundary * 0.1+loss_regularization_2*0.1)
    RAM_net_optimizer.zero_grad()
    loss_state.backward()
    RAM_net_optimizer.step()

    new_state = RAM_net(seed_for_RAM)
    new_est_phsical_error= torch.mean(torch.maximum(error_net_1(current_state), error_net_2(current_state)), dim=1)
    new_loss_boundary = torch.mean(F.relu(current_state - torch.ones_like(current_state)) + \
                               F.relu(torch.zeros_like(current_state) - current_state), dim=1)
    new_loss_regularization_2 = torch.std(current_state, dim=1)
    new_est_error = new_est_phsical_error + new_loss_boundary  * 0.1 + new_loss_regularization_2 * 0.1
    smallest_error = torch.min(new_est_error)
    need_state = new_state[torch.where(new_est_error == smallest_error)][0]
    return torch.max(loss_diff, loss_diff_2), loss_state, smallest_error.cpu().detach().numpy(), need_state.cpu().detach().numpy().squeeze()

def error_estimator_training(state_set, error_set,error_estimator,error_estimator_optimizer, loss_function, delay, estimator_index):
    error_estimator[estimator_index].train()
    state_set = state_set.to("cuda")
    error_set = error_set.to("cuda")
    # norm_label_pred = RAM_net(test_sample)
    for i in range(delay):
        error_net_pred = error_estimator[estimator_index](state_set)
        loss_diff = loss_function(error_net_pred, error_set)
        # if loss_diff < 1e-4:
        #     break
        error_estimator_optimizer.zero_grad()
        loss_diff.backward()
        error_estimator_optimizer.step()
    error_estimator[estimator_index].eval()
    return loss_diff.cpu().detach().numpy(),i

def RAM_net_training_inference( RAM_net_optimizer, seed_for_RAM,error_estimator,error_estimator_number,error_zoomer,
                               error_threshold,RAM_delay):
    valid_flag = 0
    for update_times in range(RAM_delay):
        current_state = seed_for_RAM
        estimated_implicit_error=[]
        for index in range(error_estimator_number):
            estimated_implicit_error.append(error_estimator[index](current_state))
        # estimated_error = torch.mean(torch.mean(torch.stack(estimated_implicit_error),dim=0),dim=1)
        estimated_error = torch.mean(torch.mean(torch.stack(estimated_implicit_error), dim=0), dim=1)
        loss_boundary = torch.mean(F.relu(current_state - torch.ones_like(current_state)) + \
                                   F.relu(torch.zeros_like(current_state) - current_state), dim=1)
        loss_regularization_2 = torch.std(current_state, dim=1)
        sum_error=estimated_error + (loss_boundary * 0.1 + loss_regularization_2 * 0.1) * error_zoomer
        smallest_error = torch.min(sum_error)
        need_state = current_state[torch.where(sum_error == smallest_error)][0]
        loss_state = torch.mean(sum_error)
        # if smallest_error.cpu().detach().numpy() < error_threshold:
        #     break
        RAM_net_optimizer.zero_grad()
        loss_state.backward()
        RAM_net_optimizer.step()
    smallest_error_np=smallest_error.cpu().detach().numpy()
    if smallest_error_np < error_threshold*1.5:
        valid_flag=1.0
    return loss_state.cpu().detach().numpy(),smallest_error_np, need_state.cpu().detach().numpy().squeeze(),valid_flag

def sample_inference(error_estimator,error_estimator_number):
    """
       used to train and sample the sample net
    :param error_estimator:
    :param error_estimator_number:
    :return:
    """
    sample_net = MLP_linear_sample(input_size=1, num_classes=11)
    sample_net.to("cuda")
    seed_for_sample = torch.rand(64, 1).to("cuda")
    current_state = sample_net(seed_for_sample*5.0)
    estimated_implicit_error = []
    for index in range(error_estimator_number):
        estimated_implicit_error.append(error_estimator[index](current_state))
    std_error = -torch.std(torch.mean(torch.stack(estimated_implicit_error), dim=-1), dim=0)
    smallest_error = torch.min(std_error)
    need_state = current_state[torch.where(std_error == smallest_error)][0]
    return need_state.cpu().detach().numpy().squeeze()



