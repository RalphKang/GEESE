import torch

from network_alter.mlp_spec import *
from torch.optim import AdamW
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def load_state_estimator(input_size=2,num_classes=20,pretrained=False):
    """
    This module is used to load the trained/untrained state estimator
    """
    state_estimator = MLP_RAM(input_size=input_size, num_classes=num_classes)
    state_estimator.to("cuda")
    if pretrained:
        model_save_dir = 'ml_model/MlP_RAM'
        state_estimator.load_state_dict(torch.load(model_save_dir))
    return state_estimator


def optimization_network_config(LR_RAM=1.5e-4, LR_error=1.5e-4, LR_sample=1.5e-4,state_size=20,error_group_size=2,batch_size=32,number_error_estimator=2):
    RAM_net = RAM_mlp_special_arc(input_size=2, num_classes=state_size)
    optimizer_RAM = AdamW(RAM_net.parameters(), lr=LR_RAM, weight_decay=0.)
    RAM_net.to("cuda")
    # according to the number of error estimator, we need to define the error estimator by the number of error estimator
    error_estimator = []
    parameter_group = []
    for i in range(number_error_estimator):
        error_estimator.append(MLP_error_est(input_size=state_size, num_classes=error_group_size).to("cuda"))
        parameter_group += list(error_estimator[i].parameters())
    error_estimator_optimizer = AdamW(parameter_group, lr=LR_error, weight_decay=0.)
    mse_loss = nn.MSELoss(reduction="mean")
    seed_for_RAM = torch.rand(batch_size, 2).to("cuda")

    return RAM_net, error_estimator, optimizer_RAM, error_estimator_optimizer, mse_loss, seed_for_RAM


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
def accumulate_kernel(seed_for_RAM):
    # seed_for_RAM=torch.clamp(seed_for_RAM,min=0.0,max=1.0)
    generated_state=torch.zeros_like(seed_for_RAM).to("cuda")
    generated_state[:, 0] = seed_for_RAM[:, 0]
    for i in range(1,generated_state.shape[1]):
        generated_state[:,i]=generated_state[:, 0]+torch.sigmoid(torch.sum(torch.abs(seed_for_RAM[:,0:i]),dim=1))*(1-generated_state[:, 0] )
    return generated_state

def RAM_net_training_inference(RAM_net,RAM_net_optimizer, seed_for_RAM,error_estimator,error_estimator_number,error_zoomer,
                               error_threshold,RAM_delay):
    valid_flag = 0
    mini_error_record=[]
    mini_state_record=[]
    RAM_net.train()
    for update_times in range(RAM_delay):
        current_state = RAM_net(seed_for_RAM)
        current_state_std= torch.std(current_state[:,0])
        std_regularization=F.relu(0.288/10.0-current_state_std)
        estimated_implicit_error=[]
        for index in range(error_estimator_number):
            estimated_implicit_error.append(error_estimator[index](current_state))
        estimated_error = torch.mean(torch.mean(torch.stack(estimated_implicit_error),dim=0),dim=1)
        # estimated_error = torch.mean(torch.mean(torch.stack(estimated_implicit_error), dim=0), dim=1)
        current_state_clip = torch.clamp(current_state, min=0.0, max=1.0)
        loss_constraint=torch.mean(torch.relu(current_state_clip[:,0:29]-current_state_clip[:,1:30]+ 1e-6),dim=1)
        loss_boundary = torch.mean(F.relu(current_state - torch.ones_like(current_state)) + \
                                   F.relu(torch.zeros_like(current_state) - current_state), dim=1)
        sum_error=(0.45*estimated_error + 0.45*loss_constraint+ 0.1*loss_boundary+std_regularization)* error_zoomer
        smallest_error = torch.min(sum_error)
        need_state = current_state[torch.where(sum_error == smallest_error)][0]
        loss_state = torch.mean(sum_error)
        # if smallest_error.cpu().detach().numpy() < error_threshold:
        #     break
        RAM_net_optimizer.zero_grad()
        loss_state.backward()
        RAM_net_optimizer.step()
        mini_error_record.append(smallest_error.cpu().detach().numpy())
        mini_state_record.append(need_state.cpu().detach().numpy())
    RAM_net.eval()
    mini_error_record=np.array(mini_error_record)
    mini_state_record=np.array(mini_state_record)
    smallest_error_np = np.min(mini_error_record)
    smallest_state_np = mini_state_record[np.argmin(mini_error_record)]
    if smallest_error_np < error_threshold*5.0:
        valid_flag=1.0
    return loss_state.cpu().detach().numpy(),smallest_error_np, smallest_state_np,valid_flag

def sample_inference(error_estimator,error_estimator_number,number_class,seed_number):
    """
       used to train and sample the sample net
    :param error_estimator:
    :param error_estimator_number:
    :param number_class:
    :param seed_number:
    :return:
    """
    sample_net = MLP_linear_sample(input_size=1, num_classes=number_class)
    sample_net.to("cuda")
    seed_for_sample = torch.rand(seed_number, 1).to("cuda")
    current_state = sample_net((seed_for_sample-0.5)*10.0)
    current_state=accumulate_kernel(current_state)
    estimated_implicit_error = []
    for index in range(error_estimator_number):
        estimated_implicit_error.append(error_estimator[index](current_state))
    std_error = -torch.std(torch.mean(torch.stack(estimated_implicit_error), dim=-1), dim=0)
    smallest_error = torch.min(std_error)
    need_state = current_state[torch.where(std_error == smallest_error)][0]
    return need_state.cpu().detach().numpy().squeeze()



