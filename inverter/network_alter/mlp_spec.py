import torch
import torch.nn as nn
from torchinfo import summary


class MLP_RAM(nn.Module):

    def __init__(self, input_size=1, num_classes=2):
        super().__init__()
        self.l1 = nn.Linear(input_size,num_classes)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = torch.sigmoid(x)*1.2-0.1
        return x

class MLP_linear_sample(nn.Module):

    def __init__(self, input_size=1, num_classes=2):
        super().__init__()
        self.l1 = nn.Linear(input_size,num_classes)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = torch.sigmoid(x)
        return x


class MLP_RAM_2(nn.Module):

    def __init__(self, input_size=1, num_classes=2):
        super().__init__()
        self.l1 = nn.Linear(input_size,10)
        self.l2 = nn.Linear(10, 20)
        self.l3 = nn.Linear(20, 10)
        self.l4 = nn.Linear(10, num_classes)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x=self.elu(x)
        x = self.l2(x)
        x=self.elu(x)
        x = self.l3(x)
        x=self.elu(x)
        x = self.l4(x)
        x = torch.sigmoid(x)*1.2-0.1
        return x

class RAM_mlp_special_arc(nn.Module):

    def __init__(self, input_size=2, middle_layer_size=256, num_classes=1):
        super().__init__()
        self.l1 = nn.Linear(input_size, middle_layer_size)
        self.l2 = nn.Linear(middle_layer_size, middle_layer_size*2)
        self.l3 = nn.Linear(middle_layer_size*2, middle_layer_size)
        self.base_layer = nn.Linear(middle_layer_size, 1)
        self.constraint_layer = nn.Linear(middle_layer_size, num_classes)
        self.elu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x=self.elu(x)
        x = self.l2(x)
        x=self.elu(x)
        x = self.l3(x)
        x=self.elu(x)
        x_base = self.base_layer(x)
        x_base = torch.sigmoid(x_base)
        x_constraint = self.constraint_layer(x)
        x_output=torch.zeros_like(x_constraint)
        x_output[:,0]=x_base[:,0]
        for i in range(1, x_constraint.shape[1]):
            x_output[:, i] = x_output[:, 0] + torch.sigmoid(
                torch.sum(torch.abs(x_constraint[:, 0:i]), dim=1)) * (1 - x_output[:, 0])
        return x_output
class MLP_error_est(nn.Module):

    def __init__(self, input_size=2, middle_layer_size=1024, num_classes=1):
        super().__init__()
        self.l1 = nn.Linear(input_size, middle_layer_size)
        self.l2 = nn.Linear(middle_layer_size, middle_layer_size*2)
        self.l3 = nn.Linear(middle_layer_size*2, middle_layer_size)
        self.l4 = nn.Linear(middle_layer_size, num_classes)
        self.elu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x=self.elu(x)
        x = self.l2(x)
        x=self.elu(x)
        x = self.l3(x)
        x=self.elu(x)
        x = self.l4(x)
        x = torch.sigmoid(x)*2.
        return x

class MLP_error_est_2(nn.Module):

    def __init__(self, input_size=2, middle_layer_size=512, num_classes=1):
        super().__init__()
        self.l1 = nn.Linear(input_size, middle_layer_size)
        self.l2 = nn.Linear(middle_layer_size, middle_layer_size*2)
        self.l3 = nn.Linear(middle_layer_size*2, middle_layer_size)
        self.l4 = nn.Linear(middle_layer_size, num_classes)
        self.elu = nn.ELU()
        self.relu= nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x=x=self.elu(x)
        x = self.l2(x)
        x=x=self.elu(x)
        x = self.l3(x)
        x=x=self.elu(x)
        x = self.l4(x)
        x = self.relu(x)
        return x
