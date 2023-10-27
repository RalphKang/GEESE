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
# class MLP_RAM(nn.Module):
#
#     def __init__(self, input_size=1, num_classes=2):
#         super().__init__()
#         self.l1 = nn.Linear(input_size,num_classes)
#         # self.l2 = nn.Linear(11, num_classes)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.l1(x)
#         # x = torch.sigmoid(x)
#         # x=self.l2(x)
#         x = torch.tanh(x)*2
#         # x[:,0]=0-x[:,0]
#         x=x+0.5
#         return x

# class MLP_error_est(nn.Module):
#
#     def __init__(self, input_size=2, middle_layer_size=15, num_classes=1):
#         super().__init__()
#         self.l1 = nn.Linear(input_size, middle_layer_size)
#         self.l4 = nn.Linear(middle_layer_size, num_classes)
#         self.elu = nn.ELU()
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.l1(x)
#         x=torch.sigmoid(x)
#         x = self.l4(x)
#         x = torch.sigmoid(x)
#         return x


# class MLP_error_est(nn.Module):
#
#     def __init__(self, input_size=2, middle_layer_size=15, num_classes=1):
#         super().__init__()
#         self.l1 = nn.Linear(input_size, middle_layer_size)
#         self.l2 = nn.Linear(middle_layer_size, middle_layer_size*4)
#         self.l3 = nn.Linear(middle_layer_size*4, middle_layer_size)
#         self.l4 = nn.Linear(middle_layer_size, num_classes)
#         self.elu = nn.ELU()
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.l1(x)
#         x=x=self.elu(x)
#         x = self.l2(x)
#         x=x=self.elu(x)
#         x = self.l3(x)
#         x=x=self.elu(x)
#         x = self.l4(x)
#         x = torch.sigmoid(x)
#         return x
class MLP_error_est(nn.Module):

    def __init__(self, input_size=2, middle_layer_size=512, num_classes=1):
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

class MLP_error_est_modact(nn.Module):

    def __init__(self, input_size=2, middle_layer_size=512, num_classes=1,num_constraint=1):
        super().__init__()
        self.l1 = nn.Linear(input_size, middle_layer_size)
        self.l2 = nn.Linear(middle_layer_size, middle_layer_size*2)
        self.l3 = nn.Linear(middle_layer_size*2, middle_layer_size)
        self.l4 = nn.Linear(middle_layer_size, num_classes)
        self.l_constraint = nn.Linear(middle_layer_size, num_constraint)
        self.elu = nn.ReLU()
        self.elu_real = nn.ELU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x=self.elu(x)
        x = self.l2(x)
        x=self.elu(x)
        x = self.l3(x)
        x=self.elu(x)
        x_object = self.l4(x)
        x_constraint = self.l_constraint(x)
        x_object = torch.sigmoid(x_object)*2.
        x_constraint = torch.tanh(x_constraint)
        x=torch.cat((x_object,x_constraint),1)
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
