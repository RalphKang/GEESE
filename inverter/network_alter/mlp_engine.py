import torch
import torch.nn as nn
from torchinfo import summary


class MLP_RAM(nn.Module):

    def __init__(self, input_size=2, num_classes=11):
        super().__init__()
        self.l1 = nn.Linear(input_size,num_classes*2)
        self.l2 = nn.Linear(num_classes*2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = torch.sigmoid(x)
        x=self.l2(x)
        x = torch.tanh(x)*2
        # x[:,0]=0-x[:,0]
        x=x+0.5
        return x

class MLP_error_est(nn.Module):

    def __init__(self, input_size=11, num_classes=1):
        super().__init__()
        self.l1 = nn.Linear(input_size, input_size*2)
        self.l2 = nn.Linear(input_size*2, input_size*4)
        self.l3 = nn.Linear(input_size*4, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = torch.sigmoid(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        x = torch.sigmoid(x) * 2
        return x
