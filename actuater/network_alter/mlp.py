import torch
import torch.nn as nn
from torchsummary import summary
class MLP(nn.Module):

    def __init__(self, input_size=201,middle_layer_size=15, num_classes=1):
        super().__init__()
        self.l1=nn.Linear(input_size,middle_layer_size)
        self.l2=nn.Linear(middle_layer_size,num_classes)
    def forward(self,x):
        x=x.view(x.size(0), -1)
        x=self.l1(x)
        x=torch.sigmoid(x)
        x=self.l2(x)
        x=torch.sigmoid(x)
        return x
# net=MLP(middle_layer_size=15)
# # net.to("cuda")
# summary(net,batch_size=1, input_size=(1,201),device='cpu')