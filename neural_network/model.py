import torch 
from torch import nn 
from torch.nn import Module,Linear,Sigmoid


class DiffEquationModel(Module):
    def __init__(self):
        super(DiffEquationModel,self).__init__()

        self.lin1=Linear(in_features=1,out_features=10)
        self.lin2=Linear(in_features=10,out_features=1)

        self.sig1=Sigmoid()
        self.sig2=Sigmoid()

    def forward(self,x):
        x=self.lin1(x).sigmoid()
        x=self.lin2(x)

        return x