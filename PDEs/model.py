import torch
from torch import nn
from torch.nn import Module, Linear


class PDEModel(Module):
    def __init__(self):
        super(PDEModel, self).__init__()

        self.lin1 = Linear(in_features=2, out_features=10)
        self.lin2 = Linear(in_features=10, out_features=1)

    def forward(self, x, y):

        xy = torch.cat((x, y), dim=1)
        xy = self.lin1(xy).sigmoid()
        xy = self.lin2(xy)

        return xy
