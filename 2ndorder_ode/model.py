import torch
from torch.nn import Module, Linear


class Order2ODE(Module):
    def __init__(self):
        super(Order2ODE, self).__init__()

        self.linear1 = Linear(1, 20)
        self.linear2 = Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x).sigmoid()
        x = self.linear2(x)

        return x
