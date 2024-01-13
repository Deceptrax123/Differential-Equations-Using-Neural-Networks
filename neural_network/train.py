import torch
from torch import nn
import numpy as np
from model import DiffEquationModel
from initialize import init_weights
from torch.autograd import grad
import matplotlib.pyplot as plt

torch.manual_seed(32)


def analytic_derivative(t, y):
    return torch.exp(-t/5)*torch.cos(t)-y/5


def network_derivative(t):
    t.requires_grad = True
    z = model(t).sum()

    grad_f = grad(z, t)[0]

    return grad_f


def nn_output(t):
    A = 0
    return t*model(t)+A


def train():
    # Set initial conditions
    t = torch.tensor(np.linspace(0, 5, 100)[:, None], dtype=torch.float32)
    t = t.view(t.size(0), 1)

    sl = 0
    model.train(True)
    for epoch in range(epochs):
        network_deri = network_derivative(t)*t+model(t)
        analytic_deri = analytic_derivative(t, nn_output(t))

        optimizer.zero_grad()
        loss = objective(network_deri, analytic_deri)

        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch)
        print("Loss: ", loss.item())

        if epoch == 0:
            sl = loss.item()

        if loss.item() < sl:
            sl = loss.item()
            torch.save(model.state_dict(), "neural_network/weights.pth")


if __name__ == '__main__':

    # Get the model
    model = DiffEquationModel()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # Losses
    objective = nn.MSELoss()

    # Hyperparameters
    epochs = 50000

    train()
