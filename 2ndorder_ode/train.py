import torch
from torch import nn
import numpy as np
from model import Order2ODE
from torch.autograd import grad
from torch.autograd.functional import hessian


def y_trial(t):
    A0 = 0
    A1 = 1
    return A0+A1*t+(t**2)*model(t)


def analytic_derivative(t, y):
    A1 = 1
    return (-1/5*torch.exp(-t/5)*torch.cos(t))-(1/5*(A1+2*t*network_derivative(t) +
                                                (t**2)*network_derivative(t)))-y


def network_derivative(t):
    t.requires_grad = True
    z = model(t).sum()

    grad_f = grad(z, t)[0]

    return grad_f


def network_derivative_order_2(t):
    t.requires_grad = True
    z = model(t).sum()

    for j in range(2):
        grad_f = grad(z, t, create_graph=True)[0]
        z = grad_f.sum()

    return grad_f


def train():
    t = torch.tensor(np.linspace(0, 2, 100)
                     [:, None], dtype=torch.float32)
    t = t.view(t.size(0), 1)

    sl = 0
    model.train(True)
    for epoch in range(epochs):
        network_der = 2*model(t)+4*t*network_derivative(t) + \
            (t**2)*network_derivative_order_2(t)
        analytic_der = analytic_derivative(t, y_trial(t))

        optimizer.zero_grad()
        loss = objective(network_der, analytic_der)

        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch)
        print("Loss: ", loss)

        if epoch == 0:
            sl = loss.item()
        elif loss.item() < sl:
            sl = loss.item()
            torch.save(model.state_dict(), "2ndorder_ode/weights.pth")


if __name__ == '__main__':

    # Get the model
    model = Order2ODE()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # Losses
    objective = nn.MSELoss()

    # Hyperparameters
    epochs = 500000

    train()
