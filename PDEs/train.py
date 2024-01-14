import torch
from torch import nn
import numpy as np
from model import PDEModel
from torch.autograd import grad
import matplotlib.pyplot as plt

torch.manual_seed(32)


def dirchilet_boundary_func_derivative_x(x, y):
    x.requires_grad = True
    A = (((1-x)*y**3)+(x*(1+y**3)*torch.exp(torch.tensor(-1)))+((1-y)*x*(torch.exp(-x)-torch.exp(torch.tensor(-1))))
         + (y*((torch.exp(torch.tensor(-x))*(x+1))-(1-x-(2*x*torch.exp(torch.tensor(-1))))))).sum()

    for j in range(2):
        grad_f = grad(A, x, create_graph=True)[0]
        A = grad_f.sum()

    return grad_f


def dirchilet_boundary_func_derivative_y(x, y):
    y.requires_grad = True
    A = (((1-x)*y**3)+(x*(1+y**3)*torch.exp(torch.tensor(-1)))+((1-y)*x*(torch.exp(-x)-torch.exp(torch.tensor(-1))))
         + (y*((torch.exp(torch.tensor(-x))*(x+1))-(1-x-(2*x*torch.exp(torch.tensor(-1))))))).sum()

    for j in range(2):
        grad_f = grad(A, y, create_graph=True)[0]
        A = grad_f.sum()

    return grad_f


def network_derivative_y(x, y):
    y.requires_grad = True
    z = model(x, y).sum()

    grad_f = grad(z, y)[0]

    return grad_f


def network_derivative_y_order_2(x, y):
    y.requires_grad = True
    z = model(x, y).sum()

    for j in range(2):
        grad_f = grad(z, y, create_graph=True)[0]
        z = grad_f.sum()

    return grad_f


def network_derivative_x(x, y):
    x.requires_grad = True
    z = model(x, y).sum()

    grad_f = grad(z, x)[0]

    return grad_f


def network_derivative_x_order_2(x, y):
    x.requires_grad = True
    z = model(x, y).sum()

    for j in range(2):
        grad_f = grad(z, x, create_graph=True)[0]
        z = grad_f.sum()

    return grad_f


def f(x, y):
    # d^2w/dx^2+d^2w/dy^2=f(x,y)

    return torch.exp(-x)*((x-2)+(y**3)+(6*y))


def train():
    x = torch.tensor(np.linspace(0, 1, 100)[:, None], dtype=torch.float32)
    x = x.view(x.size(0), 1)

    y = torch.tensor(np.linspace(0, 1, 100)[:, None], dtype=torch.float32)
    y = y.view(y.size(0), 1)

    model.train(True)

    sl = 0
    for epoch in range(epochs):
        # Get all gradients
        network_deri_x = network_derivative_x(x, y)
        network_deri_x_order_2 = network_derivative_x_order_2(x, y)
        network_deri_y = network_derivative_y(x, y)
        network_deri_y_order_2 = network_derivative_y_order_2(x, y)
        dirchilet_x = dirchilet_boundary_func_derivative_x(x, y)
        dirchilet_y = dirchilet_boundary_func_derivative_y(x, y)

        laplacian_x = dirchilet_x-(y*(1-y))*(-2*model(x, y)+(2*(1-2*x)*network_deri_x) +
                                             (x*(1-x)*network_deri_x_order_2))

        laplacian_y = dirchilet_y-(x*(1-x))*(-2*model(x, y)+(2*(1-2*y)*network_deri_y) +
                                             (y*(1-y)*network_deri_y_order_2))

        # d^2w/dx^2+d^2w/dy^2
        laplacian = laplacian_x+laplacian_y

        # f(x,y)
        f_x_y = f(x, y)

        # Perform backpropagation
        optimizer.zero_grad()
        loss = objective(laplacian, f_x_y)

        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch)
        print("Loss: ", loss.item())

        torch.save(model.state_dict(), "PDEs/weights.pth")


if __name__ == '__main__':

    # Get the model
    model = PDEModel()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # Losses
    objective = nn.MSELoss()

    # Hyperparameters
    epochs = 50000

    train()
