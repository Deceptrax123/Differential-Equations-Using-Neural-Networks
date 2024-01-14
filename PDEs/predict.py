import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PDEModel
from mpl_toolkits import mplot3d


def A(x, y):
    return ((1-x)*y**3)+(x*(1+y**3)*torch.exp(torch.tensor(-1)))+((1-y)*x*(torch.exp(-x)-torch.exp(torch.tensor(-1))))\
        + (y*((torch.exp(torch.tensor(-x))*(x+1)) -
           (1-x-(2*x*torch.exp(torch.tensor(-1))))))


def plot_y():
    x = torch.tensor(np.linspace(0, 1, 100)
                     [:, None], dtype=torch.float32)
    x = x.view(x.size(0), 1)

    y = torch.tensor(np.linspace(0, 1, 100)
                     [:, None], dtype=torch.float32)
    y = y.view(y.size(0), 1)

    model = PDEModel()
    model.load_state_dict(torch.load("PDEs/weights.pth"))

    X, Y = np.meshgrid(x.detach().numpy(), y.detach().numpy())

    x_compu = torch.tensor(X[0, :].T).view(100, 1)
    y_compu = torch.tensor(Y[0, :].T).view(100, 1)

    psi = A(x_compu, y_compu)+(x_compu*(1-x_compu) *
                               y_compu*(1-y_compu)*model(x_compu, y_compu))
    analytic = torch.exp(-x_compu)*(x_compu+y_compu**3)

    Z, _ = np.meshgrid(psi.detach().numpy(), psi.detach().numpy())
    analytic_Z, _ = np.meshgrid(analytic.detach().numpy(),
                                analytic.detach().numpy())

    # Plotting
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8, label="Neural")
    ax.set_title("Neural Solution of PDE", fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.plot_surface(X, Y, analytic_Z, cmap='viridis', alpha=0.8)
    ax1.set_title("Analytic Solution of PDE", fontsize=14)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_zlabel("z", fontsize=12)
    plt.show()


plot_y()
