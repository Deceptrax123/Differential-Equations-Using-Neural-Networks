import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DiffEquationModel


def plot_y():
    t = torch.tensor(np.linspace(0, 5, 200000)[:, None], dtype=torch.float32)
    t = t.view(t.size(0), 1)

    model = DiffEquationModel()
    model.load_state_dict(torch.load("1storder_ode/weights.pth"))

    y_t = t*model(t)

    plt.plot(t.detach().numpy(), y_t.detach().numpy())
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Solution using Neural Network")
    plt.show()


plot_y()
