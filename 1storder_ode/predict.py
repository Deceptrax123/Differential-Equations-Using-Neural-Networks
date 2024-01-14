import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DiffEquationModel


def plot_y():
    t = torch.tensor(np.linspace(0, 5, 200000)[:, None], dtype=torch.float32)
    t = t.view(t.size(0), 1)

    model = DiffEquationModel()
    model.load_state_dict(torch.load("1storder_ode/weights.pth"))

    A0 = 0
    y_t = A0+t*model(t)
    analytic = torch.exp(-t/5)*torch.sin(t)

    plt.plot(t.detach().numpy(), y_t.detach().numpy(), 'orange',
             label="Neural")
    plt.plot(t.detach().numpy(), analytic, 'g--', label="Analytic")

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Neural and Analytic Solution")
    plt.legend()
    plt.show()


plot_y()
