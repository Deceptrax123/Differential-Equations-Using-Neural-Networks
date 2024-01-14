import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Order2ODE


def plot_y():
    t = torch.tensor(np.linspace(0, 2, 100000)
                     [:, None], dtype=torch.float32)
    t = t.view(t.size(0), 1)

    model = Order2ODE()
    model.load_state_dict(torch.load("2ndorder_ode/weights.pth"))

    A0 = 0
    A1 = 1
    y_t = A0+A1*t+(t**2)*model(t)
    analytic_solution = torch.exp(-t/5)*torch.sin(t)

    plt.plot(t.detach().numpy(), y_t.detach().numpy(), label="Neural")
    plt.plot(t.detach().numpy(), analytic_solution, label="Analytic")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Analytic and Neural Solution")
    plt.legend()
    plt.show()


plot_y()
