import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Order2ODE


def plot_y():
    t = torch.tensor(np.linspace(0, 4, 100)
                     [:, None], dtype=torch.float32)
    t = t.view(t.size(0), 1)

    model = Order2ODE()
    model.load_state_dict(torch.load("2ndorder_ode/weights.pth"))

    A0 = 0
    A1 = 1
    y_t = A0+A1*t+(t**2)*model(t)

    plt.plot(t.detach().numpy(), y_t.detach().numpy())
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Solution using Neural Network")
    plt.show()


plot_y()
