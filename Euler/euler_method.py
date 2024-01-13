import torch 
import numpy as np 
import matplotlib.pyplot as plt 

def plot_y(t,y):
    plt.plot(t,y)
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Solution of Differential equation")
    plt.show()

def solve():
    #Define parameters
    h=0.00005 #Step size
    time_steps=5 #Discrete units of time

    #Linear Approximation
    #y_(t+1)=y_n+f_n(t_n+1-tn)

    steps=[0]
    y=[0]

    t0=torch.tensor(0,dtype=torch.float32)
    y0=torch.tensor(0,dtype=torch.float32)

    while t0<time_steps:
        #Compute dy/dt=f(t0,y0) at t=t0
        f0=y0+(-(1/2)*((torch.exp(t0/2)*torch.sin(5*t0))))+(5*torch.exp(t0/2)*torch.cos(5*t0))

        y1=y0+f0*h
        t1=t0+h 

        steps.append(t1.item())
        y.append(y1.item())

        #Update
        t0=t1
        y0=y1
    
    plot_y(steps,y)
    

if __name__=='__main__':
    solve()