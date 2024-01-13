import torch 
from torch import nn 
import numpy as np 
from model import DiffEquationModel
from initialize import init_weights
from torch.autograd import grad
import matplotlib.pyplot as plt 

def analytic_derivative(t,y):
    return y+(-(1/2)*((torch.exp(t/2)*torch.sin(5*t))))+(5*torch.exp(t/2)*torch.cos(5*t))

def network_derivative(t):
    t.requires_grad=True
    z=model(t).sum()
    
    grad_f=grad(z,t)[0]

    return grad_f

def nn_output(t):
    A=0
    return t*model(t)+A

def train():
    #Set initial conditions
    t=torch.tensor(np.linspace(0,5,200)[:,None],dtype=torch.float32)
    t=t.view(t.size(0),1)

    model.train(True)
    for epoch in range(epochs):        
        network_deri=network_derivative(t)*t+model(t)
        analytic_deri=analytic_derivative(t,nn_output(t))

        optimizer.zero_grad()
        loss=objective(network_deri,analytic_deri)
        
        loss.backward()
        optimizer.step()

        print("Epoch: ",epoch)
        print("Loss: ",loss.item())
    
    torch.save(model.state_dict(),"neural_network/weights.pth")

if __name__=='__main__':
    
    #Get the model
    model=DiffEquationModel()

    init_weights(model)
    optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001,betas=(0.9,0.999))

    #Losses
    objective=nn.MSELoss()

    #Hyperparameters
    epochs=5000000

    train()