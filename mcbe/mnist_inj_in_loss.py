import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import tqdm
import os
import mcbe
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras.datasets

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = np.array([X_train[i].flatten() for i in range(X_train.shape[0])])
X_test = np.array([X_test[i].flatten() for i in range(X_test.shape[0])])

#normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# size of the layers
l1 = 784*2
l2 = 128

class Model(nn.Module):
    def __init__(self, input_dim, l1):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, l1)
        self.layer2 = nn.Linear(l1, l2)
        self.layer3 = nn.Linear(l2, 10)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
    
class Maxbias_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, max_bias, bias):
        return 0.1*np.linalg.norm(np.max(np.array([bias - max_bias, np.zeros_like(bias)]),axis=0))
    



X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

EPOCHS  = 50
num_iter = 10
accuracys = []
accuracys_inj = []

for i in tqdm.trange(num_iter):

    model     = Model(X_train.shape[1],l1)
    model_inj = Model(X_train.shape[1],l1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer_inj = torch.optim.Adam(model_inj.parameters(), lr=0.01)
    loss_fn   = nn.CrossEntropyLoss()
    loss_fn_maxbias = Maxbias_loss()

    weights = []
    weights_norm = []
    norms = []
    biases = np.zeros([EPOCHS + 1,l1])

    weights_inj = []
    weights_norm_inj = []
    norms_inj = []
    biases_inj = np.zeros([EPOCHS + 1,l1])

    w1 = model.layer1.weight
    b1 = model.layer1.bias
    m1 = w1.shape[0]
    n1 = w1.shape[1]
    norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
    w1_norm = torch.div(w1,norm)
    w1_norm[w1_norm == np.inf] = 0

    w1_inj = model_inj.layer1.weight
    b1_inj = model_inj.layer1.bias
    m1_inj = w1_inj.shape[0]
    n1_inj = w1_inj.shape[1]
    norm_inj = w1_inj.pow(2).sum(keepdim=True,dim=1).sqrt()
    w1_norm_inj = torch.div(w1_inj,norm_inj)
    w1_norm_inj[w1_norm_inj == np.inf] = 0

        
    weights.append(w1.detach().numpy())
    weights_norm.append(w1_norm.detach().numpy())
    norms.append(norm.detach().numpy())
    biases[0,:] = b1.detach().numpy()

    weights_inj.append(w1_inj.detach().numpy())
    weights_norm_inj.append(w1_norm_inj.detach().numpy())
    norms_inj.append(norm_inj.detach().numpy())
    biases_inj[0,:] = b1_inj.detach().numpy()

    #epoch counter
    k = 1


    loss_list     = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    loss_list_inj     = np.zeros((EPOCHS,))
    accuracy_list_inj = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()

        if epoch%5 == 0:
            est_alpha = mcbe.dd_mcbe(W=np.array(weights)[-1],X_train = X_train, num_estimation_points=10000,dd_method="blowup")

        y_pred_inj = model_inj(X_train)
        loss_inj = loss_fn(y_pred_inj, y_train) + loss_fn_maxbias(max_bias = est_alpha , bias = b1.detach().numpy())
        loss_list_inj[epoch] = loss_inj.item()
        
        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer_inj.zero_grad()
        loss_inj.backward()
        optimizer_inj.step()
        
        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}, Accuracy {correct.mean()}")

            y_pred_inj = model_inj(X_test)
            correct_inj = (torch.argmax(y_pred_inj, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list_inj[epoch] = correct_inj.mean()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss inj {loss_inj.item()}, Accuracy inj {correct_inj.mean()}")
        
        
        
        w1 = model.layer1.weight
        b1 = model.layer1.bias

        w1_inj = model_inj.layer1.weight
        b1_inj = model_inj.layer1.bias

        norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
        w1_norm = torch.div(w1,norm)
        w1_norm[w1_norm == np.inf] = 0

        norm_inj = w1_inj.pow(2).sum(keepdim=True,dim=1).sqrt()
        w1_norm_inj = torch.div(w1_inj,norm_inj)
        w1_norm_inj[w1_norm_inj == np.inf] = 0
        
        weights.append(w1.detach().numpy())
        weights_norm.append(w1_norm.detach().numpy())
        norms.append(norm.detach().numpy())
        biases[k,:] = b1.detach().numpy()

        weights_inj.append(w1_inj.detach().numpy())
        weights_norm_inj.append(w1_norm_inj.detach().numpy())
        norms_inj.append(norm_inj.detach().numpy())
        biases_inj[k,:] = b1_inj.detach().numpy()
        
        k = k+1

    accuracys.append(accuracy_list)
    accuracys_inj.append(accuracy_list_inj)

#save data
np.save("accuracys_mnist_inj_in_loss.npy",accuracys)
np.save("accuracys_inj_mnist_inj_in_loss.npy",accuracys_inj)  