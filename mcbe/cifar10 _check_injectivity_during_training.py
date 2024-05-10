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
from skimage.color import rgb2gray

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

#transform to grayscale
X_train = rgb2gray(X_train)
X_test = rgb2gray(X_test)

X_train = np.array([X_train[i].flatten() for i in range(X_train.shape[0])])
X_test = np.array([X_test[i].flatten() for i in range(X_test.shape[0])])

#normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

l2 = 256
l3 = 128

class Model(nn.Module):
    def __init__(self, input_dim, l1):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, l1)
        self.layer2 = nn.Linear(l1, l2)
        self.layer3 = nn.Linear(l2, l3)
        self.layer4 = nn.Linear(l3, 10)

        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.softmax(self.layer4(x), dim=1)
        return x
    
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

y_train = y_train.squeeze()
y_test = y_test.squeeze()

EPOCHS = 50
num_iter = 10
percent_inj2 = []
redundandency = 2

for i in tqdm.trange(num_iter):

    # size of the layers
    l1 = 1024*redundandency

    model     = Model(X_train.shape[1],l1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn   = nn.CrossEntropyLoss()

    weights = []
    weights_norm = []
    norms = []
    biases = np.zeros([EPOCHS + 1,l1])
    percent_inj_list = []

    w1 = model.layer1.weight
    b1 = model.layer1.bias
    m1 = w1.shape[0]
    n1 = w1.shape[1]
    norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
    w1_norm = torch.div(w1,norm)
    w1_norm[w1_norm == np.inf] = 0

        
    weights.append(w1.detach().numpy())
    weights_norm.append(w1_norm.detach().numpy())
    norms.append(norm.detach().numpy())
    biases[0,:] = b1.detach().numpy()

    #epoch counter
    k = 1


    loss_list     = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()
        
        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}, Accuracy {correct.mean()}")
        
        
        
        w1 = model.layer1.weight
        b1 = model.layer1.bias

        if epoch % 5 == 0:
            print("check inj")
            # check injectivity 
            percent_inj = mcbe.check_injectivity_naive(w1.detach().numpy(),b1.detach().numpy(),points=X_test[:1000,:],iter=X_test[:1000,:].shape[0])
            percent_inj_list.append(percent_inj)
            print("percent_inj: ", percent_inj)

        norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
        w1_norm = torch.div(w1,norm)
        w1_norm[w1_norm == np.inf] = 0
        
        weights.append(w1.detach().numpy())
        weights_norm.append(w1_norm.detach().numpy())
        norms.append(norm.detach().numpy())
        biases[k,:] = b1.detach().numpy()
        
        k = k+1

    percent_inj2.append(percent_inj_list)

mean_inj2 = np.mean(percent_inj2,axis=0)
std_inj2 = np.std(percent_inj2,axis=0)
np.save('mean_inj2_cifar.npy',mean_inj2)
np.save('std_inj2_cifar.npy',std_inj2)

percent_inj3 = []
redundandency = 3

for i in tqdm.trange(num_iter):

    # size of the layers
    l1 = 1024*redundandency

    model     = Model(X_train.shape[1],l1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn   = nn.CrossEntropyLoss()

    weights = []
    weights_norm = []
    norms = []
    biases = np.zeros([EPOCHS + 1,l1])
    percent_inj_list = []

    w1 = model.layer1.weight
    b1 = model.layer1.bias
    m1 = w1.shape[0]
    n1 = w1.shape[1]
    norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
    w1_norm = torch.div(w1,norm)
    w1_norm[w1_norm == np.inf] = 0

        
    weights.append(w1.detach().numpy())
    weights_norm.append(w1_norm.detach().numpy())
    norms.append(norm.detach().numpy())
    biases[0,:] = b1.detach().numpy()

    #epoch counter
    k = 1


    loss_list     = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()
        
        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}, Accuracy {correct.mean()}")
        
        
        
        w1 = model.layer1.weight
        b1 = model.layer1.bias

        if epoch % 5 == 0:
            print("check inj")
            # check injectivity 
            percent_inj = mcbe.check_injectivity_naive(w1.detach().numpy(),b1.detach().numpy(),points=X_test[:1000,:],iter=X_test[:1000,:].shape[0])
            percent_inj_list.append(percent_inj)
            print("percent_inj: ", percent_inj)

        norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
        w1_norm = torch.div(w1,norm)
        w1_norm[w1_norm == np.inf] = 0
        
        weights.append(w1.detach().numpy())
        weights_norm.append(w1_norm.detach().numpy())
        norms.append(norm.detach().numpy())
        biases[k,:] = b1.detach().numpy()
        
        k = k+1

    percent_inj3.append(percent_inj_list)

mean_inj3 = np.mean(percent_inj3,axis=0)
std_inj3 = np.std(percent_inj3,axis=0)
np.save('mean_inj3_cifar.npy',mean_inj3)
np.save('std_inj3_cifar.npy',std_inj3)

percent_inj9 = []
redundandency = 9

for i in tqdm.trange(num_iter):

    # size of the layers
    l1 = 1024*redundandency

    model     = Model(X_train.shape[1],l1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn   = nn.CrossEntropyLoss()

    weights = []
    weights_norm = []
    norms = []
    biases = np.zeros([EPOCHS + 1,l1])
    percent_inj_list = []

    w1 = model.layer1.weight
    b1 = model.layer1.bias
    m1 = w1.shape[0]
    n1 = w1.shape[1]
    norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
    w1_norm = torch.div(w1,norm)
    w1_norm[w1_norm == np.inf] = 0

        
    weights.append(w1.detach().numpy())
    weights_norm.append(w1_norm.detach().numpy())
    norms.append(norm.detach().numpy())
    biases[0,:] = b1.detach().numpy()

    #epoch counter
    k = 1


    loss_list     = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()
        
        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}, Accuracy {correct.mean()}")
        
        
        
        w1 = model.layer1.weight
        b1 = model.layer1.bias

        if epoch % 5 == 0:
            print("check inj")
            # check injectivity 
            percent_inj = mcbe.check_injectivity_naive(w1.detach().numpy(),b1.detach().numpy(),points=X_test[:1000,:],iter=X_test[:1000,:].shape[0])
            percent_inj_list.append(percent_inj)
            print("percent_inj: ", percent_inj)

        norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
        w1_norm = torch.div(w1,norm)
        w1_norm[w1_norm == np.inf] = 0
        
        weights.append(w1.detach().numpy())
        weights_norm.append(w1_norm.detach().numpy())
        norms.append(norm.detach().numpy())
        biases[k,:] = b1.detach().numpy()
        
        k = k+1

    percent_inj9.append(percent_inj_list)

mean_inj9 = np.mean(percent_inj9,axis=0)
std_inj9 = np.std(percent_inj9,axis=0)
np.save('mean_inj9_cifar.npy',mean_inj9)
np.save('std_inj9_cifar.npy',std_inj9)