
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import tqdm
import mcbe
from sklearn.preprocessing import StandardScaler
import keras.datasets
import argparse

#parse arguments
parser = argparse.ArgumentParser(description='give number of estimation points')
parser.add_argument('--num_estimation_points', type=int, help='number of estimation points for ddmcbe')
args = parser.parse_args()

num_estimation_points = args.num_estimation_points
print("ddmcbe running with",num_estimation_points,"estimation points")

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = np.array([X_train[i].flatten() for i in range(X_train.shape[0])])
X_test = np.array([X_test[i].flatten() for i in range(X_test.shape[0])])

#normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# size of the layers
l1 = 1024
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
    
model     = Model(X_train.shape[1],l1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()
print(model)

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

#initialize lists

EPOCHS  = 50

weights = []
weights_norm = []
norms = []
biases = np.zeros([EPOCHS + 1,l1])

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
    norm = w1.pow(2).sum(keepdim=True,dim=1).sqrt()
    w1_norm = torch.div(w1,norm)
    w1_norm[w1_norm == np.inf] = 0
    
    weights.append(w1.detach().numpy())
    weights_norm.append(w1_norm.detach().numpy())
    norms.append(norm.detach().numpy())
    biases[k,:] = b1.detach().numpy()
    
    k = k+1
   
print("Training done")

num_iter = 10

inj_kd = []
inj_rd = []
inj_blowup = []
inj_mcbe = []
inj_og = []
inj_small_bias = []

for i in tqdm.tqdm(range(num_iter)):

    #kernel density estimation
    est_alpha_kd = mcbe.dd_mcbe(np.array(weights)[-1], X_train, num_estimation_points, dd_method="kde")
    percent_inj_kd = mcbe.check_injectivity_naive(W = np.array(weights)[-1], b=est_alpha_kd, points=X_test,iter=X_test.shape[0])
    inj_kd.append(percent_inj_kd)

    #blown up data
    est_alpha_blowup = mcbe.dd_mcbe(np.array(weights)[-1], X_train,  num_estimation_points, dd_method="blowup")
    percent_inj_blowup = mcbe.check_injectivity_naive(W = np.array(weights)[-1], b=est_alpha_blowup, points=X_test,iter=X_test.shape[0])
    inj_blowup.append(percent_inj_blowup)

    #mcbe
    est_alpha = mcbe.mcbe(polytope=np.array(weights)[-1],N=num_estimation_points,distribution="normal",radius=np.max(np.array(X_train)), sample_on_sphere=False)
    percent_inj = mcbe.check_injectivity_naive(W = np.array(weights)[-1], b=est_alpha, points=X_test,iter=X_test.shape[0])
    inj_mcbe.append(percent_inj)

inj_mnist = pd.DataFrame({"kd":inj_kd,"blowup":inj_blowup,"mcbe":inj_mcbe})
inj_mnist.to_csv("inj_mnist" + str(num_estimation_points) +".csv")