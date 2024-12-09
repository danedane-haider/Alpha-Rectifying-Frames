import mcbe
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# tetrahedron frame
W = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])/np.sqrt(3)
alpha, alpha_list =  mcbe.mcbe(W,10000,return_alpha_list=True)

alpha_lists = []
for iter in tqdm(range(100)):
    W = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])/np.sqrt(3)
    alpha, alpha_list =  mcbe.mcbe(W,10000,return_alpha_list=True)
    alpha_lists.append(alpha_list)

alpha_lists = np.array(alpha_lists)
eucl_dis_lists = np.linalg.norm(np.subtract(alpha_lists,-1/(np.sqrt(3))),axis=2)
means = eucl_dis_lists.mean(axis=0)
stds = eucl_dis_lists.std(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(means)
plt.fill_between(np.arange(len(means)),means-stds,means+stds,alpha=0.2)
plt.xlabel("iteration")
plt.ylabel("Euclidean distance to $\\alpha^{\\mathbb{S}}$")  
plt.grid(linestyle='--', alpha=0.5)
plt.xticks([0,50,100,150,200],["0","2500","5000","7500","10000"])
plt.yscale('log')
plt.show()