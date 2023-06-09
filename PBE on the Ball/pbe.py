# This file contains all functions to compute the PBE on the ball for a matrix W and do facet-specific reconstruction:
# - row-wise normalization
# - export in homogemeous coordinates
# - check for omnidirectionality
# - compute alpha^S via convex optimization
# - read Polymake output
# - PBE
# - facet-specific reconstruction 

import torch
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import csv


def norm_row(W):
    """
    takes a weight matrix W and normalizes the rows
    """
    if torch.is_tensor(W):
        W = W.detach().numpy()
    norm = np.linalg.norm(W, axis=1)
    W_norm = W / norm[:, None]
    return W_norm, norm


def hom(W, filename='Test'):
    """
    takes a weight matrix W and saves it in homogemeous coordiantes, ready to be used by Polymake
    """
    np.set_printoptions(precision=4, threshold=10_000, suppress=True)
    if torch.is_tensor(W):
        W = W.detach().numpy()
    h = np.ones((W.shape[0], 1))
    W_hom = np.concatenate((h, W), axis=1)
    mat = np.matrix(W_hom)
    with open(filename + '.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
    return


def is_omnidir(W):
    """
    takes a weight matrix W and checks for omnidirectionality, binary output
    """
    if torch.is_tensor(W):
        W = W.detach().numpy()
    WT = np.transpose(W)
    m = W.shape[0]
    n = W.shape[1]
    if np.linalg.matrix_rank(W) != n:
        return print('The system is not a frame')
    WW = np.concatenate([WT, [np.ones(m)]])
    ones = np.ones(m)
    zeros = np.concatenate([np.zeros(n), [1]])
    res = linprog(ones, A_eq=WW, b_eq=zeros)

    if res['message'] == 'The algorithm terminated successfully and determined that the problem is infeasible.':
        return False
    elif res['message'] == 'Optimization terminated successfully.':
        if np.any(res['x'] < 1e-10) == True:
            print('0 lies at or very very close to the boundary of the polytope.')
            return True
        else:
            return True


def alpha_S(F):
    """
    computes alpha^S for one facet F
    """
    if torch.is_tensor(F):
        F = F.detach().numpy()
    m, n = F.shape
    FT = F.T
    sol = []
    for i in range(m):
        f = np.matmul(F[i, :], FT)
        c = cp.Variable(m)
        soc_constraints = [cp.SOC(1, FT @ c)]  # cp.SOC(1, x) --> ||x||_2 <= 1.
        prob = cp.Problem(cp.Minimize(f @ c), soc_constraints + [c >= np.zeros(m)])
        result = prob.solve()
        sol.append(prob.value)
    return min(sol)
 

def read_facets(filename="facets.csv"):
    """
    reads the vertex-facets incidences as .csv file from Polymake and writes them into an array
    """
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        facets = [[int(num) for num in row[0][1:-1].split()] for row in csv_reader]
    return facets


def pbe(W, filename="facets.csv", radius=1):
    """
    PBE: takes a weight matrix W, the vertex-facets incidence .csv file from Polymake and a radius
    output: radius**-1 * alpha^B 
    """
    facets = read_facets(filename)

    if torch.is_tensor(W):
        W = W.detach().numpy()
    if is_omnidir(W) == False:
        return 'The frame is not omnidirectional'
    W_norm, norm = norm_row(W)

    m, n = W.shape

    beta = []
    k = 0
    for f in facets:
        gam = []
        # go through all facet vertices with index v1 < v2 to safe computations
        for v1 in f[:-1]:
            k = k + 1
            for v2 in f[k:]:
                gam.append(np.dot(W_norm[v1], W_norm[v2]))
            if min(gam) >= 0:
                beta.append(0)
            else:
                beta.append(alpha_S(W_norm[f]))
        k = 0

    alpha_norm = []
    for i in range(0, m):
        gam = []
        for f in range(0, len(facets)):
            if i in facets[f]:
                gam.append(beta[i])
        if not gam:
            return 'Some vertex slipped into the interior of the polytope in Polymake via a numerical error'
        alpha_norm.append(min(gam))

    alpha = np.multiply(alpha_norm, np.reciprocal(norm.T))*radius**(-1)
    return alpha


def relu(x, W, b):
    """
    computes the forward pass of a ReLU-layer (convention here: negative bias)
    """
    z = np.dot(W, x) - b
    return z * (z > 0)


def relu_inv(z, W, b, filename="facets.csv", mode='facet'):
    """
    reconstructs x from z = ReLU(Wx - b) using a facet-specific left-inverse 
    setting mode to something else will use the whole active sub-frame
    """
    facets = read_facets(filename)
    I = np.where(z > 0)[0]
    if mode == 'facet':
        for i in range(0, len(facets)):
            if all(k in I for k in facets[i]):
                break
        f_ind = facets[i]
        print('Facet', i, 'with vertices', f_ind, 'is used for reconstruction.')
    else:
        f_ind = I
    W_f = W[f_ind,:]
    b_f = b[f_ind]
    z_f = z[f_ind]
    x = np.linalg.lstsq(W_f, z_f + b_f, rcond=None)[0] # equivalent to synthesis with the canonical dual frame
    return x