# This file contains all functions to compute the PBE on the scaled sphere, the closed ball for a matrix W and do facet-specific reconstruction:
# - row-wise normalization
# - check for omnidirectionality
# - compute alpha^S via convex optimization
# - PBE
# - facet-specific reconstruction 

import torch
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import scipy as sp
import csv
from scipy.spatial import ConvexHull


def norm_row(W):
    """
    takes a weight matrix W and normalizes the rows
    """
    if torch.is_tensor(W):
        W = W.detach().numpy()
    norm = np.linalg.norm(W, axis=1)
    W_norm = W / norm[:, None]
    return W_norm, norm


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

def facets(W):
    """
    computes the facets of the normalized row vectors of the matrix W
    use only with low dimensions!!!!
    """
    if torch.is_tensor(W):
        W = W.detach().numpy()
    hull = ConvexHull(W)
    facets = hull.simplices

    return list(list(facet) for facet in facets)

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


def pbe(W, facets, K='sphere', radius=1):
    """
    The Polytope Bias Estimation for approximating the maximal bias on K.

    Input: a weight matrix W, the list of vertex-facet incidences, the data domain K as string ('sphere', 'ball') and a radius
    Output: radius**-1 * alpha^K
    """

    if torch.is_tensor(W):
        W = W.detach().numpy()
    if is_omnidir(W) == False:
        return 'The frame is not omnidirectional'
    W_norm, norm = norm_row(W)

    m, n = W.shape
    alpha_norm = []

    for vert in range(0, m):
        neighbours = np.unique(np.array(facets)[[vert in facet for facet in facets]])
        corr_vec = W_norm[vert, :].dot(W_norm[neighbours, :].T)
        min_corr = np.min(corr_vec)
        if min_corr < 0:
            min_corr = alpha_S(W_norm[neighbours])
        if K == 'sphere':
            alpha_norm.append(min_corr)
        elif K == 'ball':
            alpha_norm.append(np.min([min_corr,0]))
        else:
            return 'Only sphere and ball are supported as data domains'

    return np.multiply(alpha_norm, np.reciprocal(norm.T)) * radius ** (-1)


def relu(x, W, b):
    """
    computes the forward pass of a ReLU-layer (convention here: negative bias)
    """
    z = np.dot(W, x) - b
    return z * (z > 0)


def relu_inv(z, W, b, facets, mode='facet'):
    """
    reconstructs x from z = ReLU(Wx - b) using a facet-specific left-inverse 
    setting mode to something else will use the whole active sub-frame
    """
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


