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



def pbe(W, list_facets = [], filename="facets.csv", radius=1):
    """
    PBE: takes a weight matrix W, the vertex-facets incidence .csv file from Polymake and a radius
    output: radius**-1 * alpha^B
    """
    if list_facets:
        facets = list_facets
    else:
        facets = read_facets(filename)

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

        alpha_norm.append(np.min([min_corr,0]))

    alpha = np.multiply(alpha_norm, np.reciprocal(norm.T)) * radius ** (-1)
    return alpha


def relu(x, W, b):
    """
    computes the forward pass of a ReLU-layer (convention here: negative bias)
    """
    z = np.dot(W, x) - b
    return z * (z > 0)


def relu_inv(z, W, b, list_facets = [], filename="facets.csv", mode='facet'):
    """
    reconstructs x from z = ReLU(Wx - b) using a facet-specific left-inverse 
    setting mode to something else will use the whole active sub-frame
    """
    if list_facets:
        facets = list_facets
    else:
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


def fb_ana(w, a=1):
    '''
    This function returns the frame analysis matrix associated to a collection of filters with decimation factor a.

    Usage:
            W = fb_ana(w, a)
    Output:
            The JN/a x N frame analysis matrix associated with w and decimation factor a.
    '''

    N = w.shape[1]
    J = w.shape[0]
    assert N % a == 0, "a must be a divisor of N"
    W = [np.vstack(sp.linalg.circulant(w[j, :]).T[::a]) for j in range(J)]
    return np.array(W).reshape(J * N // a, N)


def randn_fb(N, J, T=None, scale=True, norm=False, analysis=True, a=1):
    '''
    This function creates a random filterbank with J filters of support T, sampled form a normal distribution and padded with zeros to have length N.
    If scale is set to True, the filters are divided by sqrt(J*T).
    If norm is set to True, the filters are normalized.
    If analysis is set to True, the function returns the frame analysis matrix of the filterbank.
    If analysis is set to False, the function returns the filterbank itself.
    The decimation factor a determined the stride in the convolution and must be a divisor of N.

    Usage:
            W = random_filterbank(N, J)
    Output:
            The NJxN analysis matrix associated with the filterbank
    '''

    assert N % a == 0, "a must be a divisor of N"

    if T == None:
        T = N
    if scale:
        w = np.random.randn(J, T) / np.sqrt(T * J)
    if norm:
        norm = np.linalg.norm(w, axis=1)
        w = w / norm[:, None]
    else:
        w = np.random.randn(J, T)
    w_pad = np.pad(w, ((0, 0), (0, N - T)), constant_values=0)
    if analysis:
        return fb_ana(w_pad, a=a)

    return w_pad

class convex_hull_method:

    def __init__(self, polytope):
        self.polytope = polytope
        self.d = polytope.shape[1]

    def facets(self):
        ##only use with low dimensions!!!!
        hull = ConvexHull(self.polytope)
        true_facets = hull.simplices

        return list(list(facet) for facet in facets)


