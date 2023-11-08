import numpy as np
import torch
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def norm_row(W):
    """
    takes a weight matrix W and normalizes the rows
    """
    if torch.is_tensor(W):
        W = W.detach().numpy()
    norm = np.linalg.norm(W, axis=1)
    W_norm = W / norm[:, None]
    return W_norm, norm

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

def random_ball(num_points, dimension, radius=1):
    random_directions = np.random.normal(size=(dimension,num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = np.random.random(num_points) ** (1/dimension)

    return radius * (random_directions * random_radii).T

def random_donut(num_points, dimension, radius_outer=1,radius_inner=0.1):
    random_directions = np.random.normal(size=(dimension,num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = np.random.uniform(radius_inner,1,size=(num_points))

    return radius_outer * (random_directions * random_radii).T

def random_point(num_points, dimension):

    return np.random.randn(num_points, dimension)

def random_sphere(num_points, dimension, radius=1):

    return radius*norm_row(np.random.randn(num_points, dimension))


def get_point(distribution, d, radius=1,radius_inner=0.1):

    if distribution == "sphere":
        return random_sphere(1, d, radius)[0][0]
    elif distribution == "normal":
        return random_point(1, d)[0]
    elif distribution == "donut":
        return random_donut(1, d, radius_outer=radius,radius_inner=radius_inner)[0]
    elif distribution == "ball":
        return random_ball(1, d, radius)[0]
    else:
        raise ValueError("distribution not found")


def get_points(distribution, num_points, d, radius=1,radius_inner=0.1):

    if distribution == "sphere":
        return random_sphere(num_points, d, radius)[0]
    elif distribution == "normal":
        return random_point(num_points, d)
    elif distribution == "donut":
        return random_donut(num_points, d, radius_outer=radius,radius_inner=radius_inner)
    elif distribution == "ball":
        return random_ball(num_points, d, radius)
    else:
        raise ValueError("distribution not found")


def mcbe(polytope, distribution="sphere", thres_range = 500, thres = 0, radius=1, radius_inner=0.1, give_subframes=False, quiet=False):
    '''
    Monte Carlo Sampling Approach for Bias Estimation

    Usage:
    distribution choose from normal, sphere, ball, donut
    stops after iteration (k + thres range) if: mean(alpha_k) - mean(alpha_{k+thres_range}) <  thres
    radius.. if distribution is ball, donut or sphere
    radius_inner.. if distribution = donut
    give_subframes.. if True: also returns subframes calculated for the points used for approximation

    Output:
    approximated bias; means of values of alpha across iterations
    '''



    d = polytope.shape[1]
    num_vert = polytope.shape[0]

    # initiate alpha as inf
    alpha = np.zeros(num_vert)
    alpha[:] = np.inf

    alphas = []

    iter = 0
    subframes = []

    while iter <= thres_range:

        # sample x
        point = get_point(distribution, d, radius,radius_inner)

        corr_x_vert = [np.dot(point, i) for i in polytope]

        #find subframes
        if give_subframes == True:
            subframe = np.argsort(corr_x_vert)[-d:]
            subframes.append(tuple(np.sort(subframe)))

        # find the d-nearest point of the polytope
        i = np.argsort(corr_x_vert)[-d]

        # if correlation is smaller than the i-th position in alpha overwrite it
        alpha[i] = np.min([alpha[i], corr_x_vert[i]])


        alphas.append(alpha.copy())

        iter = iter+1


    diff_thres = np.linalg.norm(np.array(alphas[-thres_range])-np.array(alphas[-1]))


    while (diff_thres > thres) or np.isnan(diff_thres):

        # sample x
        point = get_point(distribution, d, radius,radius_inner)

        corr_x_vert = [np.dot(point, i) for i in polytope]

        # find subframes
        if give_subframes == True:
            subframe = np.argsort(corr_x_vert)[-d:]
            subframes.append(tuple(np.sort(subframe)))

        # find the d-nearest point of the polytope
        i = np.argsort(corr_x_vert)[-d]

        # if correlation is smaler than the i-th position in alpha overwrite it
        alpha[i] = np.min([alpha[i], corr_x_vert[i]])

        alphas.append(alpha.copy())

        iter = iter + 1

        diff_thres = np.linalg.norm(np.array(alphas[-thres_range])-np.array(alphas[-1]))

    if not quiet:
        print("Bias estimation converged after", iter, "iterations")

    if give_subframes == True:
        return alpha/np.linalg.norm(polytope,axis=1), set(subframes)
    else:
        return alpha/np.linalg.norm(polytope,axis=1),


def check_injectivity(d, num_vert, iter, distribution="sphere", thres_range = 500, thres = 0, radius=1, radius_inner=0.1, b=0):
    '''distribution, thres_range, thres, radius, radius_inner.. parameters for sample_method()
    iter.. number of injectivity tests run
    checks injectivity with given parameter iter times and returns percentage of injectivity in the trials'''

    bool_injective = []

    for i in tqdm(range(0, iter)):
        W = norm_row(np.random.randn(num_vert, d))[0]

        x = get_point(distribution,d,radius,radius_inner)

        alpha = mcbe(W, distribution, thres_range, thres, radius, radius_inner, quiet=True)
        bool_injective.append(np.sum(relu(x, W, alpha) > b) >= d)

    return np.mean(bool_injective)


def injectivity_on_test_set(W, distribution, num_test_points, num_iter, radius=1, radius_inner=0.1):

    """checks injectivity on test set random drawn from distribution for every trainings iteration
    Usage:
    W.. weight matrix
    num_test_points.. number of points in test set
    num_iter.. number of trainings iterations
    distribution, radius, radius_inner.. parameter vor mcbe

    Output:
    percent of samples in the test set for which the relu layer with W and the estimated alpha is injective for every
    iteration"""

    d = W.shape[1]
    num_vert = W.shape[0]

    # initiate alpha as inf
    alpha = np.zeros(num_vert)
    alpha[:] = np.inf

    means_alpha = []

    test_points = get_points(distribution, num_test_points, d, radius, radius_inner)

    inj = []

    for it in tqdm(range(num_iter)):
        # sample x
        point = get_point(distribution, d, radius, radius_inner)

        corr_x_vert = [np.dot(point, i) for i in W]

        # find the d-nearest point of the polytope
        i = np.argsort(corr_x_vert)[-d]

        # if correlation is smaller than the i-th position in alpha overwrite it
        alpha[i] = np.min([alpha[i], corr_x_vert[i]])

        means_alpha.append(np.mean(alpha))

        # check injectivity
        inj_temp = []

        if np.max(alpha) < np.inf:
            for x in test_points:
                inj_temp.append(np.sum(relu(x, W, alpha) != 0) >= d)

            inj.append(np.mean(inj_temp))

    return inj

def min_N(p,d,n):
    assert p <= 1
    assert p > 0
    assert d > 0
    return int(np.ceil(np.log((1-p)/n**d)/np.log((n**d-1)/n**d)))


