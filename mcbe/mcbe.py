import numpy as np
import torch
import os
import scipy
import matplotlib.pyplot as plt
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


def solve_N(d, epsilon, starting_estimate=100):
    '''solve for N so that (log(N)/N)^(1/d) <= epsilon to find min sampling points N so that the expected value of the
    Euclidean covering radius of the sphere is asymptotically?? epsilon
    starting estimate is a hyper parameter from scipy.optimize.fsolve'''

    def objective(x):
        kappa_d = (1 / d) * (scipy.special.gamma((d + 1) / 2) / (np.sqrt(np.pi) * scipy.special.gamma(d / 2)))
        return (np.log(x) / (x*kappa_d)) ** (1 / d) - epsilon

    return scipy.optimize.fsolve(objective, starting_estimate)[0]

def mcbe(polytope, N, distribution="sphere", radius=1, radius_inner=0.1, give_subframes=False, plot=False, iter_plot = 100):
    '''
    Monte Carlo Sampling Approach for Bias Estimation

    Usage:
    distribution choose from sphere, ball, donut the space from which the data points will be drawn
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

    subframes = []
    points = []

    test_points = get_points(distribution, num_vert, d, radius,radius_inner)
    percent_inj = []

    for i in range(int(np.ceil(N))):

        # sample x
        point = get_point("sphere", d, radius)
        points.append(point)

        corr_x_vert = [np.dot(point, i) for i in polytope]

        #find subframes
        if give_subframes == True:
            subframe = np.argsort(corr_x_vert)[-d:]
            subframes.append(tuple(np.sort(subframe)))

        # find the d-nearest point of the polytope
        i = np.argsort(corr_x_vert)[-d]

        # if correlation is smaller than the i-th position in alpha overwrite it
        alpha[i] = np.min([alpha[i], corr_x_vert[i]])

        if plot == True:
            percent_inj.append(check_injectivity_naive(polytope, alpha, iter_plot, distribution, radius, radius_inner, points=test_points))


    if distribution == "ball":
        #if distribution is ball set all positive alpha values to zero
        alpha[alpha >= 0] = 0

    if distribution == "donut":
        alpha = alpha*radius_inner

    if plot == True:
        percent_inj.append(
            check_injectivity_naive(polytope, alpha, iter_plot, distribution, radius, radius_inner, points=test_points))

        plt.plot(percent_inj)
        plt.xlabel("iteration")
        plt.ylabel("percent of test set injective")
        plt.title("training process")


    if give_subframes == True:
        return alpha/np.linalg.norm(polytope,axis=1), set(subframes), points
    else:
        return alpha/np.linalg.norm(polytope,axis=1)


def check_injectivity_naive(W, b, iter, distribution="sphere", radius=1, radius_inner=0.1, points=[]):
    '''distribution, thres_range, thres, radius, radius_inner.. parameters for sample_method()
    iter.. number of injectivity tests run
    checks injectivity with given parameter iter times and returns percentage of injectivity in the trials'''

    bool_injective = []
    d = W.shape[1]

    if list(points):
        iter = len(points)


    for i in range(0, iter):

        if list(points):
            x = points[i]

        else:
            x = get_point(distribution,d,radius,radius_inner)

        bool_injective.append(np.sum(relu(x, W, b) > 0) >= d)

    return np.mean(bool_injective)







