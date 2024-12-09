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
    random_radii = np.random.uniform(radius_inner**dimension,radius_outer**dimension,size=(num_points)) ** (1/dimension)

    return (random_directions * random_radii).T

def random_point(num_points, dimension):

    return np.random.randn(num_points, dimension)

def random_sphere(num_points, dimension, radius=1):

    return radius*norm_row(np.random.randn(num_points, dimension))[0], radius*norm_row(np.random.randn(num_points, dimension))[1]


def get_point(distribution, d, radius=1,radius_inner=0.1, positive = False, nonnegative = False):
    """sample point from distribution of dimensionality d
    if positive=True all coordinates are positive
    if nonnegative=True not all coordinates are negative"""

    point = np.array([-1]*d)

    if nonnegative:
        while np.all(point < 0):
            if distribution == "sphere":
                point =  random_sphere(1, d, radius)[0][0]
            elif distribution == "normal":
                point =  random_point(1, d)[0]
            elif distribution == "donut":
                point =  random_donut(1, d, radius_outer=radius,radius_inner=radius_inner)[0]
            elif distribution == "ball":
                point =  random_ball(1, d, radius)[0]
            else:
                raise ValueError("distribution not found")

    else:
        if distribution == "sphere":
            point =  random_sphere(1, d, radius)[0][0]
        elif distribution == "normal":
            point =  random_point(1, d)[0]
        elif distribution == "donut":
            point =  random_donut(1, d, radius_outer=radius,radius_inner=radius_inner)[0]
        elif distribution == "ball":
            point =  random_ball(1, d, radius)[0]
        else:
            raise ValueError("distribution not found")

    if positive:
        return np.abs(point)
    else:
        return point



def get_points(distribution, num_points, d, radius=1,radius_inner=0.1, positive = False, nonnegative = False):
    """sample num_points points from distribution of dimensionality d
        if positive=True all coordinates are positive
        if nonnegative=True not all coordinates are negative"""
    points = []

    if nonnegative:
        while len(points) < num_points:
            if distribution == "sphere":
                samples = random_sphere(num_points, d, radius)[0]
                nonneg_sample = samples[[np.any(s > 0) for s in samples]]
                for sample in nonneg_sample:
                    points.append(sample)
            elif distribution == "normal":
                samples = random_point(num_points, d)
                nonneg_sample = samples[[np.any(s > 0) for s in samples]]
                for sample in nonneg_sample:
                    points.append(sample)
            elif distribution == "donut":
                samples = random_donut(num_points, d, radius_outer=radius,radius_inner=radius_inner)
                nonneg_sample = samples[[np.any(s > 0) for s in samples]]
                for sample in nonneg_sample:
                    points.append(sample)
            elif distribution == "ball":
                samples = random_ball(num_points, d, radius)
                nonneg_sample = samples[[np.any(s > 0) for s in samples]]
                for sample in nonneg_sample:
                    points.append(sample)
            else:
                raise ValueError("distribution not found")

    else:
        if distribution == "sphere":
            points = random_sphere(num_points, d, radius)[0]
        elif distribution == "normal":
            points = random_point(num_points, d)
        elif distribution == "donut":
            points = random_donut(num_points, d, radius_outer=radius, radius_inner=radius_inner)
        elif distribution == "ball":
            points = random_ball(num_points, d, radius)
        else:
            raise ValueError("distribution not found")
    if positive:
        return np.abs(np.array(points)[:12])
    else:
        return np.array(points)[:12]


def solve_N(d, epsilon, starting_estimate=100):
    """solve for N so that (log(N)/N)^(1/d) <= epsilon to find min sampling points N so that the expected value of the
    Euclidean covering radius of the sphere is asymptotically?? epsilon
    starting estimate is a hyper parameter from scipy.optimize.fsolve"""

    def objective(x):
        kappa_d = (1 / d) * (scipy.special.gamma((d + 1) / 2) / (np.sqrt(np.pi) * scipy.special.gamma(d / 2)))
        return (np.log(x) / (x*kappa_d)) ** (1 / d) - epsilon

    N = scipy.optimize.fsolve(objective, starting_estimate)[0]
    return int(np.ceil(N))

def solve_eps(d, N):
    '''solve for epsilon so that (log(N)/N)^(1/d) <= epsilon to find min sampling points N so that the expected value of the
        Euclidean covering radius of the sphere is asymptotically?? epsilon'''

    kappa_d = (1 / d) * (scipy.special.gamma((d + 1) / 2) / (np.sqrt(np.pi) * scipy.special.gamma(d / 2)))

    return ((np.log(N))/(N*kappa_d))**(1/d)


def mcbe(polytope, N, distribution="sphere", radius=1, radius_inner=0.1, give_subframes=False, plot=False, iter_plot = 100, K_positive = False, init=True, sample_on_sphere = True, return_alpha_list = False, return_plot_data = False, remove_covering_radius = False):
    '''
    Monte Carlo Sampling Approach for Bias Estimation

    Usage:
    distribution choose from sphere, ball, donut the space from which the data points will be drawn
    radius.. if distribution is ball, donut or sphere
    radius_inner.. if distribution = donut
    give_subframes.. if True: also returns subframes calculated for the points used for approximation
    plot.. if true plot the injectivity of a test set per iteration
    iter_plot.. number of testsamples used in the plot
    K_positive.. set True if the Set K is known to be positive

    Output:
    approximated bias
    '''

    d = polytope.shape[1]
    num_vert = polytope.shape[0]

    # initiate alpha as inf
    alpha = np.zeros(num_vert)
    alpha[:] = np.inf
    alpha_list = []

    subframes = []
    points = []

    if K_positive:
        positive = True
        nonnegative = True
    else:
        positive = False
        nonnegative = False

    test_points = get_points(distribution, num_vert, d, radius, radius_inner, positive=positive)
    percent_inj = []

    if init == True:
        # initiate alpha by cross correlations among Phi
        for i in range(num_vert):
            corr_x_vert = [np.dot(polytope[i,:], phi) for phi in polytope]
            idx = np.argsort(corr_x_vert)[-d]
            alpha[i] = np.min([alpha[idx], corr_x_vert[idx]])

    for i in range(int(np.ceil(N))):

        if i % 50 ==1:
            alpha_list.append(alpha/np.linalg.norm(polytope,axis=1))

        # sample x
        if sample_on_sphere == True:
            point = get_point("sphere", d, radius, nonnegative=nonnegative)
            points.append(point)

        else:
            point = get_point(distribution, d, radius, radius_inner, positive=positive, nonnegative=nonnegative)
            points.append(point)

        corr_x_vert = [np.dot(point, phi) for phi in polytope]

        #find subframes
        if give_subframes == True:
            subframe = np.argsort(corr_x_vert)[-d:]
            subframes.append(tuple(np.sort(subframe)))

        # find the d-nearest point of the polytope
        idx_list = np.argsort(corr_x_vert)[-d:]

        # if correlation is smaller than the i-th position in alpha overwrite it
        for idx in idx_list:
            alpha[idx] = np.min([alpha[idx], corr_x_vert[idx]])

        if plot == True:
            # remove covering radius 
            alpha_plot = alpha
            if remove_covering_radius == True:
                covering_radius = (np.log(i)/i)**(1/d)
                alpha_plot = alpha_plot - covering_radius
            percent_inj.append(check_injectivity_naive(polytope, alpha_plot, iter_plot, distribution, radius, radius_inner, points=test_points))

    if sample_on_sphere == True:   

        if distribution == "ball":
            #if distribution is ball set all positive alpha values to zero
            alpha[alpha >= 0] = 0

        if distribution == "donut":
            alpha = alpha*radius_inner

    if remove_covering_radius == True:
        covering_radius = (np.log(N)/N)**(1/d)
        alpha = alpha - covering_radius


    if plot == True:
        percent_inj.append(
            check_injectivity_naive(polytope, alpha, iter_plot, distribution, radius, radius_inner, points=test_points))

        plt.plot(percent_inj)
        plt.xlabel("iteration")
        plt.ylabel("percent of test set injective")


    if give_subframes == True:
        return alpha/np.linalg.norm(polytope,axis=1), set(subframes), points
    elif return_alpha_list == True:
        return alpha/np.linalg.norm(polytope,axis=1), alpha_list
    elif return_plot_data == True:
        return alpha/np.linalg.norm(polytope,axis=1), percent_inj
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
    

    





    

