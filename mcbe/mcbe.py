from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import pbe
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from random import randint
from collections import Counter

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

    return radius*pbe.norm_row(np.random.randn(num_points, dimension))


def get_point(distribution, d, radius=1,radius_inner=0.1):

    if distribution == "sphere":
        return random_sphere(1, d, radius)[0][0]
    if distribution == "normal":
        return random_point(1, d)[0]
    if distribution == "donut":
        return random_donut(1, d, radius_outer=radius,radius_inner=radius_inner)[0]
    if distribution == "ball":
        return random_ball(1, d, radius)[0]

def mcbe(polytope, distribution="sphere", thres_range = 500, thres = 0.001, radius=1, radius_inner=0.1):
    '''
    Monte Carlo Sampling Approach for Bias Estimation

    Usage:
    distribution choose from normal, sphere, ball, donut
    stops after iteration (k + thres range) if: mean(alpha_k) - mean(alpha_{k+thres_range}) <  thres
    radius.. if distribution is ball, donut or sphere
    radius_inner.. if distribution = donut

    Output:
    approximated bias; means of values of alpha across iterations
    '''



    d = polytope.shape[1]
    num_vert = polytope.shape[0]

    # initiate alpha as inf
    alpha = np.zeros(num_vert)
    alpha[:] = np.inf

    means_alpha = []

    iter = 0

    while iter <= thres_range:

        # sample x
        point = get_point(distribution, d, radius,radius_inner)

        corr_x_vert = [np.dot(point, i) for i in polytope]

        # find the d-nearest point of the polytope
        i = np.argsort(corr_x_vert)[-d]

        # if correlation is smaller than the i-th position in alpha overwrite it
        alpha[i] = np.min([alpha[i], corr_x_vert[i]])

        means_alpha.append(np.mean(alpha))

        iter = iter+1


    diff_thres = np.mean(means_alpha[-thres_range]) - np.mean(means_alpha[-1])

    while (diff_thres > thres) or np.isnan(diff_thres):

        # sample x
        point = get_point(distribution, d, radius,radius_inner)

        corr_x_vert = [np.dot(point, i) for i in polytope]

        # find the d-nearest point of the polytope
        i = np.argsort(corr_x_vert)[-d]

        # if correlation is smaler than the i-th position in alpha overwrite it
        alpha[i] = np.min([alpha[i], corr_x_vert[i]])

        means_alpha.append(np.mean(alpha))

        iter = iter + 1
        diff_thres = np.mean(means_alpha[-thres_range]) - np.mean(means_alpha[-1])

    print("alpha converged after", iter, "iterations")


    return alpha/np.linalg.norm(polytope,axis=1), means_alpha


def check_injectivity(W, iter, distribution="sphere", thres_range = 500, thres = 0.001, radius=1, radius_inner=0.1):
    '''W.. parameter Matrix
    distribution, thres_range, thres, radius, radius_inner.. parameters for sample_method()
    iter.. number of injectivity tests run
    checks injectivity with given parameter iter times and returns percentage of injectivity in the trials'''

    bool_injective = []
    d = W.shape[1]
    num_vert = W.shape[0]

    for i in range(0, iter):
        W = pbe.norm_row(np.random.randn(num_vert, d))[0]

        x = get_point(distribution,d,radius,radius_inner)

        alpha, means_alpha = mcbe(W, distribution, thres_range, thres, radius, radius_inner)
        bool_injective.append(np.sum(pbe.relu(x, W, alpha) != 0) >= d)

    return np.mean(bool_injective)


