import numpy as np
import warnings

def relu(W, x, a):
    """
    Computes the forward pass of a ReLU-layer (convention here: negative bias)

    Parameters:
    W:  ndarray [m x n]
        Weight matrix
    x:  ndarray [n x 1]
        Input
    a:  ndarray [m x 1]
        Bias vector
    
    Returns:
    y : ndarray
        The vector after applying the frame algorithm.
    """
    z = W @ x - a
    z = z * (z > 0)
    return np.where(z == -0.0, 0.0, z)

def relu_frame_algorithm(W, a, out, ver='relu', it=100):
    """
    Implements the ReLU frame algorithm.
    
    Parameters:
    W:  ndarray [m x n]
        The weight matrix of the ReLU layer
    a:  ndarray [m x 1]
        The bias vector of the ReLU layer
    out : ndarray [m x 1]
        The output of the ReLU layer with weight matrix W and bias a.
    ver : string
        The version of the frame algorithm. Valid strings are
        'full' for the original frame algorithm. In this case, *out* should come from a linear layer
        'relu_naive' for the naive relu frame algorithm
        'relu' for the modified relu frame algorithm (default)
    iterations : int
        The number of iterations to perform. Default 100
    
    Returns:
    y : ndarray
        The vector after applying the frame algorithm.
    """
    S = W.T @ W
    lams = np.linalg.eigvals(S)
    A = np.min(lams)
    B = np.max(lams)
    lam = 2 / (A + B)

    coeff = out

    I_a = np.asarray(coeff > 0).nonzero()[0]

    if I_a.size < W.shape[-1]:
        warnings.warn("The ReLU layer is not injective, perfect reconstruction is impossible!")

    y = np.zeros([W.shape[-1],1])
    
    if ver == 'relu_naive':
        W_active = W[I_a,:]
        for i in range(it):
            coeff_y = W @ y
            y = y + lam * W_active.T @ (coeff[I_a] + a[I_a] - coeff_y[I_a])
        return y
            
    if ver == 'relu':  
        W_active = W[I_a,:]
        for i in range(it):          
            coeff_y = W @ y
            I_y = np.asarray(coeff_y >= a).nonzero()[0]
            I_y = [x for x in I_y if x not in I_a]
            W_y = W[I_y,:]
            y = y + lam * ( W_active.T @ (coeff[I_a] + a[I_a] - coeff_y[I_a]) + W_y.T @ (a[I_y] - coeff_y[I_y]) )     
        return y