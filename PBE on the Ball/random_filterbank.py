import numpy as np
import scipy as sp

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
    assert N%a == 0, "a must be a divisor of N"
    W = [np.vstack(sp.linalg.circulant(w[j,:]).T[::a]) for j in range(J)]
    return np.array(W).reshape(J*N//a,N)


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
    
    assert N%a == 0, "a must be a divisor of N"

    if T == None:
        T = N
    if scale:
        w = np.random.randn(J, T)/np.sqrt(T*J)
    if norm:
        norm = np.linalg.norm(w, axis=1)
        w = w/norm[:,None]
    else:
        w = np.random.randn(J, T)
    w_pad = np.pad(w, ((0,0),(0, N-T)), constant_values=0)
    if analysis:
        return fb_ana(w_pad, a=a)

    return w_pad