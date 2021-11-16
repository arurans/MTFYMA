import numpy as np
from scipy import special


def Bernstein(i, n, t):
    """
    Construct bernstein polynomial og n-th degree for point i at time t.
    Supports array of t.

    Arguments
    ---------------------------------------
        i: int
            Point
        n: int
            degree of polynomial
        t: float
            time-step to be evaluated, can be array
    
    Returns
    ---------------------------------------
    The Bernstein-polynomial
    """
    return special.binom(n, i) * t ** i * (1 - t) ** (n - i)

def deCasteljau(P, t):
    """
    deCasteljau algorithm

    Arguments
    ---------------------------------------
        P: dim x n - matrix
            dim is the dimension of the points and n is number of points
        t: float
            time-step to be evaluated
    
    Returns
    ---------------------------------------
    tuple:
        1st element:
            The point of bernstein polynomial at time t.
        2nd element:
            The point of each level of Bernstein polynomial at time t.
    """
    d, K = P.shape #dimension, number of points
    Pvecs = np.zeros((K, d, K))
    Pvecs[0] = P

    for k in range(1, K):
        for i in range(K-k):
            Pvecs[k,:,i] = (1-t) * Pvecs[k-1,:,i] + t * Pvecs[k-1,:,i+1]

    return (Pvecs[-1,:,0], Pvecs)
