from deCasteljau import *

def compositeBezier(P, t):
    """
    Composite bezier function

    Arguments
    ---------------------------------------
        P: m x dim x degree matrix
            m is the number of segments
            dim is the dimension of the problem
            degree is the number of points for each segment, the actual degree is one less
        t: float 
            Time step to be evaluated at
    Returns
    ---------------------------------------
    The point at time t
    """
    m, dim, degree = P.shape
    b = np.zeros((dim))
    for i in range(1, m + 1):
        if i - 1 <= t < i:
            for j in range(degree):
                b += Bernstein(j, degree - 1, t - i + 1) * P[i-1, :, j]
    return b

def interpolate_periodic(A, V, degree = 4):
    """
    Interpolate each point and draw cubic bezier curves between them (as default)

    Arguments
    ---------------------------------------
        A:dim x m
            dim is dimension of the problem
            m is the number of interpolation points
    Returns
    ---------------------------------------
    P: m x dim x degree matrix
        m is the number of points
        dim is the problem dimension
        degree is the number of points for each segment, the actual degree is one lower
    t:
        array of time-points
    """
    dim, m = A.shape
    P = np.zeros((m, dim, degree))
    
    for i in range(m-1):
        P[i,:] = np.array([A[:,i], A[:,i] + 1/3*V[:,i], A[:,i+1] - 1/3*V[:,i+1], A[:,i+1]]).T
    
    P[-1,:] = np.array([A[:,-1], A[:,-1] + 1/3*V[:,-1], A[:,0] - 1/3*V[:,0], A[:,0]]).T

    t = np.linspace(0, m, m*1000, endpoint = False)

    return P, t


def mod_P(P, matrix, const = 0):
    """
    Transform P with a matrix multiplication and addition of constant vector

    Arguments
    ---------------------------------------
        P: m x dim x degree matrix
            m is the number of segments
            dim is the dimension of the problem
            degree is the number of points for each segment, the actual degree is one less
        matrix: dim x dim matrix
            Multiply each point with the matrix
        const: dim x 1 matrix
            add const to each point
    Returns
    ---------------------------------------
    modified: matrix with same shape as P
        the tranformed matrix
    """
    modified = np.zeros(P.shape)
    m, dim, degree = modified.shape
    
    for i in range(m):
        for j in range(degree):
            modified[i,:,j] = matrix @ P[i,:,j] + const

    return modified