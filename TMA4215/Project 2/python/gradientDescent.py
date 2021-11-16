from matplotlib import pyplot as plt
import numpy as np
from deCasteljau import *
from compositeBezier import *

##Plotting functions specific for gradientDescent. Could have been much cleaner...
def plotCommands_subplots(xlabel, ylabel, title, ax, xscale = False, yscale = False, legend = False, printString = ""):
    #Function containing the usual commands when plotting with matplotlib.pyplot
    #If log-scaled x- and/or y-axis is needed, set xscale and/or yscale to True
    #If the figure has a label, set legend = True
    #If extra information is needed, use printString
    ax.set(xlabel = xlabel, ylabel = ylabel)
    ax.set_title(title)
    # if xscale:
    #     ax.xscale('log')
    # if yscale:
    #     plt.yscale('log')
    if legend:
        ax.legend()
    ax.grid()
    if printString:
        print(printString)
    return None

def draw_bezier_comp_subplots(P, A, t, title, ax, imageIndex = 0):
    """
    Draw the composite bezier curve, with the interpolation points. 
    Use this to plot as a subplot

    Arguments
    ---------------------------------------
        P: m x dim x degree matrix
            m is the number of segments
            dim is the dimension of the problem
            degree is the number of points for each segment, the actual degree is one less
        A: dim x m - matrix
            dim is the dimension of the problem
            m is the number of points 
        t: float 
            Time step to be evaluated at
        title: string
            title of plot-figure
        ax: matplotlib ax-object
            axis to plot on in subplot
    Returns
    ---------------------------------------
    None
    """

    B = np.array([compositeBezier(P, t) for t in t])

    ax.plot(B[:,0], B[:,1], color = "red", label = "Bézier composition")
    ax.scatter(A[0], A[1], color = "forestgreen", label = "Interpolation points")
    plotCommands_subplots("x", "y", title, ax, legend = True)

    return None

def gradG_periodic(P,d,lambd):
    F=np.zeros(P.shape)
    m=P.shape[0]//2
    for i in range(0,m-1):
        F[2*i] = lambd*(P[2*i]-d[i])+12*(P[2*(i-1)]-6*P[2*(i-1)+1]+16*P[2*i]-12*P[2*i+1]+P[2*(i+1)])
        F[2*i+1] = 36*(P[2*(i-1)+1]-4*P[2*i]+4*P[2*i+1]-2*P[2*(i+1)]+P[2*(i+1)+1])
    F[-2] = lambd*(P[-2]-d[-1])+12*(P[-4]-6*P[-3]+16*P[-2]-12*P[-1]+P[0])
    F[-1] = 36*(P[-3]-4*P[-2]+4*P[-1]-2*P[0]+P[1])
    return F

def gradF(P):
    #Compute and return the gradient of F
    F=np.zeros(P.shape)
    m=P.shape[0]//2-1
    for i in range(m-1):
        F[2*i] += 12*(2*P[2*i]-3*P[2*i+1]+P[(i+1)*2])
        F[2*i+1] += 36*(-P[2*i]+2*P[2*i+1]-2*P[(i+1)*2]+P[(i+1)*2+1])
        F[2*i+2] += 12*(P[2*i]-6*P[2*i+1]+14*P[(i+1)*2]-9*P[(i+1)*2+1])
        F[2*i+3] += 36*(P[2*i+1]-3*P[(i+1)*2]+2*P[(i+1)*2+1])

    F[-4] += 12*(2*P[-4]-3*P[-3]+P[-1])
    F[-3] += 36*(-P[-4]+2*P[-3]-P[-2])
    F[-2] += 36*(-P[-3]+2*P[-2]-P[-1])
    F[-1] += 36*(P[-4]-3*P[-2]+2*P[-1])
    
    return F
                
def gradG(P,d,lambd):
    #Compute and return the gradient of G
    m= P.shape[0]//2-1
    G = np.zeros(P.shape)
    for i in range(m):
        G[2*i] = P[2*i] - d[i]

    G[-1] = P[-1] - d[-1]
    G = lambd* G + gradF(P)
    return G

def GradientDescent(P0, d, lam, grad, Ptol=10**(-5), maxiter=10000, stepsize=0.0001):
    """
    Gradient descent of G given datapoints d

    Arguments
    ---------------------------------------
    P0: 2m x dim
        m is the number of interpolation points 
        dim is the problem dimension
    d: mxdim
        interpolation points
    lam: float
        weighting of the different terms
    grad: function
        the gradient function (periodic or non-periodic)
    Ptol: float
        error-tolerance
    maxiter: int
        maximum number of iterations
    stepsize: float
        stepsize for each iteration
    Returns
    ---------------------------------------
    Pk1: 
        The modified P after applying gradient descent
    """
    k=0
    Pk,Pk1=P0.copy(),P0.copy()
    curr_tol = float("inf")
    while (k<=maxiter) and curr_tol>=Ptol:
        Pk = Pk1.copy()
        dk = -grad(Pk, d, lam)
        Pk1 = Pk + stepsize * dk
        k += 1
        curr_tol = np.linalg.norm(Pk1-Pk)
    return Pk1


def modify_P(P):
    """
    Modify P not to store redundant data, to save memory

    Arguments
    ---------------------------------------
    P: m x dim x deg matrix
        m is the number of segments
        dim is the problem dimension
        deg is the number of points for each segment, the actual is one lower
    
    Returns
    ---------------------------------------
    P_new: 2m x dim matrix
    """
    
    m, dim, deg = P.shape
    P_new = np.zeros((2*m, dim)) 

    for i in range(m):
        P_new[2*i:2*i+2] = P[i,:,:2].T


    return P_new

def reconstruct_P(P, degree = 4):
    """
    Reconstruct P to contain all points (inverse of modify_P)

    Arguments
    ---------------------------------------
    P: 2m x dim matrix
        m is the number of segments
        dim is the problem dimension
    degree: int
        the number of points for each segment, the actual degree is one lower
    Returns
    ---------------------------------------
    rec_P: m x dim x degree matrix
        reconstructed matrix
    """
    n, dim = P.shape
    m = n//2
    rec_P = np.zeros((m, dim, degree))

    for i in range(m - 1):
        rec_P[i,:,:2] = P[2*i:2*i+2].T
        next_p = P[2*(i+1):2*(i+1)+2].T
        rec_P[i,:,2] = 2*next_p[:,0] - next_p[:,1]
        rec_P[i,:,3] = next_p[:,0]
    
    rec_P[-1,:,:2] = P[-2:].T
    first_p = P[:2].T
    rec_P[-1,:,2] = 2*first_p[:,0] - first_p[:,1]
    rec_P[-1,:,3] = first_p[:,0]

    return rec_P


def OptimizedBezier(P, A, t, d, title, ax, lam = 10):
    """
    Optimize Bézier curve with gradient descent

    Arguments
    ---------------------------------------
    P: m x dim x deg matrix
        m is the number of segments
        dim is the problem dimension
        deg is the number of points for each segment, the actual is one lower
    A: dim x m matrix
    t: float array of time-steps
    d: initial data-point
    title:string
        title of plot
    ax: matplotlib axis object
        the axis to plot in the subplot
    lam: float
        weighting of terms
    Returns
    ---------------------------------------
    None
    """
    P = modify_P(P)
    P = GradientDescent(P, d, grad = gradG_periodic, lam = lam) # Normal until lamda = 19785, with stepsize = 0.0001
    P = reconstruct_P(P)
    draw_bezier_comp_subplots(P, A, t, title + " (periodic Gradient Descent), " + r"$\lambda$" + f" = {lam}", ax)

    return None

def modify_P_non_periodic(P):
    """
    Modify P not to store redundant data, to save memory. For non-periodic problem

    Arguments
    ---------------------------------------
    P: m x dim x deg matrix
        m is the number of segments
        dim is the problem dimension
        deg is the number of points for each segment, the actual is one lower
    
    Returns
    ---------------------------------------
    P_new: 2m+2 x dim matrix
    """
    m, dim, deg = P.shape
    P_new = np.zeros((2*m+2, dim)) 

    for i in range(m):
        P_new[2*i:2*i+2] = P[i,:,:2].T

    P_new[-2:] = P[-1,:,-2:]

    return P_new

def reconstruct_P_non_periodic(P, degree = 4):

    """
    Reconstruct P to contain all points (inverse of modify_P). For non-periodic gradient descent

    Arguments
    ---------------------------------------
    P: 2m x dim matrix
        m is the number of segments
        dim is the problem dimension
    degree: int
        the number of points for each segment, the actual degree is one lower
    Returns
    ---------------------------------------
    rec_P: m x dim x degree matrix
        reconstructed matrix
    """
    n, dim = P.shape
    m = n//2 - 1
    rec_P = np.zeros((m, dim, degree))

    for i in range(m - 1):
        rec_P[i,:,:2] = P[2*i:2*i+2].T
        next_p = P[2*(i+1):2*(i+1)+2].T
        rec_P[i,:,2] = 2*next_p[:,0] - next_p[:,1]
        rec_P[i,:,3] = next_p[:,0]
    
    rec_P[-1] = P[-4:].T

    return rec_P

def construct_d(A):
    #Create new matrix with the first point in A as its last. The size is one more point than A
    dim, n = A.shape
    d = np.zeros((n+1, dim))
    d[:-1] = A.T
    d[-1] = A[:,0].T

    return d

def OptimizedBezier_non_periodic(P, A, t, d, title, ax, lam = 10):
    """
    Optimize Bézier curve with gradient descent, non-periodic problem

    Arguments
    ---------------------------------------
    P: m x dim x deg matrix
        m is the number of segments
        dim is the problem dimension
        deg is the number of points for each segment, the actual is one lower
    A: dim x m matrix
    t: float array of time-steps
    d: initial data-point
    title:string
        title of plot
    ax: matplotlib axis object
        the axis to plot in the subplot
    lam: float
        weighting of terms
    Returns
    ---------------------------------------
    None
    """
    P = modify_P_non_periodic(P)
    P = GradientDescent(P, d, grad = gradG, lam = lam) # Normal until lamda = 19785, with stepsize = 0.0001
    P = reconstruct_P_non_periodic(P)
    draw_bezier_comp_subplots(P, A, t, title + " (non-periodic Gradient Descent), " + r"$\lambda$" + f" = {lam}", ax)

    return None