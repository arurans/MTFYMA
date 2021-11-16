from matplotlib import pyplot as plt
from deCasteljau import *
from compositeBezier import *

def plotCommands(xlabel, ylabel, title, xscale = False, yscale = False, legend = False, printString = ""):
    #Function containing the usual commands when plotting with matplotlib.pyplot
    #If log-scaled x- and/or y-axis is needed, set xscale and/or yscale to True
    #If the figure has a label, set legend = True
    #If extra information is needed, use printString
    plt.xlabel(xlabel, size = 20)
    plt.ylabel(ylabel, size = 20)
    plt.title(title, size = 27)
    if xscale:
        plt.xscale('log')
    if yscale:
        plt.yscale('log')
    if legend:
        plt.legend()
    plt.grid()
    if printString:
        print(printString)
    plt.show()
    return None

def plot_Bernstein(n, t):
    """
    Plot the Bernstein-polynomials of all degrees up to n

    Arguments
    ---------------------------------------
        n: int
            degree of polynomial
        t: float
            time-step to be evaluated, can be array
    
    Returns
    ---------------------------------------
    None
    """
    for i in range(n+1):
        plt.plot(t, Bernstein(i, n, t), label = f"i = {i}")
        
    plotCommands("t", r"$B_{i,n}(t)$", f"Bernstein polynomials up to degree n = {n}", legend=True)

    return None

def draw_each_level(P, t):
    """
    Draw each level of bernstein polynomial

    Arguments
    ---------------------------------------
        P: dim x n - matrix
            dim is the dimension of the points and n is number of points
        t: float
            time-step to be evaluated
    
    Returns
    ---------------------------------------
    None
    """

    styles = ["ro-", "bo-", "mo-", "ko-"]
    labels = ["linear", "Quadratic", "Cubic", "Quartic"]
    bezier, Pvecs = deCasteljau(P, t)
    degree, dim, n = Pvecs.shape

    for deg in range(degree-1):#For each degree
        vals = np.zeros((dim, n - deg))
        for p in range(n - deg - 1):#For each point in degree deg
            vals = Pvecs[deg,:,:n-deg]
            plt.plot(Pvecs[deg, 0, p], Pvecs[deg, 1, p], "ro-")
        
        plt.plot(vals[0], vals[1], styles[deg], label = labels[deg])
    
    plt.scatter(bezier[0], bezier[1], color = "forestgreen")

    return None

def draw_bezier_comp(P, A, t, title):
    """
    Draw the composite bezier curve, with the interpolation points

    Arguments
    ---------------------------------------
        P: m x dim x degree matrix
            m is the number of segments
            dim is the dimension of the problem
            degree is the number of points for each segment, the actual degree is one less
        t: float 
            Time step to be evaluated at
        title: string
            title of plot-figure
    Returns
    ---------------------------------------
    None
    """

    B = np.array([compositeBezier(P, t) for t in t])

    plt.plot(B[:,0], B[:,1], color = "red", label = "Bézier composition")
    plt.scatter(A[0], A[1], color = "forestgreen", label = "Interpolation points")
    plotCommands("x", "y", title, legend = True)

    return None

def draw_bezier(P, t):
    """
    Draw the composite bezier curve

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
    None
    """
    B = np.array([compositeBezier(P, t) for t in t])
    plt.plot(B[:,0], B[:,1], color = "red", label = "Bézier composition")
    plotCommands("x", "y", "Bézier Curve", legend = True)
    return None