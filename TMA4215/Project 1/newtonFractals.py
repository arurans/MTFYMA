import numpy as np
from matplotlib import pyplot as plt
from numba import njit as superspeed

class newtonFractal:

    def __init__(self, f, Df, n, x_start = -1, x_end = 1):

        """
        Initilize for computing newton Fractals.
        
        Parameters
        ---------------------------------
        f: fnc
            my_newton is applied on f
        Df: fnc
            derivative of f
        n: int
            n^2 is the number of total numbers we are going to simulate from the complex plane
        x_start: float
            complex plane start
        x_end: float
            complex plane end
        Returns
        ---------------------------------
        x: numpy-array
            root(s) of the problem with error < tolerance if maxiter was not exceeded
        i+1: int
            Number of iterations 
        
        """
        x = np.linspace(x_start, x_end, n)
        X, Y = np.meshgrid(x, x*1j)
        self.n = n
        self.x0 = X + Y#Complex plane as a grid of initial conditions
        self.f = f
        self.Df = Df

        return None


    def newtons_method(self, maxiter, fnc, path, tol = 0):
        self.maxiter = maxiter
        """
        Newton iteration of a function f, and save to image
        
        Parameters
        ---------------------------------
        maxiter: int
                maximum number of iterations (5000 default)
        fnc: string
            equation being solved
        path: string
            filepath to save image
        tol: float
            error tolerance of method (0 default)
                    
        Returns
        ---------------------------------
        x: numpy-array
            root(s) of the problem with error < tolerance if maxiter was not exceeded
        i+1: int
            Number of iterations 
        
        """
        
        #Assigned to variables to compute it once per iteration
        fx = self.f(self.x0)
        Dfx = self.Df(self.x0)

        A = self.x0.copy()
        
        for i in range(maxiter):
            if (np.abs(fx) < tol).all():#Check if all elements are below the tolerance
                break
            A = A - np.divide(fx,Dfx)#Able to divide numpy arrays element-wise
            fx = self.f(A)
            Dfx = self.Df(A)

        self.A = A
        self.iterations = i + 1

        A_plot = np.angle(self.A, deg = True) + 180
        plt.imshow(A_plot, cmap= "gnuplot2", vmin = 0, vmax = 360)
        title = f"Gridsize: {self.n}x{self.n}" + " Convergence of " + fnc + f" Max iterations: {maxiter}"
        plt.title(title)
        plt.savefig(path)
        return None

    def get_solutions(self):
        #Return problem solutions, and the number of iterations
        return self.A, self.iterations