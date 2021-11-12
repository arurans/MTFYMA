from newtonFractals import *
from functions import *
from time import time

n = 2048

def main():
    start = time()
    NF_maxiter15 = newtonFractal(g, Dg, n)
    NF_maxiter15.newtons_method(15, r"$z^5 = 1$", "Plots/NF Z^5 15")
    NF_maxiter25 = newtonFractal(g, Dg, n)
    NF_maxiter25.newtons_method(25, r"$z^5 = 1$", "Plots/NF Z^5 25")
    NF_maxiter40 = newtonFractal(g, Dg, n)
    NF_maxiter40.newtons_method(40, r"$z^5 = 1$", "Plots/NF Z^5 40")
    print("Execution time: ", time() - start)


if __name__ == "__main__":
    main()