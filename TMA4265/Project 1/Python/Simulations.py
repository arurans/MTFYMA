import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from plot1 import *

class Population:

    def __init__(self, Y0, population, alpha = 0.005, gamma = 0.10):
        self.population = population #Population count
        self.Y0 = Y0 #Initial condition for the simulation of the form Y = [Susceptible, Infected, Recovered] at time 0
        self.alpha = alpha #probability of recovered individual getting susceptible
        self.gamma = gamma #probability of infected individual becoming recovered

    def __plot_population(self, Y, days, path):
        #Plot one simulation of a population days number of days
        x = np.arange(days+1)
        #x_axis = np.arange(0, days+1, 50)
        #y = np.arange(0, self.population + 1, 100)
        plt.scatter(x, Y[0],marker='o',color='blue')
        plt.scatter(x, Y[1],marker='o',color='red')
        plt.scatter(x, Y[2],marker='o',color='green')
        #plt.xticks(x_axis, x_axis, size = 20)
        #plt.yticks(y, y, size = 20)
        plt.legend(["Susceptible","Infected","Recovered"])#"Expected distribution",
        plotCommands("Days", "Number of individuals", "Number of susceptible, infected and recovered for " + f"{days}" + " days", path)

        return None

    def simulate_population(self, days, path):
        #Simulate disease in population for days number of days, and save plot
        Y = np.zeros((self.Y0.size, days + 1), dtype = np.intc)
        Y[:,0] = self.Y0

        for i in range (1, days + 1):
            BinS = np.random.binomial(Y[0,i-1],0.5*Y[1,i-1]/self.population)#Binomial distribution of susceptible individuals becoming infected
            BinI = np.random.binomial(Y[1,i-1], self.gamma)#Binomial distribution of infected individuals becoming recovered
            BinR = np.random.binomial(Y[2,i-1], self.alpha)#Binomial distribution of recovered individuals becoming suscaptible
            Y[0,i] = Y[0,i-1] + (BinR-BinS)
            Y[1,i] = Y[1,i-1] + (BinS-BinI)
            Y[2,i] = Y[2,i-1] + (BinI-BinR)

        self.__plot_population(Y, days, path)

        return None

    def introduce_infected(self, prob, Y_infected):

        return None






@jit(nopython = True)
def simulate_individual(P, X0 = 0, n = 7300):
    """
    Simulates an individual for n days.
    
    Arguments
    ---------------------------------------
    P: NxN square matrix
        Transition probability matrix
    X0: int
        Initial state (0, 1 or 2)
    n: int
        Number of days to be simulated
        
    Returns
    ---------------------------------------
    pi: 1x3 array (1x(number of states))
        Simulated long-run mean
    """
    
    state = np.zeros(n, dtype = np.intc)#The state (either 0, 1 or 2) for all times
    state[0] = int(X0)#Initial state
    
    
    random = np.random.random(n)
    for i in range(1, n):
        #Find the transition probabilities from state i-1
        #Then find the first element in array greater than random, index determines state i
        state[i] = np.where(P[state[i-1]]>random[i])[0][0] 
        
    last_half = n//2#We are only interested in estimating based on the last half
    state_last_half = state[last_half:]
    last_half = np.float64(last_half)
    pi_0 = (state_last_half == 0).sum()#Count number of state 0 in the last half of array
    pi_1 = (state_last_half == 1).sum()
    pi_2 = (state_last_half == 2).sum()
        
    pi = np.array([pi_0, pi_1, pi_2]).T/last_half #Compute estimated long mean run
    
    return pi

def simulate_multiple_times(P, X0 = 0, M = 30):
    """
    Simulates an individual M times.
    
    Arguments
    ---------------------------------------
    P: NxN square matrix
        Transition probability matrix
    X0: int
        Initial state (0, 1 or 2)
    M: int
        Number of times to be simulated
        
    Returns
    ---------------------------------------
    pi: 1x3 array (number of states)
        Average long-run mean (expectation value)
    std: 1x3 array (number of states)
        Standard deviation of each pi_i
    """
    
    pi_estimates = np.zeros((3, M))
    for i in range(M):
        pi_estimates[:,i] = simulate_individual(P)
    
    #Calculate average of every pi_i
    pi = np.array([np.average(pi_estimates[0]), np.average(pi_estimates[1]), np.average(pi_estimates[2])])
    #Compute the variance of each pi_i
    std = np.sqrt(np.array([np.var(pi_estimates[0]), np.var(pi_estimates[1]), np.var(pi_estimates[2])])/(M-1))
    
    return pi, std

@jit(nopython = True)
def simulate_Y(Y0, alpha = 0.005, gamma = 0.10, n = 300, N = 1000):
    """
    Simulates N individuals over n time-steps.
    
    Arguments
    ---------------------------------------
    Y0: numpy-array with length 3
        Initial condition for the simulation of the form Y = [Susceptible, Infected, Recovered] at time 0
    alpha: float
        probability of recovered individual getting susceptible, default 0.005
    gamma: float
        probability of infected individual becoming recovered, default 0.10
    n: int
        Number of time-steps, default 300
    N: int
        Number of individuals in the population, default 1000
        
    Returns
    ---------------------------------------
    Y: 3x(n+1) matrix
        Number of susceptible, infected and recovered individuals at each time step
    """
    Y = np.zeros((Y0.size, n + 1), dtype = np.intc)
    Y[:,0] = Y0

    for i in range (1, n + 1):
        BinS = np.random.binomial(Y[0,i-1],0.5*Y[1,i-1]/N)#Binomial distribution of susceptible individuals becoming infected
        BinI = np.random.binomial(Y[1,i-1],gamma)#Binomial distribution of infected individuals becoming recovered
        BinR = np.random.binomial(Y[2,i-1],alpha)#Binomial distribution of recovered individuals becoming suscaptible
        Y[0,i] = Y[0,i-1] + (BinR-BinS)
        Y[1,i] = Y[1,i-1] + (BinS-BinI)
        Y[2,i] = Y[2,i-1] + (BinI-BinR)
    return Y