from time import time
from Simulations import *

days = 1000
population = 1000000
number_of_cities = 10

alpha = 0.005
beta = 0.01
gamma = 0.10


P_modified = np.array([
    [1 - beta, 1, 0],
    [0, 1-gamma, 1],
    [alpha, 0, 1]
])

#Simulate infected population for one population once

cities = []
for i in range(number_of_cities):
    susceptible = np.random.randint(700000,population)
    infected = population - susceptible
    Y0 = np.array([susceptible, infected, 0])

    cities.append(Population(Y0, population))

def main():
    start = time()
    counter = 0 
    for city in cities:
        counter += 1
        city.simulate_population(days, "populationSimulation" + f"{counter}")
    print("Execution time: ", time() - start)


if __name__ == "__main__":
    main()