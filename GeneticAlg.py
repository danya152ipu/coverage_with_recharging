import random
import sys
import pandas as pd
import operator
import numpy as np
import time
from matplotlib import pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    route = random.sample(cityList[1:], len(cityList)-1)
    return [cityList[0]] + route
def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['sel_prob'] = df.Fitness/df.Fitness.sum()
    inc_list = df.Index.tolist()
    prob_list = df.sel_prob.tolist()
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    selectionResults += [ int(idx) for idx in np.random.choice(np.array(inc_list),len(popRanked) - eliteSize,p = np.array(prob_list))]
    return selectionResults

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1[1:]))
    geneB = int(random.random() * len(parent1[1:]))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        childP1.append(parent1[1:][i])
    childP2 = [item for item in parent2[1:] if item not in childP1]
    if startGene == 0:
        child = [parent1[0]] + childP1 + childP2
    else:
        child = [parent1[0]] + childP2 + childP1
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))
    for i in range(0, eliteSize):
        children.append(matingpool[i])
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(1,len(individual)):
        rand_v = random.uniform(-1,1)
        if(abs(rand_v) < mutationRate):
            swapWith = swapped + int(rand_v * min(swapped - 1,len(individual) - swapped))
            if swapWith == 0:
              swapWith += 1
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual
def mutatePopulation(population,eliteSize, mutationRate):
    mutatedPop = []
    for elite_ind in range(eliteSize):
        mutatedPop.append(population[elite_ind])
    for ind in range(eliteSize, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children,eliteSize, mutationRate)
    return nextGeneration
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations,plot = False):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(Fitness(pop[rankRoutes(pop)[0][0]]).routeDistance()))
    if plot:
      progress = []
      progress.append(1 / rankRoutes(pop)[0][1])
    bestRouteL = sys.maxsize
    bestGen = 0
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        routeLength=Fitness(pop[rankRoutes(pop)[0][0]]).routeDistance()
        if routeLength < bestRouteL:
          bestRouteL = routeLength
          bestGen = i
        if plot:
          progress.append(routeLength)
    if plot:
      plt.plot(progress)
      plt.ylabel('Distance')
      plt.xlabel('Generation')
      plt.show()
    print(f"Final distance:{Fitness(pop[rankRoutes(pop)[0][0]]).routeDistance()} founnd at {bestGen} generation")
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute