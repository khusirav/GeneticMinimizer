from cmath import pi
import random as rnd
from statistics import mean
from time import time
import numpy as np
import matplotlib.pyplot as plt
import math

def fcnTest(var):
    return 15-(var-15)**2 + 0.007*(var-15)**4 + 2*var

def fcn1(var):
    return math.sin(var) + math.sin(var**2)

def fcn2(vars):
    return (vars[1] - vars[0]**2 * 5.1/(4*pi**2)+vars[0] * 5/pi - 6)**2 + 10*(1-1/(8*pi))*math.cos(vars[0])+10

def rateMinAndSortDesc(population : np.ndarray, fitnessFunction):
    
    fitnessVals = {i : fitnessFunction(population[i]) for i in range(population.shape[0])}
    fitnessVals = np.asarray(sorted(fitnessVals.items(), key = lambda item: item[1]), dtype = np.float64)
    fitnessVals[:, 1] = fitnessVals[-1, 1] - fitnessVals[:, 1]
 
    return fitnessVals

def fitnessToProbabilities(fitnessValues: np.ndarray):

    if fitnessValues[0, 1] == 0:
        fitnessValues[:, 1] = fitnessValues[:, 1] + 1
    fitnessValues[:, 1] = np.divide(fitnessValues[:, 1], (np.sum(fitnessValues[:, 1])))
    return fitnessValues

def uniform(population, fintnessFunction, parentsQuantity):
    
    #исключения
    if population.shape[0] < parentsQuantity:
        print('uniform selection method: Invalid parents quantity or population size')
        return[]

    if population.shape[0] == parentsQuantity:
        return np.asarray(population, dtype = np.float64)

    #выбор
    parents = []
    for i in range(parentsQuantity):
        j = rnd.randint(0, population.shape[0] - 1)
        parents.append(population[j])
        population = np.delete(population, j, axis = 0)

    return np.asarray(parents)

def roulette(population: np.ndarray, fitnessFunction, parentsQuantity):
    
    #исключения
    if population.shape[0] < parentsQuantity:
        print('Roulette selection method: Invalid parents quantity or population size')
        return None

    if population.shape[0] == parentsQuantity:
        return np.asarray(population, dtype = np.float64)

    #одномерный случай
    if len(population.shape) == 1:
        parents = np.zeros(parentsQuantity)
    #n-мерный случай
    else:   
        parents = np.zeros([parentsQuantity, population.shape[1]])

    for i in range(parentsQuantity):
        #оценка особей и получения вероятности выбора каждой в зависимости от значения функции приспособленности
        fitnessVals = rateMinAndSortDesc(population, fitnessFunction)
        probabilities = fitnessToProbabilities(fitnessVals)
        
        #выбор
        temp = rnd.random()
        for parent_index, probability in probabilities:
            if temp < probability:
                temp = int(parent_index)
                break
            else:
                temp = temp - probability

        parents[i] = population[temp]
        population = np.delete(population, temp, axis=0)

    return parents

def stochasticUniform(population: np.array, fitnessFunction, parentsQuantity):

    #одномерный случай
    if len(population.shape) == 1:
        parents = np.zeros(parentsQuantity)
    #n-мерный случай
    else:   
        parents = np.zeros([parentsQuantity, population.shape[1]])

    #оценка особей и получения вероятности выбора каждой в зависимости от значения функции приспособленности
    fitnessVals = rateMinAndSortDesc(population, fitnessFunction)
    probabilities = fitnessToProbabilities(fitnessVals)
    
    #разметка "рулетки"
    step = 1/parentsQuantity
    start = rnd.random()*step
    marks = []
    for i in range(parentsQuantity):
        marks.append(start + i*step) 

    #выбор
    for i in range(len(marks)):
        for parent_index, probability in probabilities:
            if marks[i] < probability:
                parents[i] = population[int(parent_index)]
                break
            else:
                marks[i] = marks[i] - probability
    
    return parents
    

def crossingoverTwoParentsRandom(parents: np.ndarray):

    #одномерный случай - кроссинговер представляет собой обмен двоичными разрядами
    if len(parents.shape) == 1:

        if parents[0] >= 0:
            signP1 = 1
        else:
            signP1 = -1

        if parents[1] >= 0:
            signP2 = 1
        else:
            signP2 = -1

        parent1str = str(bin(round(abs(parents[0]))))[2:]
        parent2str = str(bin(round(abs(parents[1]))))[2:] 
        if len(parent1str) > len(parent2str):
            while len(parent1str) > len(parent2str):
                parent2str = '0' + parent2str
        else:
            while len(parent2str) > len(parent1str):
                parent1str = '0' + parent1str

        if len(parent1str) == 1:
            return np.asarray((float(int(parent1str, 2)*signP1), float(int(parent2str, 2)*signP2)))

        crossingoverPosition = rnd.randint(1, len(parent1str) - 1)
        temp = parent1str
        parent1str = parent1str[0:crossingoverPosition] + parent2str[crossingoverPosition:]
        parent2str = parent2str[0:crossingoverPosition] + temp[crossingoverPosition:]

        return np.asarray((int(parent1str, 2)*signP1, int(parent2str, 2)*signP2), dtype = np.float64)
    
    #n-мерный случай - кроссинговер представляет собой обмен генами
    else:
        crossingoverPosition = rnd.randint(1, parents.shape[1] - 1)
        children = np.array([np.hstack((parents[0][:crossingoverPosition], parents[1][crossingoverPosition:])), np.hstack((parents[1][:crossingoverPosition], parents[0][crossingoverPosition:]))])
        return np.asarray(children, dtype = np.float64)

def gaussianMutator(children: np.ndarray, initialRangeL, initialRangeR, scale, shrink):
 
    if len(children.shape) == 1:
        for i in range(children.shape[0]):
            children[i] = children[i] + rnd.gauss(0, scale*(1-shrink)*(initialRangeR - initialRangeL))
        return children
    else:
        for i in range(children.shape[0]):
            geneNumber = rnd.randint(0, children.shape[1] - 1)
            children[i][geneNumber] = children[i][geneNumber] + rnd.gauss(0, scale*(1-shrink)*(initialRangeR - initialRangeL))
        return children 

def geneticMinimiser(population: np.ndarray, fitnessFunction, selectionMethod, crossingoverProbability, mutationProbability, generationsQuantity, fitnessLimit, initialRangeL, initialRangeR, mutatorScale, mutatorShrink, stallGenerations):

    generationNumber = 0
    currentFitness = fitnessLimit + 1
    initialPopulation = population
    shrinkStep = mutatorShrink/generationsQuantity
    currentShrink = 0
    stall = 0
    minFitness = []
    meanFitness = []

    meanFitness.append(mean([fitnessFunction(i) for i in population]))
    minFitness.append(min([fitnessFunction(i) for i in population]))
    
    

    while generationNumber < generationsQuantity and currentFitness > fitnessLimit and stall < stallGenerations:
        
        initialPopulationSize = initialPopulation.shape[0]
        newPopulation = []
        while(len(newPopulation) < initialPopulationSize):
            
            parents = selectionMethod(initialPopulation, fitnessFunction, 2)
            if rnd.random() <= crossingoverProbability:
                parents = crossingoverTwoParentsRandom(parents)
            if rnd.random() <= mutationProbability:
                parents = gaussianMutator(parents, initialRangeL, initialRangeR, mutatorScale, currentShrink)
            
            if len(parents.shape) == 1:
                if parents[0] > initialRangeR:
                    parents[0] = initialRangeR
                if parents[0] < initialRangeL:
                    parents[0] = initialRangeL
            
                if parents[1] > initialRangeR:
                    parents[1] = initialRangeR
                if parents[1] < initialRangeL:
                    parents[1] = initialRangeL
            else:
                for par in range(parents.shape[1]):
                    if parents[0][par] > initialRangeR:
                        parents[0][par] = initialRangeR
                    if parents[0][par] < initialRangeL:
                        parents[0][par] = initialRangeL
                    if parents[1][par] > initialRangeR:
                        parents[1][par] = initialRangeR
                    if parents[1][par] < initialRangeL:
                        parents[1][par] = initialRangeL
            
            currentShrink = currentShrink + shrinkStep
            newPopulation.append(parents[0])
            if initialPopulationSize - len(newPopulation) >= 1:
                newPopulation.append(parents[1])

        prevFitness = min([fitnessFunction(i) for i in initialPopulation])   
        initialPopulation = np.asarray(newPopulation, dtype = np.float64) 
        currentFitness = min([fitnessFunction(i) for i in initialPopulation])
        minFitness.append(currentFitness)
        meanFitness.append(mean([fitnessFunction(i) for i in initialPopulation]))
        if currentFitness >= prevFitness:
            stall = stall + 1
        else:
            stall = 0

        generationNumber = generationNumber + 1

    if currentFitness < fitnessLimit:
        print('Algorithm stopped by fitness limit')
    elif stall >= stallGenerations:
        print('Algorithm stopped by stall generations')
    else:
        print('Algorithm stopped by iterations limit')

    print('Iterations number: ', generationNumber)
    return initialPopulation, meanFitness, minFitness, generationNumber

truemin = 0.397887 #настоящий минимум функции
runs = 1 #количество запусков
initR = 5 #правая граница интервала поиска минимума
initL = 0 #левая граница интервала поиска минимума
Shrink = 0.5 #степень уменьшения "Гауссова колокола" за все итерации
Scale = 0.03 #степень уменьшения "Гауссова колокола" за все итерации
PopulationSize = 10 #размер популяции
IterationsQuantity = 100 #максимальное количество итераций 
stallGenerations = 5 #максимальное количество поколений без положительных изменений в фитнесс-функции при котором алгоритм не будет остановлен  
selectionMethod = roulette #метод выбора особей для скрещивания 
foundxylst = []
foundzlst = []
iterslst = [] 
ts1 = time()

for j in range(runs):

    x = np.arange(initL, initR, (initR-initL)*0.01)
    y = np.arange(initL, initR, (initR-initL)*0.01)
    xgrid, ygrid = np.meshgrid(x, y)
    f_2 = (ygrid - xgrid**2 * 5.1/(4*pi**2)+ xgrid * 5/pi - 6)**2 + 10*(1-1/(8*pi))*np.cos(xgrid)+10
    plt.contour(xgrid, ygrid, f_2)
    x = np.random.uniform(initL, initR, (PopulationSize, 2))
    newpop, meanFitness, minFitness, iters = geneticMinimiser(x, fcn2, selectionMethod, 0.3, 0.5, IterationsQuantity, truemin*0.5, initL, initR, Scale, Shrink, stallGenerations)
    iterslst.append(iters)
    newpopy = []
    for i in newpop:
        newpopy.append(fcn2(i))
    for i in range(len(newpop)):
        plt.scatter(newpop[i][0], newpop[i][1])
    plt.show()
    plt.plot(range(len(minFitness)), minFitness, color = 'red', label = 'min fitness')
    plt.plot(range(len(meanFitness)), meanFitness, color = 'blue', label = 'mean fitness')
    plt.legend()
    plt.show()
    fitnessVals = {i : fcn2(newpop[i]) for i in range(newpop.shape[0])}
    fitnessVals = np.asarray(sorted(fitnessVals.items(), key = lambda item: item[1]), dtype = np.float64)
    minz = fitnessVals[0][1]
    minxy = newpop[int(fitnessVals[0][0])]
    print('minz = ', minz)
    foundzlst.append(minz)
    print('minxy = ', minxy)
    foundxylst.append(list(minxy))

#средние показатели за runs итераций
ts2 = time()
print('x1 = ', round(mean(foundxylst[:][0]), 2))
print('x2 = ', round(mean(foundxylst[:][1]), 2))
print('f(X) = ', round(mean(foundzlst), 4))
print('Iters = ', mean(iterslst))
print('time = ', round((ts2 - ts1)/runs, 2),'s')
