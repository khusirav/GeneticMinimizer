# GeneticMinimizer
Simple genetic algorithm implementation for finding function minimum

## Installation (Windows)

Download the repo and open it in cmd.

Create python virtual environment using venv:
```
$ python -m venv genetic_venv
```
Activate created environment:
```
$ genetic_venv\scripts\activate
```
Install python libs from requirements.txt:
```
$ pip install -r requirements.txt
```
Run geneticMinimizer:
```
$(genetic_venv) geneticMinimier.py
```

## Usage
Define function using python:
``` python
def fcn2(vars):
    return (vars[1] - vars[0]**2 * 5.1/(4*pi**2)+vars[0] * 5/pi - 6)**2 + 10*(1-1/(8*pi))*math.cos(vars[0])+10
```

Set the algorithm parameters:
``` python
truemin = 0.397887 #настоящий минимум функции (только для определённого критерия останова)
runs = 1 #количество запусков
initR = 5 #правая граница интервала поиска минимума
initL = 0 #левая граница интервала поиска минимума
Shrink = 0.5 #степень уменьшения "Гауссова колокола" за все итерации
Scale = 0.03 #степень уменьшения "Гауссова колокола" за все итерации
PopulationSize = 10 #размер популяции
IterationsQuantity = 100 #максимальное количество итераций 
stallGenerations = 5 #максимальное количество поколений без положительных изменений в фитнесс-функции при котором алгоритм не будет остановлен  
selectionMethod = roulette #метод выбора особей для скрещивания 
```
Run cycle: 
``` python
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
```

Print the data:
```python
#средние показатели за runs итераций
ts2 = time()
print('x1 = ', round(mean(foundxylst[:][0]), 2))
print('x2 = ', round(mean(foundxylst[:][1]), 2))
print('f(X) = ', round(mean(foundzlst), 4))
print('Iters = ', mean(iterslst))
print('time = ', round((ts2 - ts1)/runs, 2),'s')
```
