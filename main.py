import random
import time
from multiprocessing import Pool
import numpy as np
from prettytable import PrettyTable

##############################################
# Esta funcion lee el archivo de la instancia
# INPUT = nombre del archivo de instancia
# OUTPUT = cantidad total de stands, lista con el tamaño de cada stand, matriz de esfuerzo
##############################################
def extract_info(filename):
    instance = open(filename, "r")
    nStands = int(instance.readline().strip())
    dStand = [int(x) for x in instance.readline().strip().split(",")]
    lines = instance.read().strip().split("\n")
    matrix = []

    for line in lines:
        matrix.append([int(x) for x in line.split(",")])
    return nStands, dStand, matrix


##############################################
# deprecated, esta funcion fue cambiada por una similar al calculo en el codigo java entregado
# Esta funcion retorna la distancia entre 2 puntos dado un orden
# INPUT = numero de puesto 1, numero de puesto 2, orden, lista de tamaños
# OUTPUT = distancia entre el punto 1 y 2
##############################################
def distance(startStand, endStand, order, dStand):
    dist = dStand[startStand] / 2
    for x in range(order.index(startStand) + 1, order.index(endStand)):
        dist = dist + dStand[order[x]]
    dist = dist + dStand[endStand] / 2
    return dist


##############################################
# esta funcion usaba distance, pero fue actualizada por un metodo mas rapido basado en el codigo java entregado
# Esta funcion retorna el fitness de la solucion
# INPUT = orden de la solucion, matriz de ezfuerzo, la lista de distancias
# OUTPUT = distancia entre el punto 1 y 2
##############################################
def fitness(order, matrix, dStand):
    fitness = 0
    for x in order:
        middleDist = 0
        for y in order[order.index(x) + 1 : len(order)]:
            # print("x es ",x,", y es ",y,", la distancia es ", distance(x, y, order, dStand), ", numero magico:",matrix[x][y])
            # fitness = fitness + matrix[x][y] * distance(x, y, order, dStand)
            fitness = fitness + matrix[x][y] * (
                dStand[x] / 2 + middleDist + dStand[y] / 2
            )
            middleDist = middleDist + dStand[y]
        # print(dist)
    # print(str(order) + " tiene un fitness de :" + str(fitness))
    # print(dist)
    return fitness


##############################################
# Estan funcion genera soluciones usando el metodo del vecino mas cercano y con un punto de inicio aleatorio
# INPUT = cantidad total de puestos, la lista de distancias de los puestos
# OUTPUT = distancia entre el punto 1 y 2
##############################################
def genNearestNeighbour(nStand, dStand, matrix):
    availableStands = list(range(0, nStand))
    random.shuffle(availableStands)
    order = [availableStands.pop()]
    while len(order) < nStand:
        c = 1000000000
        next = -1
        for i in availableStands:
            if abs(dStand[i] + dStand[order[0]]) / 2 < c:
                next = i
                c = abs(dStand[i] + dStand[order[0]]) / 2
        availableStands.remove(next)
        order.insert(0, next)
    order.reverse()
    return (fitness(order, matrix, dStand), tuple(order))


##############################################
# Esta funcion cruza 2 soluciones usando el algorimo de cruzamiento ordenado visto en clase
# INPUT = orden de solucion 1 y orden de solucion 2
# OUTPUT = hijo resultante 1 e hijo resultante 2
##############################################
def crossover(order1, order2, matrix, dStand):
    # print("this is P1" + str(order1))
    # print("this is P2" + str(order2))
    points = [random.randint(0, len(order1)), random.randint(0, len(order1))]
    # points.append(random.randint(0,len(order1)))
    order1 = list(order1[1])
    order2 = list(order2[1])

    s1 = ["x"] * len(order1)
    s2 = ["x"] * len(order1)
    for x in range(min(points), max(points)):
        s2[x] = order2[x]
        s1[x] = order1[x]
    for x in s1[min(points) : max(points)]:
        order2.remove(x)
    for x in s2[min(points) : max(points)]:
        order1.remove(x)
    for x in range(min(points)):
        s1[x] = order2.pop(0)
        s2[x] = order1.pop(0)
    for x in range(max(points), len(s1)):
        s1[x] = order2.pop(0)
        s2[x] = order1.pop(0)
    # print("this is H1" + str(s1))
    # print("this is H2" + str(s2))
    return (fitness(s1, matrix, dStand), tuple(s1)), (
        fitness(s2, matrix, dStand),
        tuple(s2),
    )


def mutacion(order, prob, matrix, dStand):
    # si el numero random es menor la probabilidad no muta y se devuelve el orden
    # sin mutacion
    if random.random() > prob:
        return order
    order = list(order[1])
    points = [random.randint(0, len(order) - 1), random.randint(0, len(order) - 1)]
    order[points[0]], order[points[1]] = order[points[1]], order[points[0]]
    return (fitness(order, matrix, dStand), tuple(order))


def mutacionPlus(order, prob, matrix, dStand):
    # si el numero random es menor la probabilidad no muta y se devuelve el orden
    # sin mutacion
    order = list(order[1])
    for x in range(len(order)):
        if random.random() < prob:
            y = random.randint(0, len(order) - 1)
            # print("index changed: x=" + str(x) + " y="+ str(y))
            order[x], order[y] = order[y], order[x]
    return (fitness(order, matrix, dStand), tuple(order))


def roulette(solutions, count):
    solList = []
    fitList = []
    selected = []
    for x in solutions:
        # nota: el la probabilidad estara representada, pero las soluciones
        # no estaran en orden.
        fitList.append(1 / x[0])
        solList.append(x[1])
    for y in range(count):
        # Fran arregla esta wea cuando empieces
        solutionSelected = random.choices(solList, fitList)
        # print(solutionSelected)
        tempIndex = solList.index(solutionSelected[0])
        selected.append((fitList[tempIndex], solutionSelected[0]))
        solList.pop(tempIndex)
        fitList.pop(tempIndex)
    return selected


def bestSolutions(solutions, count):
    afitness = lambda x: x[0]
    return sorted(solutions, key=afitness, reverse=False)[0:count]


# worker=0, mutationProb=0.1, nHijos=100, generationLimit=200
def simulate(instance, configs, worker=0):
    print("starting ", worker)
    nStand, dStand, matrix = instance["nStand"], instance["dStand"], instance["matrix"]
    mutationProb, nHijos, generationLimit = configs["mutationProb"], configs["generationLimit"], configs["nHijos"]

    solutions = {tuple(genNearestNeighbour(nStand, dStand, matrix))}
    while len(solutions) < 100:
        solutions.add(tuple(genNearestNeighbour(nStand, dStand, matrix)))
    solutions = list(bestSolutions(solutions, 50))

    stopTime = time.time() + 1
    generation = 0

    # while time.time() < stopTime:
    while generation < generationLimit:
        generatedSolutions = list()
        while len(generatedSolutions) < nHijos:
            p1, p2 = roulette(solutions, 2)
            temp1, temp2 = crossover(p1, p2, matrix, dStand)
            generatedSolutions.append(mutacionPlus(temp1, mutationProb, matrix, dStand))
            generatedSolutions.append(mutacionPlus(temp2, mutationProb, matrix, dStand))
        solutions = bestSolutions((solutions + generatedSolutions), 50)
        print(
            "worker ",
            worker,
            "generation " + str(generation),
            " mejor fitness : ",
            solutions[0][0],
            " con una solucion de: ",
            solutions[0][1],
        )
        solutions = list(solutions)
        generation = generation + 1
    return worker, solutions[0][0]


def main(instance, configs):
    table = PrettyTable()
    tableStats = PrettyTable()
    table.field_names = ["ejecucion", "f(x)"]
    tableStats.field_names = ["stat", "value"]

    runs = 10
    print("start")
    startTime = time.time()
    temp = [(instance, configs, worker)for worker in range(runs)]
    #print(temp[1])
    #print(simulate(temp[0][0], temp[0][1], temp[0][2] ))
    with Pool() as p:
       data = p.starmap(simulate, [(instance, configs, worker)for worker in range(runs)])
    table.add_rows(data)
    fitList = [x[1] for x in data]
    tableStats.add_row(["media", np.mean(fitList)])
    tableStats.add_row(["desviacion estandar", np.std(fitList)])
    tableStats.add_row(["T ejecucion", round(time.time() - startTime, 2)])
    print(table)
    print(tableStats)


if __name__ == "__main__":
    filename = "Instancias/QAP_sko56_04_n"
    nStand, dStand, matrix = extract_info(filename)

    instance = {"nStand": nStand, "dStand": tuple(dStand), "matrix": matrix}
    #  print(data["matrix"])
    configs = {"generationLimit": 200, "mutationProb": 0.1, "nHijos": 100}

    main(instance, configs)
