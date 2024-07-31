import numpy as np
import pandas as pd
import random
from random import choices
import math
from operator import itemgetter
import itertools


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            l = l.rstrip().split()
            l = [int(coord) for coord in l]
            data.append(l)
    return data[0][0], data[1:]


def create_initial_pop(size, cities):
    # randomly generate the intial population
    initial_pop = [np.random.permutation(cities).tolist() for _ in range(size)]
    return initial_pop 


def fitness(path): # the higher the more optimal
    # Input: 
    #   path: a list of numpy vectors representing coordinates 
    # Output: the inverse of Eucleadian distance covered visiting the coordinates sequentially
    distance = 0
    
    last_stop = len(path) - 1 

    for i in range(last_stop):
        distance += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))

    # back to the starting point
    distance += np.linalg.norm(np.array(path[0]) - np.array(path[last_stop]))

    return 1/distance


def rank_fitness(population):
    # Takes a list of paths and return a sorted list based on fitness scores
    rank_list = []
    for i, individual in enumerate(population):
        fit_score = fitness(individual) 
        rank_list.append((i, fit_score))
        rank_list.sort(key=itemgetter(1), reverse=True)
    return rank_list # sorted(rank_list, key = lambda x: x[1], reverse=True) 


def select_parents(size, population, rank_list, top_k):
    # rank_list: a list of tuples of index and fitness scores sorted in descending order.
    # top_k: the best k routes we will retain unconditionally

    elite_parents = []

    for i in range(top_k):
        elite_parents.append(population[rank_list[i][0]])

    # get all combinations of the elite parents
    parents_list = [list(parents) for parents in list(itertools.combinations(elite_parents, 2))]

    # generate probability distribution for each fitness score
    denomitor = sum(score for i, score in rank_list)

    ranked_probs = [(i, score/denomitor) for i, score in rank_list]

    locs = list(map(lambda x: population[x[0]], ranked_probs))
    probs = list(map(lambda x: x[1], ranked_probs))


    # run the Roulette Wheel
    cnt = size - len(parents_list)
    while cnt <= size:
        parents = choices(locs, weights = probs, k=2)
        parents_list.append(parents)
        cnt += 2

    return parents_list


def resolve_conflict(cities, comparison_list):
    # check if comparison list visits all cities in cities exactly once
    
    cities = [tuple(city) for city in cities]
    comparison_list_copy = [tuple(city) for city in comparison_list]
    
    # get cities missing from the comparison list
    missed = iter(set(cities) - set(comparison_list_copy))

    # visited cities in the comparison list
    visited = set()

    for i, city in enumerate(comparison_list_copy):
        if city not in visited:
            visited.update({city})
        else:
            comparison_list[i] = list(next(missed))

    
    return comparison_list

def crossover(cities, parent1, parent2, start_index, end_index):

    child1 = parent1[0:start_index] + parent2[start_index:end_index+1] + parent1[end_index+1:len(parent1)]
    child2 = parent2[0:start_index] + parent1[start_index:end_index+1] + parent2[end_index+1:len(parent2)]

    child1 = resolve_conflict(cities, child1)# resolve conflict and duplication 
    child2 = resolve_conflict(cities, child2)

    return child1, child2


def mutate(path, mutate_prob):
    # take a path and mutate it with a mutation probability
    for i in range(len(path)): 
        fate = np.random.choice([0,1], p=[1-mutate_prob, mutate_prob])
    
        if fate == 1:
            new_loc = random.randint(0, len(path)-1)
            
            city1 = path[i]
            city2 = path[new_loc]
            
            path[i] = city2
            path[new_loc] = city1
            
    return path


if __name__ == '__main__':
    
    # parse data

    num_cities, cities = read_data('input.txt')

    size = 1000

    parent_size = math.ceil(size/2)

    start_index = math.floor(size/2)
    end_index = math.floor(size)

    MUTATE_PROB = 0.03
    TOP_K = 10

    population = create_initial_pop(size, cities)

    max_gen = max(50, (1/num_cities) * 7500)

    rank_list = rank_fitness(population)
    max_fitness = rank_list[0][1]
    best_path = population[rank_list[0][0]]

    gen = 0
    while gen < max_gen:
        
        new_population = []
        
                
        parents_list = select_parents(parent_size, population, rank_list, TOP_K)


        for parents in parents_list:
            new_population.append(parents[0])
            new_population.append(parents[1])

            child1, child2 = crossover(cities, parents[0], parents[1], start_index, end_index)
            child1 = mutate(child1, MUTATE_PROB)
            child2 = mutate(child2, MUTATE_PROB)
            new_population.append(child1)
            new_population.append(child2)


        population = new_population 

        rank_list = rank_fitness(population)

        if rank_list[0][1] > max_fitness:
            max_fitness = rank_list[0][1]
            best_path = population[rank_list[0][0]]

        gen += 1


    with open('output.txt', 'w') as f:
            for city in best_path + [best_path[0]]:
                f.write(" ".join([str(coord) for coord in city]))
                f.write("\n")


    
