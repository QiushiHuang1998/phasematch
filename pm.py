import random
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from multiprocessing import Pool
from pathlib import Path
import subprocess
from linecache import getline

def get_file_lengths(files):
    lens = []
    for file in files:
        output = subprocess.check_output(f"wc -l {file}" ,shell=True)
        l = int(output.decode('utf-8').split()[0])
        lens.append(int(l))
    return lens



init_poscar1 = Structure.from_file("POSCAR1")
init_poscar2 = Structure.from_file("POSCAR2")

# define the number of processors
proc = 10
# define the range of values for each dimension
DIM_RANGE = (-2, 3)

# define the number of dimensions
DIMENSIONS = 18

# define the population size and number of generations
POP_SIZE = 10000000
ITER_SIZE = 100
NUM_GENERATIONS = 2000

# define the crossover and mutation probabilities

SUBSTITUTE_NUMBER = 0

det_folder = "det"


det_files = list(Path(det_folder).glob("det_*.txt"))
lengths_det_files = get_file_lengths(det_files)
num_det_files = len(lengths_det_files)


def generate_individual1():
    file_id = random.randint(0,num_det_files-1)
    det_file = str(det_files[file_id])
    individual = []
    for _ in range(2):
        num = (random.randint(0, (lengths_det_files[file_id]-1)//3) * 3) + 1
        for j in range(3):
            linelist = getline(det_file, num + j).strip().split()
            for e in linelist:
                individual.append(int(e))
    return individual

def generate_individual():
    while True:
        individual = generate_individual1()
        individual_hash = hash(tuple(individual))
        if individual_hash not in individual_set:
            individual_set.add(individual_hash)
            return individual

individual_set = set()

def generate_population(size):
    """
    Generate a population of individuals with the given size.
    """
    return [generate_individual() for _ in range(size)]


def init_clean_up(population, iteration_size):
    """
    Generate a population of individuals with the given size.
    """
    p = Pool(proc)
    fitnesses = p.map(eval_fitness, population)
    print(2)
    p.close()
    p.join()
    sort_fit = sorted(fitnesses)
    least_fit_indices = [fitnesses.index(value) for value in sort_fit[0:iteration_size]]
    new = []
    for ind in least_fit_indices:
        new.append(population[ind])
    return new


def eval_fitness(individual):
    a1, a2, a3, b1, b2, b3, c1, c2, c3, a4, a5, a6, b4, b5, b6, c4, c5, c6 = individual
    matrix1 = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
    matrix2 = np.array([[a4, a5, a6], [b4, b5, b6], [c4, c5, c6]])

    # make determinant of both transformation matrices equal
    if round(np.abs(np.linalg.det(matrix1)), 8) != round(np.abs(np.linalg.det(matrix2)), 8):
        return 100000

    lattice1 = np.array(init_poscar1.as_dict()["lattice"]["matrix"])
    lattice2 = np.array(init_poscar2.as_dict()["lattice"]["matrix"])
    transformed_lattice1 = np.dot(matrix1, lattice1)
    transformed_lattice2 = np.dot(matrix2, lattice2)
    length1 = np.linalg.norm(transformed_lattice1, axis=1)
    length2 = np.linalg.norm(transformed_lattice2, axis=1)
    if np.abs(np.linalg.det(matrix1))<=0.01 or np.abs(np.linalg.det(matrix2))<=0.01:
        return 10000000
    
    # rearrange the positions of basis vectors in the cell according to the length of each basis vector
    combine1 = []
    for i in range(3):
        combine1.append((length1[i], tuple(transformed_lattice1[i]), tuple(matrix1[i])))
    combine1 = sorted(combine1, key=lambda x: x[0])
    for i in range(3):
        transformed_lattice1[i] = np.array(list(combine1[i][1]))
        matrix1[i] = np.array(list(combine1[i][2]))
        length1[i] = combine1[i][0]
    combine2 = []
    for i in range(3):
        combine2.append((length2[i], tuple(transformed_lattice2[i]), tuple(matrix2[i])))
    combine2 = sorted(combine2, key=lambda x: x[0])
    for i in range(3):
        transformed_lattice2[i] = np.array(list(combine2[i][1]))
        matrix2[i] = np.array(list(combine2[i][2]))
        length2[i] = combine2[i][0]

    # calculate the length change after the transformation
    length_diff = np.abs(length1 - length2)
    length_ratio = length_diff

    # calculate the angle changes after the transformation
    angs = []
    for i in range(3):
        if i < 2:
            cos_angle = np.abs(np.dot(transformed_lattice1[i], transformed_lattice1[i + 1])) / (
                    np.linalg.norm(transformed_lattice1[i]) * np.linalg.norm(transformed_lattice1[i + 1]))
            cos_angle2 = np.abs(np.dot(transformed_lattice2[i], transformed_lattice2[i + 1])) / (
                    np.linalg.norm(transformed_lattice2[i]) * np.linalg.norm(transformed_lattice2[i + 1]))
        else:
            cos_angle = np.abs(np.dot(transformed_lattice1[i], transformed_lattice1[0])) / (
                    np.linalg.norm(transformed_lattice1[i]) * np.linalg.norm(transformed_lattice1[0]))
            cos_angle2 = np.abs(np.dot(transformed_lattice2[i], transformed_lattice2[0])) / (
                    np.linalg.norm(transformed_lattice2[i]) * np.linalg.norm(transformed_lattice2[0]))
        if np.abs(cos_angle) > 0.7 or np.abs(cos_angle2) > 0.7:
            return 1000
        angle_diff = (np.arccos(cos_angle) - np.arccos(cos_angle2))
        angs.append(angle_diff)

    # perform transformation to init structures
    transformation1 = SupercellTransformation(matrix1)
    transformation2 = SupercellTransformation(matrix2)
    new_structure1 = transformation1.apply_transformation(init_poscar1)
    new_structure2 = transformation2.apply_transformation(init_poscar2)
    atom_species1 = [str(e) for e in new_structure1.types_of_species]
    atom_num1 = np.array([int(s.strip(atom_species1[i])) for i, s in enumerate(new_structure1.formula.split())])
    atom_pos1 = new_structure1.frac_coords
    atom_species2 = [str(e) for e in new_structure2.types_of_species]
    atom_num2 = np.array([int(s.strip(atom_species2[i])) for i, s in enumerate(new_structure2.formula.split())])
    atom_pos2 = new_structure2.frac_coords

    # check if atomic numbers and species are equal
    if len(atom_species1) != len(atom_species2) or atom_num1.any() != atom_num2.any() or np.size(atom_pos1) != np.size(
            atom_pos2):
        print("Something is wrong with the input structures!")
        exit()

    N_species = len(atom_species1)
    N_atoms = int(np.sum(atom_num1))

    idx = []
    min_dists = []
    for s in range(N_species):
        for i in range(int(np.sum(atom_num1[:s])), int(np.sum(atom_num1[:(s + 1)]))):
            dist = []
            min_dist = None
            for j in range(int(np.sum(atom_num1[:s])), int(np.sum(atom_num1[:(s + 1)]))):
                dist_vec1 = atom_pos1[i, :] - atom_pos2[j, :]
                dist_vec2 = dist_vec1.copy()
                for k in range(3):
                    if dist_vec1[k] > 0.5:
                        dist_vec2[k] = 1 - dist_vec1[k]
                    elif dist_vec1[k] < -0.5:
                        dist_vec2[k] = 1 + dist_vec1[k]
                dist_val = np.linalg.norm(dist_vec2)
                dist.append(dist_val)
                if min_dist is None or dist_val < min_dist:
                    min_dist = dist_val
            idx.append(int(np.argmin(dist) + np.sum(atom_num1[:s])))
            min_dists.append(min_dist)
    min_dists = np.array(min_dists)
    min_dists_sum = np.sum(np.square(min_dists))
    value =0
    L=0
    A=0
    for i in length_ratio:
        L += i ** 2
    for i in angs:
        A += i ** 2
    value=(A+1)*(L+1)*(min_dists_sum+1)

    return value


def eval_fitness2(individual):
    """
    Evaluate the fitness of an individual by squaring each of its 18
    dimensions and adding them together.
    """
    return sum([x ** 2 for x in individual])


def select_parents(population):
    """
    Select two parents from the population using tournament selection.
    """
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    if eval_fitness(parent1) < eval_fitness(parent2):
        return parent1
    else:
        return parent2


def crossover(parent1, parent2):
    """
    Perform two-point crossover on the two parents to create a new offspring.
    """
    point1 = random.randint(1, DIMENSIONS - 2)
    point2 = random.randint(point1, DIMENSIONS - 1)
    offspring = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    return offspring


def mutate(individual):
    """
    Flip a random dimension in the individual with a probability of 0.05.
    """
    if random.random() < 0.5:
        index = random.randint(0, DIMENSIONS - 1)
        individual[index] = random.randint(*DIM_RANGE)


def evolve_population(population):
    """
    Evolve the population for one generation by selecting parents, performing
    crossover and mutation, and replacing the least fit individual with the
    offspring. Returns the evolved population.
    """
    fitnesses = [eval_fitness(individual) for individual in population]
    sort_fit = sorted(fitnesses, reverse=True)
    least_fit_indices = [fitnesses.index(value) for value in sort_fit[0:SUBSTITUTE_NUMBER]]
    for ind in least_fit_indices:
        offspring = crossover(select_parents(population), select_parents(population))
        mutate(offspring)
        population[ind] = offspring
    return population


def find_best_individual(population):
    """
    Find the best individual in the population by evaluating their fitness.
    """
    fitnesses = [eval_fitness(individual) for individual in population]
    best_index = fitnesses.index(min(fitnesses))
    return population[best_index]


def main():
    # initialize the population

    population = generate_population(POP_SIZE)
    print(1)
    population = init_clean_up(population, ITER_SIZE)
    print(3)
    # evolve the population for NUM_GENERATIONS generations
    for generation in range(NUM_GENERATIONS):
        population = evolve_population(population)
        best_individual = find_best_individual(population)
        print("Generation {}: Best Fitness = {}".format(generation + 1, eval_fitness(best_individual)))

    outfit = np.array([eval_fitness(individual) for individual in population])
    outinfo = np.hstack((population, outfit.reshape(len(outfit), 1)))
    np.savetxt("last_generation", np.array(outinfo), fmt="%.8f")

    # print the final best individual
    best_individual = find_best_individual(population)
    print("Final Best Individual: {}".format(best_individual))

    lattice1 = np.array(init_poscar1.as_dict()["lattice"]["matrix"])
    lattice2 = np.array(init_poscar2.as_dict()["lattice"]["matrix"])
    a1, a2, a3, b1, b2, b3, c1, c2, c3, a4, a5, a6, b4, b5, b6, c4, c5, c6 = best_individual
    matrix1 = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
    matrix2 = np.array([[a4, a5, a6], [b4, b5, b6], [c4, c5, c6]])
    transformed_lattice1 = np.dot(matrix1, lattice1)
    transformed_lattice2 = np.dot(matrix2, lattice2)
    length1 = np.linalg.norm(transformed_lattice1, axis=1)
    for i in length1:
        if i < 0.01:
            return 100000
    length2 = np.linalg.norm(transformed_lattice2, axis=1)
    for i in length2:
        if i < 0.01:
            return 100000

    # rearrange the positions of basis vectors in the cell according to the length of each basis vector
    combine1 = []
    for i in range(3):
        combine1.append((length1[i], tuple(transformed_lattice1[i]), tuple(matrix1[i])))
    combine1 = sorted(combine1, key=lambda x: x[0])
    for i in range(3):
        transformed_lattice1[i] = np.array(list(combine1[i][1]))
        matrix1[i] = np.array(list(combine1[i][2]))
        length1[i] = combine1[i][0]
    combine2 = []
    for i in range(3):
        combine2.append((length2[i], tuple(transformed_lattice2[i]), tuple(matrix2[i])))
    combine2 = sorted(combine2, key=lambda x: x[0])
    for i in range(3):
        transformed_lattice2[i] = np.array(list(combine2[i][1]))
        matrix2[i] = np.array(list(combine2[i][2]))
        length2[i] = combine2[i][0]

    # calculate the length change after the transformation
    length_diff = np.abs(length1 - length2)
    length_ratio = length_diff

    angs = []
    for i in range(3):
        if i < 2:
            cos_angle = np.abs(np.dot(transformed_lattice1[i], transformed_lattice1[i + 1])) / (
                    np.linalg.norm(transformed_lattice1[i]) * np.linalg.norm(transformed_lattice1[i + 1]))
            cos_angle2 = np.abs(np.dot(transformed_lattice2[i], transformed_lattice2[i + 1])) / (
                    np.linalg.norm(transformed_lattice2[i]) * np.linalg.norm(transformed_lattice2[i + 1]))
        else:
            cos_angle = np.abs(np.dot(transformed_lattice1[i], transformed_lattice1[0])) / (
                    np.linalg.norm(transformed_lattice1[i]) * np.linalg.norm(transformed_lattice1[0]))
            cos_angle2 = np.abs(np.dot(transformed_lattice2[i], transformed_lattice2[0])) / (
                    np.linalg.norm(transformed_lattice2[i]) * np.linalg.norm(transformed_lattice2[0]))
        angle_diff = (np.arccos(cos_angle) - np.arccos(cos_angle2))
        angs.append(angle_diff)
        print(np.arccos(cos_angle))
        print(np.arccos(cos_angle2))

    transformation1 = SupercellTransformation(matrix1)
    transformation2 = SupercellTransformation(matrix2)

    new_structure1 = transformation1.apply_transformation(init_poscar1)
    new_structure1.to(filename='POSCAR-i.vasp', fmt='poscar')
    new_structure2 = transformation2.apply_transformation(init_poscar2)
    new_structure2.to(filename='POSCAR-f.vasp', fmt='poscar')

    print("transformed lattice 1")
    print(transformed_lattice1)
    print("transformed lattice 2")
    print(transformed_lattice2)
    print(length1)
    print(length2)
    print(angs)


if __name__ == "__main__":
    main()
