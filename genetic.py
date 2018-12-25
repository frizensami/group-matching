'''
Genetic algorithm to select an optimal grouping of individuals based on their teammate preferences.
'''
import random
import itertools
import copy
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import gmtime, strftime
import signal
import sys
import time

# Parameters for the problem
NUM_PARTICIPANTS = 30
PARTICIPANTS_PER_GROUP = 2
assert(NUM_PARTICIPANTS % PARTICIPANTS_PER_GROUP == 0)
NUM_GROUPS = int(NUM_PARTICIPANTS / PARTICIPANTS_PER_GROUP)

# Parameters for the genetic algorithm
POPULATION_SIZE = 1000
NUM_GENERATIONS = 10000
NUM_ELITISM = int(POPULATION_SIZE / 4) # Keep these number of best-performing individuals for the rest of the algo
NUM_REST_PARENTS = int(POPULATION_SIZE / 4) # The rest of the "non-elite" parents
NUM_CHILDREN = POPULATION_SIZE - NUM_ELITISM - NUM_REST_PARENTS
POSITIVE_WEIGHT = 100
NEGATIVE_WEIGHT = -1000
MUTATION_CHANCE = 0.5
MUTATION_NUM_SWAPS = 1#int(NUM_PARTICIPANTS / 5)
HALL_OF_FAME_SIZE = 5

# Printing params
DEBUG = False

# Non-constants
hall_of_fame = []
ranking = [[NEGATIVE_WEIGHT for x in range(NUM_PARTICIPANTS)]
           for y in range(NUM_PARTICIPANTS)]
for i in range(NUM_PARTICIPANTS):
    ranking[i][i] = 0 # cannot rank yourself
    if i % 2 == 0:
        # really want the person 'next' to you (only one "correct answer")
        ranking[i][i + 1] = POSITIVE_WEIGHT
    else:
        # really want the person 'next' to you (only one "correct answer")
        ranking[i][i - 1] = POSITIVE_WEIGHT


# Seed random?
def is_valid_grouping(grouping):
    # number of groups must be correct
    groups_cor = len(grouping) == NUM_GROUPS
    # number of individuals per group must be correct
    num_groupmem_cor = len(list(filter(lambda g: len(g) != PARTICIPANTS_PER_GROUP, grouping))) == 0
    # All individuals should be included
    all_included = set(list(itertools.chain.from_iterable(grouping))) == set(range(NUM_PARTICIPANTS))

    return (groups_cor and num_groupmem_cor and all_included)


# Gets the list of participant numbers and randomizes splitting them up
# into groups
def generateRandomGrouping():
    participants = [i for i in range(NUM_PARTICIPANTS)]
    random.shuffle(participants)
    idx = 0

    grouping = []
    for g in range(NUM_GROUPS):
        group = []
        for p in range(PARTICIPANTS_PER_GROUP):
            group.append(participants[idx])
            idx += 1
        grouping.append(group)
    return grouping

# Generate an initial list of of groupings by randomly creating them


def generateInitialPopulation(population_size):
    population = []
    for i in range(population_size):
        population.append(generateRandomGrouping())
    return population


def print_population(population):
    for p in population:
        print(p)


def print_ranking(ranking):
    rank_source = 0
    rank_target = 0
    for row in ranking:
        rank_target = 0
        for col in row:
            print(str(rank_source) + " -> " +
                  str(rank_target) + ": " + str(col))
            rank_target += 1
        print("------------")
        rank_source += 1

# Given a single group in a grouping, evaluate that group's fitness
# TODO: Don't use sum only! Use sum only if relationship is not asymmmetricalgg
def group_fitness(group):
    group_fitness_score = 0
    # All-pairs sum of rankings
    for participant in group:
        for other_participant in group:
            group_fitness_score += ranking[participant][other_participant]
            # print(str(participant) + "-> " + str(other_participant) + " - new grp fit: " + str(group_fitness_score))
    return group_fitness_score


# Given a single grouping, evaluate its overall fitness
def fitness(grouping):
    fitness_score = 0
    for group in grouping:
        fitness_score += group_fitness(group)
    return fitness_score

# We will select some number of parents to produce the next generation of offspring
# Modifies the population_with_fitness param, cannot be used after
def select_parents_to_breed(sorted_population_with_fitness):
    selected_parents = []
    # Sort the incoming population by their fitness with the best individuals being at the end
    # population_with_fitness.sort(key=lambda x: x[1])

    # Select the most elite individuals to breed and carry on to next gen as well
    for i in range(NUM_ELITISM):
        selected_parents.append(sorted_population_with_fitness.pop())

    # Select the rest of the mating pool by random chance
    # TODO: This needs to be a weighted sample!
    selected_parents.extend(random.sample(sorted_population_with_fitness, NUM_REST_PARENTS))
    #print("Selected parents")
    #print_population(selected_parents)
    # Don't return the weights
    return list(map(lambda x: x[0], selected_parents))

# Potentially the most important function
# Given two sets of groupings - we need to produce a new valid grouping
def breed_two_parents(p1, p2):
    child = []
    # Create a child between these two by randomly selecting groups from their pool
    group_pool = copy.deepcopy(p1) + copy.deepcopy(list(p2))
    child = random.sample(group_pool, NUM_GROUPS)
    #print("Initial child of " + str(p1) + " and " + str(p2) + ": \n" + str(child))

    # We need to "correct" the child so that it can be a valid group
    # This also introduces a form of mutation
    # We first need to find out where the repeats are in every group, and which participants are left out
    missing_participants = set(range(NUM_PARTICIPANTS))
    repeat_locations = {} # mapping between (participant num) -> [(groupidx, memberidx)]
    for groupidx, group in enumerate(child):
        for memberidx, participant in enumerate(group):
            if participant in repeat_locations:
                repeat_locations[participant].append((groupidx, memberidx))
            else:
                repeat_locations[participant] = [(groupidx, memberidx)]
            missing_participants = missing_participants - set([participant])
            
    # repeat_locations = {k:v for (k,v) in repeat_locations.items() if len(v) > 1}
    
    # For each set of repeats, save one repeat location each for each repeated participant, but the rest need to be overwritten (therefore we're taking a sample of len(v) - 1)
    repeat_locations = [random.sample(v, len(v) - 1) for (k,v) in repeat_locations.items() if len(v) > 1]
    # Flatten list
    repeat_locations = list(itertools.chain.from_iterable(repeat_locations))

    #print("Missing participants: " + str(missing_participants))
    #print("Repeat locations to replace: " + str(repeat_locations))

    # Now we insert the missing participants into a random repeat location
    random_locations_to_replace = random.sample(repeat_locations, len(missing_participants))
    for idx, missing_participant in enumerate(missing_participants):
        groupidx, memberidx = random_locations_to_replace[idx]
        #print("Replacing val at : " + str(random_locations_to_replace[idx]) + " with " + str(missing_participant))
        child[groupidx][memberidx] = missing_participant
    #print("Final child: " + str(child))
    return child

def breed(parents):
    children = []
    # Randomize the order of parents to allow for random breeding
    randomized_parents = random.sample(parents, len(parents))
    # We need to generate NUM_CHILDREN children, so breed parents until we get that
    for i in range(NUM_CHILDREN):
        child = breed_two_parents(randomized_parents[i % len(randomized_parents)], randomized_parents[(i + 1) % len(randomized_parents)])
        children.append(child)
        #print("Got child: " + str(child))
        #print("Children: " + str(children))
        #print()
    return children

def mutate(population):
    for grouping in population:
        if random.random() < MUTATION_CHANCE:
            # Mutate this grouping
            for i in range(MUTATION_NUM_SWAPS):
                # Swap random group members this many times
                # Pick two random groups
                groups_to_swap = random.sample(range(NUM_GROUPS), 2)
                group_idx1 = groups_to_swap[0]
                group_idx2 = groups_to_swap[1]
                participant_idx1 = random.choice(range(PARTICIPANTS_PER_GROUP))
                participant_idx2 = random.choice(range(PARTICIPANTS_PER_GROUP))
                # Make the swap
                temp = grouping[group_idx1][participant_idx1]
                grouping[group_idx1][participant_idx1] = grouping[group_idx2][participant_idx2]
                grouping[group_idx2][participant_idx2] = temp




def create_new_halloffame(old_hof, sorted_population_with_fitness):
    old_hof.extend(sorted_population_with_fitness[-HALL_OF_FAME_SIZE:])
    old_hof.sort(key=lambda x: x[1])
    return old_hof[-HALL_OF_FAME_SIZE:]

def animate(i):
    ax1.clear()
    ax1.plot(xs,ys)

def exit_handler(sig, frame):
        print("\nEvolution complete or interrupted. \n")
        print("\n----- Final Hall Of Fame ----- ")
        print_population(hall_of_fame)

        # Draw final results
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(1,1,1)
        fig.suptitle('Fitness vs number of generations', fontsize=20)
        plt.xlabel('Number of Generations', fontsize=16)
        plt.ylabel('Best Fitness Achieved', fontsize=16)
        plt.plot(xs, ys)
        fig.savefig(str(NUM_PARTICIPANTS) + 'p-' + str(NUM_GROUPS) + 'g-' + str(NUM_GENERATIONS) + 'gen-' + strftime("%Y-%m-%d--%H:%M:%S", gmtime()) + '.png')
        plt.show()
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_handler)

    population = generateInitialPopulation(POPULATION_SIZE)
    print()
    print("Initial population:")
    print_population(population)
    print("Ranking:")
    print(ranking)

    # Plotting params
    xs = []
    ys = []


    # Set up initial state for generations
    generation = 0
    best_match = ([], -sys.maxsize - 1)
    start_time = time.time()
    while generation < NUM_GENERATIONS:
        print("\n---------- GENERATION " + str(generation) + " ----------")
        population_with_fitness = list(map(lambda gs: (gs, fitness(gs)), population))

        if DEBUG:
            print("Population with fitness: ")
            print_population(population_with_fitness)

        # Everyone after this wants this list to be sorted, so immediately sort
        population_with_fitness.sort(key=lambda x: x[1])
        # Update the "Hall of Fame"
        hall_of_fame = create_new_halloffame(hall_of_fame, population_with_fitness)

        if DEBUG:
            print("Hall of Fame: ")
            print_population(hall_of_fame)

        # Note a "best grouping"
        # best_grouping = max(population_with_fitness, key=lambda x: x[1])
        #if best_grouping[1] > best_match[1]:
        #    best_match = copy.deepcopy(best_grouping)

        parents = select_parents_to_breed(population_with_fitness)
        if DEBUG:
            print("Parents: " + str(parents))

        children = breed(parents)
        if DEBUG:
            print("Children: ")
            print_population(children)

        # Create new population, append parents and children together
        new_population = []
        new_population.extend(parents)
        new_population.extend(children)

        if DEBUG:
            print("Pre-mutation: ")
            print_population(new_population)
        mutate(new_population)
        if DEBUG:
            print("Post-mutation: ")
            print_population(new_population)
        population = new_population
        
        # Just a check to make sure all of the new generation are valid groups
        assert(all(map(is_valid_grouping, new_population)))

        # TODO: Display generation stats on a graph?
        best_fitness_so_far = hall_of_fame[-1][1]
        print("Best fitness at generation " + str(generation) + " = " + str(best_fitness_so_far))
        xs.append(generation)
        ys.append(best_fitness_so_far)
        
        # Measure time
        iter_time = time.time()
        time_per_generation = (iter_time - start_time) / (generation + 1)
        time_remaining_seconds = time_per_generation * (NUM_GENERATIONS - generation - 1)
        print("Time remaining: " + str(round(time_remaining_seconds, 2)) + " s |or| " + str(round(time_remaining_seconds/60, 2)) + " min |or| " + str(round(time_remaining_seconds / 3600, 2)) + " hours")

        # Move on to next generation
        generation += 1

    exit_handler(None, None)

