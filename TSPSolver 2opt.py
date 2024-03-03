import random
import matplotlib.pyplot as plt


# city class that contains name,x and y members
class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

# euclidean theorem
def distance(city1, city2):
    return ((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2) ** 0.5

# total distance (fitness function)
def total_distance(tour):
    total = 0
    for i in range(len(tour) - 1):
        total += distance(tour[i], tour[i + 1])
    total += distance(tour[-1], tour[0])  # Returning to the starting city
    return total

# rank based
def rank_based_selection(population):
    ranked_population = sorted(population, key=lambda x: total_distance(x))
    total_rank = sum(range(1, len(population) + 1))
    probabilities = [rank / total_rank for rank in range(1, len(population) + 1)]

    selected = []
    for _ in range(len(population) // 2):
        selected_parents = random.sample(range(len(population)), 2)
        parent1 = ranked_population[selected_parents[0]]
        parent2 = ranked_population[selected_parents[1]]
        selected.append(parent1 if total_distance(parent1) < total_distance(parent2) else parent2)

    return selected

# roulette based 
def roulette_based_selection(population):
    total_fitness = sum(1 / total_distance(child) for child in population)
    probabilities = [1 / (total_distance(child) * total_fitness) for child in population]

    selected = []
    for _ in range(len(population) // 2):
        selected_parents = random.sample(range(len(population)), 2)
        parent1 = population[selected_parents[0]]
        parent2 = population[selected_parents[1]]
        selected.append(parent1 if total_distance(parent1) < total_distance(parent2) else parent2)

    return selected

# cycle crossover
def cycle_crossover(parent1, parent2):
    cycle_start = random.randint(0, len(parent1) - 1)
    child = [-1] * len(parent1)

    while child[cycle_start] == -1:
        child[cycle_start] = parent1[cycle_start]
        cycle_start = parent2.index(parent1[cycle_start])

    for i in range(len(parent1)):
        if child[i] == -1:
            child[i] = parent2[i]

    return child

# random slide mutation
def random_slide_mutation(child,mutation_rate):
    if random.random() < mutation_rate:
        start = random.randint(0, len(child) - 1)
        length = random.randint(2, len(child) // 2)
        subset = child[start:start + length]
        child[start:start + length] = random.sample(subset, len(subset))
    return child

# insert mutation
def insert_mutation(child, mutation_rate):
    if random.random() < mutation_rate:
        index1 = random.randrange(len(child))
        index2 = random.randrange(len(child))

        if index1 > index2:
            index1, index2 = index2, index1  # Ensure index1 is less than index2

        element = child.pop(index2)
        child.insert(index1 + 1, element)
    return child

# initial population generation
def generate_initial_population(city_list, population_size):
    population = [random.sample(city_list, len(city_list)) for _ in range(population_size)]
    return population



# read file part
def read_tsp_data(file_path):
    city_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("NODE_COORD_SECTION"):
                break
        for line in file:
            if line.strip() == "EOF":
                break
            parts = line.split()
            city_list.append(City(int(parts[0]), float(parts[1]), float(parts[2])))
    return city_list
#2 opt
def two_opt_optimization(tour):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1:
                    continue  # No reverse for adjacent cities
                new_tour = tour[:]
                new_tour[i:j] = reversed(new_tour[i:j])
                if total_distance(new_tour) < total_distance(tour):
                    tour = new_tour
                    improved = True
                 
    return tour
 #3 opt moving
def three_opt_move(tour, i, j, k):
    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:k][::-1] + tour[k:]
    return new_tour

# 3 opt
def three_optimization(tour):
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                for k in range(j + 1, len(tour)):
                    if k - i == 2:
                        continue
                    old_distance = (
                        distance(tour[i], tour[i + 1])
                        + distance(tour[j], tour[j + 1])
                        + distance(tour[k], tour[(k + 1) % len(tour)])
                    )
                    new_tour = three_opt_move(tour, i + 1, j + 1, k + 1)
                    new_distance = (
                        distance(new_tour[i], new_tour[i + 1])
                        + distance(new_tour[j], new_tour[j + 1])
                        + distance(new_tour[k], new_tour[(k + 1) % len(new_tour)])
                    )
                    if new_distance < old_distance:
                        tour = new_tour
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return tour

def apply_optimizations(population, two_opt_prob, three_opt_prob):
    subset_size = int(len(population) * 0.2)
    subset_size = min(subset_size, len(population))

    if subset_size == 0:
        return population

    subset_indices = random.sample(range(len(population)), subset_size)
    subset = [population[i] for i in subset_indices]

    for i, child in zip(subset_indices, apply2opt(subset, two_opt_prob)):
        population[i] = child

    for i, child in zip(subset_indices, apply3opt(subset, three_opt_prob)):
        population[i] = child

    return population

def calculate_average_fitness(population):
    total_fitness = sum(total_distance(child) for child in population)
    average_fitness = total_fitness / len(population)
    return average_fitness

def genetic_algorithm(city_list, population_size, generations, mutation_rate,two_opt_prob,three_opt_prob,opt_subset_prob):
    population = generate_initial_population(city_list, population_size)
    best_solution = min(population, key=lambda x: total_distance(x))
    initial_best_fitness = total_distance(best_solution)
    best_distances = [total_distance(best_solution)]
    average_fitness_values = [calculate_average_fitness(population)]

    for generation in range(generations):
        next_population = [best_solution]  # Transferring the best solution to next generation

        if random.random() < 0.5:
            parents = rank_based_selection(population[:population_size // 2])
        else:
            parents = roulette_based_selection(population[:population_size // 2])

        #print(f"Generation {generation + 1} - Selected Parents:")
        #for i, (parent1, parent2) in enumerate(zip(parents_rank, parents_roulette)):
            #print(f"Pair {i + 1}: Rank-Based - {total_distance(parent1)}, Roulette-Based - {total_distance(parent2)}")

        # Crossover and Mutation
        for _ in range(population_size - 1):
            parent1, parent2 = random.sample(parents, 2)
            child = cycle_crossover(parent1, parent2)
            if random.random() < 0.5:
                child = insert_mutation(child, mutation_rate)
            else:
                child = random_slide_mutation(child, mutation_rate)
            
            if random.random() < opt_subset_prob:
                #print(f"Applying optimizations at generation {generation + 1}")
                child = apply_optimizations([child], two_opt_prob, three_opt_prob)[0]
            
            next_population.append(child)

        population = next_population
        best_solution = min(population, key=lambda x: total_distance(x))
        best_distances.append(total_distance(best_solution))
        average_fitness_values.append(calculate_average_fitness(population))

        final_best_fitness = total_distance(best_solution)
    print(f"Generation {generation + 1} - Best Fitness: {final_best_fitness}")
    print("---------------------------------")    
    print(f"Initial Best Fitness: {initial_best_fitness}")
    print(f"Final Best Fitness: {final_best_fitness}")
    #print(f"Average Fitness Over Generations:")
    #for gen, avg_fitness in enumerate(average_fitness_values):
    #    if gen+1==generations+1:
    #        break
    #    print(f"Generation {gen+1} average:  {avg_fitness}")

    return best_solution, best_distances

def apply2opt(child,two_opt_prob):
    if random.random() < two_opt_prob:
        
                new_child = two_opt_optimization(child)
                if total_distance(new_child) < total_distance(child):
                    child = new_child 
    return child

def apply3opt(child,three_opt_prob):
    if random.random() < three_opt_prob:
                new_child = three_optimization(child)
                if total_distance(new_child) < total_distance(child):
                    child = new_child 
    return child

file_path = "C:\\Users\\ASUS\\Downloads\\berlin52.txt"  #change this path 
city_list = read_tsp_data(file_path)

population_size = 100
generations = 1000
mutation_rate=0.5
two_opt_prob=0.02
three_opt_prob=0.0
opt_subset_prob = 0.02

best_solution, best_distances = genetic_algorithm(city_list, population_size, generations, mutation_rate,two_opt_prob,three_opt_prob,opt_subset_prob)



# graph part
plt.plot(range(generations + 1), best_distances)
plt.xlabel('Generation')
plt.ylabel('Total Distance')
plt.title('Best Solution Improvement Over Generations without 2-opt')
plt.ticklabel_format(style='plain', axis='y')
plt.show()

x_coordinates = [city.x for city in best_solution]
y_coordinates = [city.y for city in best_solution]

# Plotting the cities
plt.scatter(x_coordinates, y_coordinates, color='blue', marker='o', label='Cities')

# Connecting the cities in the best solution
for i in range(len(best_solution) - 1):
    plt.plot([best_solution[i].x, best_solution[i + 1].x], [best_solution[i].y, best_solution[i + 1].y], color='red')

# Connecting the last city to the first city to complete the tour
plt.plot([best_solution[-1].x, best_solution[0].x], [best_solution[-1].y, best_solution[0].y], color='red')

# Adding labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Final Generation\'s Best Solution')

# Displaying the plot
plt.show()
