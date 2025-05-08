import random
from deap import base,creator,tools,algorithms

# 🧠 1. Create a population of possible solutions (individuals)

# Define fitness function
creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
creator.create("Individual",list,fitness=creator.FitnessMin)

#STEP 2️⃣: How to build individuals and a population
toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-0.5,0.5)

#Register how to create an individual and a population
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=3)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

#STEP 3️⃣:  Evaluate their fitness using a problem-specific function
def eval_func(individual):
    return sum(x**2 for x in individual),
# The comma is important to return a tuple"

toolbox.register("evaluate",eval_func)

#STEP 4️⃣: Define evolution rules (how nature works)
# Select good individuals to be parents
toolbox.register("select", tools.selTournament, tournsize=3)

# Crossover: combine two parents to create offspring
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Mutate: make small changes to an individual
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)

#STEP 5️⃣: Run the evolution process
population = toolbox.population(n=50)
generations = 20

for gen in range(generations):
    # Step 1: Make kids from current generation
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

    # Step 2: Evaluate how good the kids are
    for ind in offspring:
        ind.fitness.values = toolbox.evaluate(ind)

    # Step 3: Pick best individuals for next generation
    population = toolbox.select(offspring, k=len(population))

#TEP 6️⃣: Get the best solution found
best_ind = tools.selBest(population, k=1)[0]
print("Best individual:", best_ind)
print("Best fitness:", best_ind.fitness.values[0])
