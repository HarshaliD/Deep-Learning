import random

# Distance matrix for 3 cities
dist = [[0, 2, 9],
        [1, 0, 6],
        [7, 3, 0]]

pheromone = [[1]*3 for _ in range(3)]

for _ in range(5):  # 5 iterations
    paths = []
    for _ in range(3):  # 3 ants
        cities = [0]
        while len(cities) < 3:
            next_city = min([i for i in range(3) if i not in cities],
                            key=lambda j: dist[cities[-1]][j] / pheromone[cities[-1]][j])
            cities.append(next_city)
        cities.append(0)  # return to start
        cost = sum(dist[cities[i]][cities[i+1]] for i in range(3))
        paths.append((cities, cost))

    # Evaporate & Update pheromones
    for path, cost in paths:
        for i in range(3):
            a, b = path[i], path[i+1]
            pheromone[a][b] += 1 / cost

# Output best path
best = min(paths, key=lambda x: x[1])
print("Best path:", best[0], "Cost:", best[1])
