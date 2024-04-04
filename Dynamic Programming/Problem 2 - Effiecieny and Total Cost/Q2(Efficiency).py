import networkx as nx
import heapq
import matplotlib.pyplot as plt

# Updated simplified graph representation with adjusted distances between Markaz
graph = {
    'I-8 Markaz': {'I-10 Markaz': 10, 'I-9 Markaz': 5},
    'I-10 Markaz': {'I-8 Markaz': 10, 'I-9 Markaz': 8, 'F-9 Markaz': 20},
    'I-9 Markaz': {'I-8 Markaz': 5, 'I-10 Markaz': 8, 'F-9 Markaz': 15},
    'F-9 Markaz': {'I-10 Markaz': 20, 'I-9 Markaz': 15}
}

# Updated heuristic function with different values for A* Search
heuristic = {
    'I-8 Markaz': 22,
    'I-10 Markaz': 18,
    'I-9 Markaz': 25,
    'F-9 Markaz': 10
}

def uniform_cost_search(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))  # Priority queue for nodes with their cumulative cost
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored_nodes = 0

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)

        if current_node == goal:
            break

        for neighbor, distance in graph[current_node].items():
            new_cost = cost_so_far[current_node] + distance
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current_node
                explored_nodes += 1  # Count explored nodes

    return explored_nodes

def astar(graph, heuristic, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))  # Priority queue for nodes with their cumulative cost
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored_nodes = 0

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)

        if current_node == goal:
            break

        for neighbor, distance in graph[current_node].items():
            new_cost = cost_so_far[current_node] + distance
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]  # A* evaluation function
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current_node
                explored_nodes += 1  # Count explored nodes

    return explored_nodes

# Example usage:
source = 'I-8 Markaz'
destination = 'F-9 Markaz'

uc_explored = uniform_cost_search(graph, source, destination)
a_star_explored = astar(graph, heuristic, source, destination)

print(f"Nodes explored in Uniform Cost Search: {uc_explored}")
print(f"Nodes explored in A* Search: {a_star_explored}")
