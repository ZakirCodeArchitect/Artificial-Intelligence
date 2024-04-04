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

    # Reconstruct the path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    
    return path, cost_so_far[goal] if goal in cost_so_far else None

def astar(graph, heuristic, start, goal):
    frontier = []
    heapq.heappush(frontier, (0 + heuristic[start], start))  # Priority queue for nodes with their cumulative cost + heuristic
    came_from = {start: None}
    cost_so_far = {start: 0}

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

    # Reconstruct the path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    
    return path, cost_so_far[goal] if goal in cost_so_far else None

# Example usage:
source = 'I-8 Markaz'
destination = 'F-9 Markaz'

# Create a graph with distances (costs) as edge labels
G = nx.Graph()
G.add_edges_from([(node1, node2, {'weight': weight}) for node1, neighbors in graph.items() for node2, weight in neighbors.items()])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('Graph of Markaz with Distances')
plt.show()

# Calculate shortest path and total cost using Uniform Cost Search
uc_path, uc_cost = uniform_cost_search(graph, source, destination)

if uc_path is None:
    print(f"No path found from {source} to {destination} using Uniform Cost Search.")
else:
    print(f"Uniform Cost Search Path from {source} to {destination}: {uc_path}")
    print(f"Uniform Cost Search Total Cost: {int(uc_cost)}" if uc_cost is not None else "Uniform Cost Search: No Valid Path")

# Calculate shortest path and total cost using A* Search
a_star_path, a_star_cost = astar(graph, heuristic, source, destination)

if a_star_path is None:
    print(f"No path found from {source} to {destination} using A* Search.")
else:
    print(f"A* Search Path from {source} to {destination}: {a_star_path}")
    print(f"A* Search Total Cost: {int(a_star_cost)}" if a_star_cost is not None else "A* Search: No Valid Path")
