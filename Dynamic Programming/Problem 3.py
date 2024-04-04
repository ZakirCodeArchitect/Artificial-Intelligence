import heapq

class TransportationProblem(object):
    def __init__(self, N, weights):
        self.N = N
        self.weights = weights
    
    def startState(self):
        return 1
    
    def isEnd(self, state):
        return state == self.N
    
    def succAndCost(self, state):
        result = []
        if state + 1 <= self.N:
            result.append(('walk', state + 1, self.weights['walk']))
        if state + 3 <= self.N:
            result.append(('run', state + 3, self.weights['run']))
        if 2 * state <= self.N:
            result.append(('tram', 2 * state, self.weights['tram']))
        return result

def uniform_cost_search(start, end, n, weights):
    heap = [(0, start)]
    visited = set()
    paths = {start: ([start], 0)}

    while heap:
        cost, node = heapq.heappop(heap)

        if node == end:
            return paths[node][0]  # Return path

        if node in visited:
            continue

        visited.add(node)

        for action, neighbor, action_cost in get_neighbors(node, n):
            new_cost = cost + weights[action] * action_cost

            if neighbor not in paths or new_cost < paths[neighbor][1]:
                paths[neighbor] = (paths[node][0] + [neighbor], new_cost)
                heapq.heappush(heap, (new_cost, neighbor))

    return None

def get_neighbors(node, n):
    neighbors = []
    if node + 1 <= n:
        neighbors.append(('walk', node + 1, 1))  # Walking
    if node + 3 <= n:
        neighbors.append(('run', node + 3, 1.5))  # Running
    if 2 * node <= n:
        neighbors.append(('tram', 2 * node, 2))  # Magic tram
    return neighbors

def generate_dataset(num_examples):
    dataset = []
    trueWeights = {'walk': 1, 'run': 1.5, 'tram': 2}  # True action costs
    for _ in range(num_examples):
        n = 20  # You can set the block number 'n' here
        start_node = 1
        end_node = n
        true_minimum_cost_path = uniform_cost_search(start_node, end_node, n, trueWeights)
        dataset.append((n, true_minimum_cost_path))
    return dataset

def structuredPerceptron(dataset):
    weights = {'walk': 0, 'run': 0, 'tram': 0}
    for _ in range(10):  # Number of iterations
        for n, truePath in dataset:
            problem = TransportationProblem(n, weights)
            predPath = uniform_cost_search(1, n, n, weights)
            
            for a in set(truePath + predPath):
                if a not in weights:
                    weights[a] = 0  # Add new action to weights
                weights[a] += truePath.count(a) - predPath.count(a)
    return weights

# Generate dataset with true minimum cost paths
examples = generate_dataset(20)

# Learn action costs using structured perceptron
learnedWeights = structuredPerceptron(examples)
print("Learned action costs:", learnedWeights)
