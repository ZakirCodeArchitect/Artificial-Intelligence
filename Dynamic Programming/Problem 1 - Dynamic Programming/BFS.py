from collections import deque

def bfs(start, end):
    queue = deque([(start, 0)])

    while queue:
        node, time = queue.popleft()

        if node == end:
            return time

        if node + 1 <= end:
            queue.append((node + 1, time + 1))  # Walking
        if node + 3 <= end:
            queue.append((node + 3, time + 1.5))  # Running
        if 2 * node <= end:
            queue.append((2 * node, time + 2))  # Magic tram

    return float('inf')
