import heapq

def uniform_cost_search(start, end):
    heap = [(0, start)]
    visited = set()

    while heap:
        time, node = heapq.heappop(heap)

        if node == end:
            return time

        if node in visited:
            continue

        visited.add(node)

        if node + 1 <= end:
            heapq.heappush(heap, (time + 1, node + 1))  # Walking
        if node + 3 <= end:
            heapq.heappush(heap, (time + 1.5, node + 3))  # Running
        if 2 * node <= end:
            heapq.heappush(heap, (time + 2, 2 * node))  # Magic tram

    return float('inf')
