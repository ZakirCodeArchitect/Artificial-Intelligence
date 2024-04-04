def dfs(start, end):
    if start == end:
        return 0
    if start > end:
        return float('inf')
    
    return min(
        dfs(start + 1, end) + 1,  # Walking
        dfs(start + 3, end) + 1.5,  # Running
        dfs(2 * start, end) + 2  # Magic tram
    )
