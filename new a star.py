from cmath import inf

def aStarAlgo(start_node, stop_node):
    open_set = set([start_node])
    closed_set = set()
    g = {start_node: 0}  # store distance from starting node
    parents = {start_node: start_node}  # parents contain an adjacency map of all nodes
    
    while len(open_set) > 0:
        n = None
        # node with the lowest f() is found
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        if n == stop_node or n not in Graph_nodes:
            break
        else:
            open_set.remove(n)
            closed_set.add(n)
            
            for (m, weight) in get_neighbors(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)

    if n == stop_node:
        path = []
        while parents[n] != n:
            path.append(n)
            n = parents[n]
        path.append(start_node)
        path.reverse()
        print('Path found: {}'.format(path))
        return path
    else:
        print('Path does not exist!')
        return None

# define function to return neighbors and their distance
# from the passed node
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

def heuristic(n):
    H_dist = {
        'S': 2,
        'A': 8,
        'B': 4,
        'C': 3,
        'D': float('inf'),
        'E': float('inf'),
        'G': 0
    }
    return H_dist[n]

Graph_nodes = {
    'S': [('A', 1), ('B', 5), ('C', 8)],
    'A': [('D', 3), ('E', 7), ('G', 9)],
    'B': [('G', 4)],
    'C': [('G', 5)],
}

aStarAlgo('S', 'G')
