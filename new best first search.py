from cmath import inf

def bestfs(start_node, stop_node):
    open_set = {start_node}
    closed_set = set()
    g = {start_node: 0}
    parents = {start_node: start_node}

    while open_set:
        n = None
        for v in open_set:
            if n == None or heuristic(v) < heuristic(n):
                n = v

        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for m in get_neighbors(n) or []:
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n

        if n == None:
            print('Path does not exist!')
            return None
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)


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
        'D': inf,
        'E': inf,
        'G': 0
    }
    return H_dist.get(n, inf)

Graph_nodes = {
    'S': ['A', 'B', 'C'],
    'A': ['D', 'E', 'G'],
    'B': ['G'],
    'C': ['G'],
}

bestfs('S', 'G')
