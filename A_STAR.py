def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node
    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
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
    print('Path does not exist!')
    return None

def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

def heuristic(n):
    H_dist = {
        'A': 11,
        'B': 6,
        'C': 5,
        'D': 7,
        'E': 3,
        'F': 6,
        'G': 5,
        'H': 3,
        'I': 1,
        'J': 0
    }
    return H_dist[n]

#Describe your graph here
Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('A', 6), ('C', 3), ('D', 2)],
    'C': [('B', 3), ('D', 1), ('E', 5)],
    'D': [('B', 2), ('C', 1), ('E', 8)],
    'E': [('C', 5), ('D', 8), ('I', 5), ('J', 5)],
    'F': [('A', 3), ('G', 1), ('H', 7)],
    'G': [('F', 1), ('I', 3)],
    'H': [('F', 7), ('I', 2)],
    'I': [('E', 5), ('G', 3), ('H', 2), ('J', 3)],
}

aStarAlgo('A', 'J')

# The time complexity is often expressed as O(b^d), where:

# b is the branching factor (maximum number of successors any node can have).
# d is the depth of the shallowest goal node.

# The space complexity is O(b^d) due to the number of nodes that may need to be stored in memory.

#Intialize the open list with the starting node along with a priority based on the combined f score(g_score+ heuristic score) of the node
#create dictionary to store parent nodes(came-from), g_score and f_score for each node
#set the g_score of starting node to 0 and f_score to heuristic score
#enter a loop while the open list is not empty
    #extrcat the node with the lowest f_score from the open list
    #if the current node is the goal node, reconstruct the path using the came_from dictionary and return it
    # for each neighbor of the current node
        #calculate the g_score of the neighbor
        #if the calculated g_score is less than the g_score of the neighbor, update the g_score and f_score of the neighbor
        #add the neighbor to the open list if it is not there already
            #calculate a tentative g_score by adding the edge cost from the current node to the neighbor
            #if this tentative g score is better than the recorded g_score of the neighbor, update the g_score and f_score of the neighbor and set the current node as its parent
            #if the neighbor is not in the open list, add it with it's f score to the open list
    #if the open list becomes empty and the goal is not reached, there is no path
    