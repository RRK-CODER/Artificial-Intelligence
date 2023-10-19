
Water Jug

from collections import defaultdict

jug1, jug2, aim = 1, 4, 3

visited = defaultdict(lambda: False)

def waterJugSolver(amt1, amt2):
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):
        print(amt1, amt2)
        print("Successfully found the aim!")
        return True

    if visited[(amt1, amt2)] == False:
        print(amt1, amt2)
        visited[(amt1, amt2)] = True

        return (waterJugSolver(0, amt2) or
                waterJugSolver(amt1, 0) or
                waterJugSolver(jug1, amt2) or
                waterJugSolver(amt1, jug2) or
                waterJugSolver(amt1 + min(amt2, (jug1-amt1)),
                               amt2 - min(amt2, (jug1-amt1))) or
                waterJugSolver(amt1 - min(amt1, (jug2-amt2)),
                               amt2 + min(amt1, (jug2-amt2))))
    else:
        return False

print("Steps:")

if not waterJugSolver(0, 0):
    print("Cannot achieve the desired result.")


# Time complexity: O(M * N) Auxiliary Space: O(M * N) where M and N are the capacities of the two jugs.
# Algorithm: Water Jug Problem Solver

# Inputs:
# - jug1: Capacity of the first jug
# - jug2: Capacity of the second jug
# - aim: Desired amount of water
# - visited: Dictionary to keep track of visited states

# 1. Define a function waterJugSolver(amt1, amt2):
#    a. If either jug1 or jug2 contains the desired amount of water, print success message and return True.
#    b. If the current state (amt1, amt2) has not been visited:
#       i. Print the current state.
#       ii. Mark the current state as visited.
#       iii. Recursively explore the following possibilities:
#            - Fill jug1 to capacity and continue exploration.
#            - Fill jug2 to capacity and continue exploration.
#            - Empty jug1 and continue exploration.
#            - Empty jug2 and continue exploration.
#            - Pour water from jug1 to jug2 and continue exploration.
#            - Pour water from jug2 to jug1 and continue exploration.
#    c. If all recursive calls return False, return False.

# Main:
# 1. Print "Steps:"
# 2. If the waterJugSolver(0, 0) returns False, print "Cannot achieve the desired result."
UCS 
def uniform_cost_search(goal, start):
    global graph, cost
    answer = []
    queue = []
    for i in range(len(goal)):
        answer.append(10**8)
    queue.append([0, start])
    visited = {}
    count = 0
    while (len(queue) > 0):
        queue = sorted(queue)
        p = queue[-1]
        del queue[-1]
        p[0] *= -1
        if (p[1] in goal):
            index = goal.index(p[1])
            if (answer[index] == 10**8):
                count += 1
            if (answer[index] > p[0]):
                answer[index] = p[0]
            del queue[-1]
            queue = sorted(queue)
            if (count == len(goal)):
                return answer
        if (p[1] not in visited):
            for i in range(len(graph[p[1]])):
                queue.append([(p[0] + cost[(p[1], graph[p[1]][i])])* -1, graph[p[1]][i]])
        visited[p[1]] = 1
    return answer

if __name__ == '__main__':
    graph, cost = [[] for i in range(8)], {}
    graph[0].append(1)
    graph[0].append(3)
    graph[3].append(1)
    graph[3].append(6)
    graph[3].append(4)
    graph[1].append(6)
    graph[4].append(2)
    graph[4].append(5)
    graph[2].append(1)
    graph[5].append(2)
    graph[5].append(6)
    graph[6].append(4)
    cost[(0, 1)] = 2
    cost[(0, 3)] = 5
    cost[(1, 6)] = 1
    cost[(3, 1)] = 5
    cost[(3, 6)] = 6
    cost[(3, 4)] = 2
    cost[(2, 1)] = 4
    cost[(4, 2)] = 4
    cost[(4, 5)] = 3
    cost[(5, 2)] = 6
    cost[(5, 6)] = 3
    cost[(6, 4)] = 7
    goal = []
    goal.append(6)
    answer = uniform_cost_search(goal, 0)
    print("Minimum cost from 0 to 6 is = ", answer[0])

# Time Complexity: O( m ^ (1+floor(l/e))) 

# where, 
# m is the maximum number of neighbors a node has 
# l is the length of the shortest path to the goal state 
# e is the least cost of an edge

# O(V+E).
# V is the number of vertices
# E is the number of edges
# '''

# instead of inserting all vertices into a priority queue, we 
# insert only the source, then one by one insert when needed. 
# In every step, we check if the item is already in the priority queue 
# (using the visited array). If yes, we perform the decrease key, else we insert it.
# Uniform-Cost Search is similar to Dijikstra’s algorithm. In this algorithm from the starting state, we will visit the adjacent states
# and will choose the least costly state then we will choose the next least costly state from the all un-visited and adjacent states of the visited states, 
# in this way we will try to reach the goal state (note we won’t continue the path through a goal state ), even if we reach the goal state we will continue searching 
# for other possible paths( if there are multiple goals). We will keep a priority queue that will give the least costly next state from all the adjacent states of visited states.

# Insert the source node into the priority queue
#remove the element with the highest priority from the priority queue
#if the removed node is the destination, print total cost and stop the algorithm
#else if check if the node is in the visited array, if not then add it to the visited array and add all the adjacent nodes with the cumulative cost as the priority in the priority queue
#repeat the steps from 2 until the priority queue is empty
A*

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
    

MAXMIN

MAX, MIN = 1000, -1000

def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]

    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    else:
        best = MAX
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

if __name__ == "__main__":
    values = [3, 5, 6, 9, 1, 2, 0, -1]
    print("The optimal value is:", minimax(0, 0, True, values, MIN, MAX))

# Time Complexity -  O(b^d)
# The space complexity is O(d) 
# b is the branching factor (maximum number of successors any node can have).
# d is the depth of the shallowest goal node.


#we define constants MAX and MIN as the initial upper and lower bounds for alpha beta
#the minmax function takes parameters 'depth' to track the depth of the tree, nodeIndex to identify the current node, 'maximizing player' to indicate if it's the maxim

# Initialization:

# Set MAX to 1000 and MIN to -1000, representing positive and negative infinity.
# The minimax function takes parameters: depth indicates the depth of the current node in the tree, nodeIndex is the index of the current node, maximizingPlayer is a boolean indicating whether it is the maximizing player's turn, values is an array representing the values at each node, and alpha and beta are parameters for alpha-beta pruning.
# Termination condition:

# If depth reaches 3 (a leaf node), the function returns the value of the node.
# Maximizing Player's Turn:

# If it's the maximizing player's turn, initialize best to MIN.
# Iterate over child nodes (0 and 1) at the current level.
# Recursively call minimax for each child node with maximizingPlayer set to False.
# Update best to the maximum of its current value and the value returned from the recursive call.
# Update alpha to the maximum of its current value and best.
# Perform alpha-beta pruning: if beta is less than or equal to alpha, break out of the loop.
# Return the best value.
# Minimizing Player's Turn:

# If it's the minimizing player's turn, initialize best to MAX.
# Iterate over child nodes (0 and 1) at the current level.
# Recursively call minimax for each child node with maximizingPlayer set to True.
# Update best to the minimum of its current value and the value returned from the recursive call.
# Update beta to the minimum of its current value and best.
# Perform alpha-beta pruning: if beta is less than or equal to alpha, break out of the loop.
# Return the best value.



Best first search

from queue import PriorityQueue

v = 14
graph = [[] for i in range(v)]

def best_first_search(actual_Src, target, n):
    visited = [False] * n
    pq = PriorityQueue()
    pq.put((0, actual_Src))
    visited[actual_Src] = True
    
    while pq.empty() == False:
        u = pq.get()[1]
        print(u, end=" ")
        if u == target:
            break

        for v, c in graph[u]:
            if visited[v] == False:
                visited[v] = True
                pq.put((c, v))
    print()

def addedge(x, y, cost):
    graph[x].append((y, cost))
    graph[y].append((x, cost))

addedge(0, 1, 3)
addedge(0, 2, 6)
addedge(0, 3, 5)
addedge(1, 4, 9)
addedge(1, 5, 8)
addedge(2, 6, 12)
addedge(2, 7, 14)
addedge(3, 8, 7)
addedge(8, 9, 5)
addedge(8, 10, 6)
addedge(9, 11, 1)
addedge(9, 12, 10)
addedge(9, 13, 2)
 
source = 0
target = 9
best_first_search(source, target, v)

# The worst-case time complexity for Best First Search is O(n * log n) where n is the number of nodes.
#     Space complexity: O(V+E)

// Pseudocode for Best First Search
# Best-First-Search(Graph g, Node start)
#     1) Create an empty PriorityQueue
#        PriorityQueue pq;
#     2) Insert "start" in pq.
#        pq.insert(start)
#     3) Until PriorityQueue is empty
#           u = PriorityQueue.DeleteMin
#           If u is the goal
#              Exit
#           Else
#              Foreach neighbor v of u
#                 If v "Unvisited"
#                     Mark v "Visited"                    
#                     pq.insert(v)
#              Mark u "Examined"                    
# End procedure

Map Coloring

V = 4
graph = [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]

def isValid(v, color, c):
    for i in range(V):
        if graph[v][i] and c == color[i]:
            return False
    return True

def mColoring(colors, color, vertex):
    if vertex == V:
        return True
    for col in range(1, colors + 1):
        if isValid(vertex, color, col):
            color[vertex] = col
            if mColoring(colors, color, vertex + 1):
                return True
            color[vertex] = 0
    return False

colors = 3
color = [0] * V

if not mColoring(colors, color, 0):
    print("Solution does not exist.")
else:
    print("Assigned Colors are:")
    for i in range(V):
        print(color[i], end=" ")

# space complexity is O(V^2)
# time complexity is O(V^E)

# Follow the given steps to solve the problem:

# Create a recursive function that takes the graph, current index, number of vertices, and color array.
# If the current index is equal to the number of vertices. Print the color configuration in the color array.
# Assign a color to a vertex from the range (1 to m).
# For every assigned color, check if the configuration is safe, (i.e. check if the adjacent vertices do not have the same color) and recursively call the function with the next index and number of vertices otherwise, return false
# If any recursive function returns true then return true
# If no recursive function returns true then return false
# Illustration:
# To color the graph, color each node one by one.
# To color the first node there are 3 choices of colors Red, Green and Blue, so lets take the red color for first node.
# After Red color for first node is fixed then we have made choice for second node in similar manner as we did for first node, then for 3rd node and so on.
# There is one important point to remember. while choosing color for the node, it should not be same as the color of the adjacent node.