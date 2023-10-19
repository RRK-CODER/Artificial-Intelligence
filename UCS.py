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
