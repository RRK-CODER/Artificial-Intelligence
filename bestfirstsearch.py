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

# // Pseudocode for Best First Search
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