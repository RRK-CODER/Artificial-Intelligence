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