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