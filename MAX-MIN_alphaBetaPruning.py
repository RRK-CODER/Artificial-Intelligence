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




