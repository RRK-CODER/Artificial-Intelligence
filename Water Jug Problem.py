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

# Time complexity: O(M * N) where M and N are the capacities of the two jugs.
# Auxiliary Space: O(M * N) due to the visited dictionary.
