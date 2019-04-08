from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import numpy as np
from sklearn import tree

### Dependence tree with randomly generated data ###
# OVERVIEW #
# Load Data
# Separate data by class value

# Calculate mean for each attrib. by class value
# Calculate Variance
# Calculate Standard Dev.

# PART 1 #

# visuals: create tree structure #

# root
x5 = Node("5")
# row 1
x7 = Node("7", parent=x5)
x4 = Node("4", parent=x5)
# row 2
x1 = Node("1", parent=x7)
x2 = Node("2", parent=x4)
x3 = Node("3", parent=x4)
# row 3
x8 = Node("8", parent=x1)
x0 = Node("0", parent=x1)
x6 = Node("6", parent=x2)
# row 4
x9 = Node("9", parent=x0)

# Render tree
for pre, fill, node in RenderTree(x5):
    print("%s%s" % (pre, node.name))

DotExporter(x5).to_picture("tree.png")


# create matrix of probabilities that given feature is 0 for each class and features #

# init with rand
w1_prob = np.array(np.random.random_sample((2, 10)))
w2_prob = np.array(np.random.random_sample((2, 10)))
w3_prob = np.array(np.random.random_sample((2, 10)))
w4_prob = np.array(np.random.random_sample((2, 10)))


# parents for each feature (j) in each matrix
tree = [5, 7, 4, 1, 2, 3, 8, 0, 6, 9]
tree_parent = [1, 7, 4, 4, 5, None, 2, 5, 1, 0]


def Generate(prob, omega, dataSize):
    # generate values based on tree and probabilities
    # dataSize is num of data points to generate

    array = np.zeros((dataSize, 11))
    for n in range(dataSize):
       for i in tree:
            if tree_parent[i] == None:
                # root
                if prob[0, i] > np.random.random_sample():
                    array[n,i] = 1
                else:
                    array[n,i]
            else:
                # children
                if np.any(array[tree_parent[i]] == 0):
                    if prob[0, i] > np.random.random_sample():
                        array[n,i] = 1
                else:
                    if prob[1, i] > np.random.random_sample():
                        array[n,i] = 1
            array[n, 10] = omega
    return array


samples = 2000
w1 = (Generate(w1_prob, 1, samples))
w2 = (Generate(w2_prob, 2, samples))
w3 = (Generate(w3_prob, 3, samples))
w4 = (Generate(w4_prob, 4, samples))


print("\nProbability matrix")
print(""" 
row 0: Pr[xi == 0 | parent == 0]
row 1: Pr[xi == 0 | parent == 1]
NOTE: It follows that for all nodes after root node,
Pr[xi == 1 | parent == 0] = 1 - Pr(row0 element i)
Pr[xi == 1 | parent == 1] = 1 - Pr(row1 element i)\n""")

# set print to 3 decimal points
np.set_printoptions(precision=3)
print(w1_prob, end='\n\n')
print(w2_prob, end='\n\n')
print(w3_prob, end='\n\n')
print(w4_prob, end='\n\n')

print("Last element in array indicates the CLASS")
print("Some output from the 2000 sample of omega 1:", end="\n" + 35*"-" + "\n")
print(w1[3], end="\n\n")
print("Some output from the 2000 sample of omega 2:", end="\n" + 35*"-" + "\n")
print(w2[3], end="\n\n")
print("Some output from the 2000 sample of omega 3:", end="\n" + 35*"-" + "\n")
print(w3[3], end="\n\n")
print("Some output from the 2000 sample of omega 4:", end="\n" + 35*"-" + "\n")
print(w4[3], end="\n\n")

# PART 2 #
# find mean of feature values from sample (cols)
# class number is the last element in each row
w1_mean = np.mean(w1, axis=0)
w2_mean = np.mean(w2, axis=0)
w3_mean = np.mean(w3, axis=0)
w4_mean = np.mean(w4, axis=0)

print("""
NOTE: The class number is the last element in each row 
omega 1 means: {}
omega 2 means: {}
omega 3 means: {}
omega 4 means: {}""".format(w1_mean, w2_mean, w3_mean, w4_mean))