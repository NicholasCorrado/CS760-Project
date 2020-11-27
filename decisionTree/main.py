import numpy as np
from ppbtree import *  # The pretty print package for decision tree
from decisionTree import * # with my implemented decision tree methods

# use initial tree method to start plot the decision tree,
# can modify stop creteiral by changing 0.05 or 0.001

# initial_tree can generate a tree based on all training datasets, use printTree = true to print the tree
tree = initial_tree('Marine_Clean.csv',printTree = True)

# predictByTree is with two parameters, predict list and the file name
result, prediction = predictByTree([2000,12,-11,87],'Marine_Clean.csv')
# predictByForest is with three parameters, predict list, file name and a parameter
# 1 means using excluding one feature for generating random forest
# 2 means using 80% of data from dataset for generating random forest
result2,prediction2 = predictByForest([2000,12,-11,87],'Marine_Clean.csv',1)
result3,prediction3 = predictByForest([2000,12,-11,87],'Marine_Clean.csv',2)

print(prediction)
print(prediction2)
print(prediction3)



