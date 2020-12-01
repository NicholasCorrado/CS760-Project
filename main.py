import numpy as np
from linearRegression import *
from regressionTree import *


if __name__ == "__main__":
    
	
	# regression tree	
    regTree = regressionTreeConstruct('Marine_Clean.csv', 10) # construct of regression tree based on marine data    
    evalTree('Marine_Clean.csv') # evalTree to check possible tree depth,choose optimal depth with goodness of fit    
    predictions = testCase('TestCase.csv',regTree) # predict based on reg tree, and plot the predictions 
