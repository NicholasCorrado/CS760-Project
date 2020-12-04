import numpy as np
from linearRegression import *
from regressionTree import *


if __name__ == "__main__":    
	
    # construct of regression tree based on marine data
    regTree = regressionTreeConstruct('Marine_Clean.csv', 10)
    # evalTree to check all possible tree depth, and choose the best depth
    r2_train, r2_test , MSE_train_tree = evalTree('Marine_Clean.csv')
    print("r2 goodness of fit is %.2f" %r2_train)
    print("MSE for training set is %.2f" %MSE_train_tree)
    # can use testCaseFunction to do prediction based on previously tree, and plot the predictions
    predictions, MSE_test_tree = testCase('TestCase.csv',regTree)
    print("MSE for testing set is %.2f" % MSE_test_tree)
