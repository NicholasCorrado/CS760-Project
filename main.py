import numpy as np
from linearRegression import *
from regressionTree import *


if __name__ == "__main__":
    # Linear Regression

    # Regression Tree
    print('Generating Regression Tree')
    regTree = regressionTreeConstruct('Marine_Clean.csv', 10)
    # evalTree to check all possible tree depth, and choose the best depth
    evalTree('Marine_Clean.csv')
    MSE_train = getTreeMSE('Marine_Clean.csv',8)
    print("MSE for training set is %.2f" %MSE_train)
    ave_MSE_train = crossValidation('Marine_Clean.csv',10)
    print("MSE for 10 fold cv is %.2f" %ave_MSE_train)
    # can use testCaseFunction to do prediction based on previously tree, and plot the predictions
    predictions, MSE_test = testCase('TestCase.csv',regTree)
    print("MSE for testing set is %.2f" % MSE_test)
