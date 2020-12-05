import numpy as np
from linearRegression import *
from regressionTree import *


if __name__ == "__main__":
    input_file ='Marine_Clean.csv'
    test_file = 'TestCase.csv'
    # Linear Regression

	
	
    # Regression Tree
    print('Generating Regression Tree')
    regTree = regressionTreeConstruct(input_file, 8) # input parameters (filename,tree max depth)
    # evalTree to check all possible tree depth, and choose the best depth
    evalTree('Marine_Clean.csv')
    MSE_train = getTreeMSE(input_file,8) # getTreeMSE(filename, optimized_depth)
    print("MSE for training set is %.2f" %MSE_train)
    ave_MSE_train = crossValidation(input_file,10)
    print("MSE for 10 fold cv is %.2f" %ave_MSE_train)
    # can use testCaseFunction to do prediction based on previously tree, and plot the predictions
    predictions, MSE_test = testCase(test_file,regTree) # testcase(test_file_name, regressionTree)
    print("MSE for testing set is %.2f" % MSE_test)
	
	
	# kNN
