import numpy as np
from linearRegression import *
from regressionTree import *


if __name__ == "__main__":
    input_file ='Marine_Clean.csv'
    test_file = 'TestCase.csv'
    # Linear Regression

    # Regression Tree
    print("-----------------------------------------------------------------------------------")
    print('Generating Regression Tree')
    regTree = regressionTreeConstruct(input_file, 8) # input parameters (filename,tree max depth)
    print("-----------------------------------------------------------------------------------")
    # # evalTree to check all possible tree depth, and choose the best depth
    evalTree('Marine_Clean.csv')
    MSE_train = getTreeMSE(input_file,8) # getTreeMSE(filename, optimized_depth)
    print("MSE for training set is %.2f" %MSE_train)
    ave_MSE_train, ave_r2 = crossValidation(input_file,10)
    print("average MSE from 10 fold cv is %.2f" %ave_MSE_train)
    print("average R2 from 10 fold cv is %.2f" %ave_r2)
    # R2_train, R2_test = getTreeR2(input_file,8)
    # print("Best Model: R2 goodness of fit for trainset is %.2f" %R2_train)
    # print("Best Model: R2 goodness of fit for testset is %.2f" %R2_test)
    R2_predict_training = predictionGoodnessOfFitForTraining(regTree,input_file)
    print("R2 for training set is %.2f" % R2_predict_training)
    # can use testCaseFunction to do prediction based on previously tree, and plot the predictions
    print("-----------------------------------------------------------------------------------")
    predictions, MSE_test = testCase(test_file,regTree) # testcase(test_file_name, regressionTree)
    print("MSE for testing set is %.2f" % MSE_test)
    R2_predict_test = predictionGoodnessOfFitForTesting(regTree,test_file)
    print("R2 for testing set is %.2f" % R2_predict_test)
    print("-----------------------------------------------------------------------------------")

