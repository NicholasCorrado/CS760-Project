import numpy as np
from linearRegression import *
from regressionTree import *
from knn import *

if __name__ == "__main__":
    input_file ='Marine_Clean.csv'
    test_file = 'TestCase.csv'
    # Linear Regression

    # Regression Tree
    print("-----------------------------------------------------------------------------------")
    print('Generating Regression Tree')
    regTree = regressionTreeConstruct(input_file, 8) # input parameters (filename,tree max depth)
    print("-----------------------------------------------------------------------------------")
    # evalTree to check all possible tree depth, and choose the best depth
    evalTree('Marine_Clean.csv')
    MSE_train = getTreeMSE(input_file,8) # getTreeMSE(filename, optimized_depth)
    print("MSE for training set is %.2f" %MSE_train)
    ave_MSE_train = crossValidation(input_file,10)
    print("average MSE from 10 fold cv is %.2f" %ave_MSE_train)
    R2_train, R2_test = getTreeR2(input_file,8)
    print("Best Model: R2 goodness of fit for trainset is %.2f" %R2_train)
    print("Best Model: R2 goodness of fit for testset is %.2f" %R2_test)

    # can use testCaseFunction to do prediction based on previously tree, and plot the predictions
    print("-----------------------------------------------------------------------------------")
    predictions, MSE_test = testCase(test_file,regTree) # testcase(test_file_name, regressionTree)
    print("MSE for testing set is %.2f" % MSE_test)
    R2_predict_training = predictionGoodnessOfFit(regTree,input_file)
    print("R2 for training set is %.2f" % R2_predict_training)
    R2_predict_test = predictionGoodnessOfFit(regTree,test_file)
    print("R2 for testing set is %.2f" % R2_predict_test)
    print("-----------------------------------------------------------------------------------")
    
    
    # kNN
#    knn = kNN(input_file)
    knn = kNN("C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\Marine_Clean_no_missing_values.csv")
    print('Estimating optimal k for kNN regression...')
    k_opt, r2_kfold, loss_kfold = knn.compute_optimal_k()
    print('Optimal k =', k_opt)
    print('Running 10-fold validation 5 times...')
    loss_test, r2_test = knn.run_test_cases(k_opt)
    print('Computing training loss')
    loss_training, r2_training = knn.compute_training_loss(k_opt)

    print("Optimal k =", k_opt)
    print("avarage 10-fold validation loss =", loss_kfold)
    print("r^2 10-fold validation =", r2_kfold)
    print("training loss =", loss_training)
    print("r^2 for training set =", r2_training)
    print("test loss =", loss_test)
    print("r^2 for test set =", r2_test)
