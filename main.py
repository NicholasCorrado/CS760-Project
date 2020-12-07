import numpy as np
from linearRegression import *
from regressionTree import *
from knn import *

if __name__ == "__main__":
    input_file = 'Marine_Clean.csv'
    test_file = 'TestCase.csv'
    print("-----------------------------------------------------------------------------------")
    print('Linear Regression')
    print("-----------------------------------------------------------------------------------")
    # Linear Regression
    allData = np.genfromtxt(input_file, delimiter=",",
                            skip_header=1)

    # shape is n,
    y = allData[:, 5:6].reshape(-1)
    # shape is nxd
    x = allData[:, :5]

    lg = LinearRegression(x, y)
    print("Theta hat {0}".format(lg.get_theta_hat()))

    sum_squared_error_prediction = sum(
        (y-lg.predict(x))**2
    )
    mse = sum_squared_error_prediction/x.shape[0]

    y_mean = sum(y)/len(y)
    sum_squared_error_mean = sum((y-y_mean)**2)

    r_squared = 1 - (sum_squared_error_prediction/sum_squared_error_mean)
    print("MSE for train set: {}\nR Squared for train set: {}".format(
        mse, r_squared))

    k_fold_mse, k_fold_r2 = lg.kfold(seed=1)
    print("Average MSE for k=10 fold CV: {}\nAverage R Squared for k=10 fold CV: {}".format(
        k_fold_mse, k_fold_r2))

    test_data = np.genfromtxt(test_file, delimiter=",",
                              skip_header=1)

    # shape is n,
    test_y = test_data[:, 5:6].reshape(-1)
    # shape is nxd
    test_x = test_data[:, :5]
    print("Results for test cases:")
    linear_regression_test_predictions = lg.predict(test_x)
    print("Test case predictions {}".format(
        linear_regression_test_predictions))
    print("MSE for test set {}".format(
        sum((test_y-lg.predict(test_x))**2)/test_x.shape[0]))

    significance = lg.get_significance()
    features = ["Date", "Latitude", "Longitude",
                "Water Temperature", "Sample Volume", "Bias"]
    print("Feature Significance Testing\n< 0 means insignificant, > 0 means significant.")
    for i in range(len(features)):
        print("At {}, feature {} is {}.".format(
            significance[i], features[i], "significant" if significance[i] > 0 else "insignificant"))

    print("-----------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------")

    # Regression Tree
    print("-----------------------------------------------------------------------------------")
    print('Generating Regression Tree')
    # input parameters (filename,tree max depth)
    regTree = regressionTreeConstruct(input_file, 8)
    print("-----------------------------------------------------------------------------------")
    # evalTree to check all possible tree depth, and choose the best depth
    evalTree('Marine_Clean.csv')
    # getTreeMSE(filename, optimized_depth)
    MSE_train = getTreeMSE(input_file, 8)
    print("MSE for training set is %.2f" % MSE_train)
    ave_MSE_train = crossValidation(input_file, 10)
    print("average MSE from 10 fold cv is %.2f" % ave_MSE_train)
    R2_train, R2_test = getTreeR2(input_file, 8)
    print("Best Model: R2 goodness of fit for trainset is %.2f" % R2_train)
    print("Best Model: R2 goodness of fit for testset is %.2f" % R2_test)

    # can use testCaseFunction to do prediction based on previously tree, and plot the predictions
    print("-----------------------------------------------------------------------------------")
    # testcase(test_file_name, regressionTree)
    predictions, MSE_test = testCase(test_file, regTree)
    print("MSE for testing set is %.2f" % MSE_test)
    R2_predict_training = predictionGoodnessOfFit(regTree, input_file)
    print("R2 for training set is %.2f" % R2_predict_training)
    R2_predict_test = predictionGoodnessOfFit(regTree, test_file)
    print("R2 for testing set is %.2f" % R2_predict_test)
    print("-----------------------------------------------------------------------------------")

    # kNN
    knn = kNN(
        "C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\Marine_Clean_no_missing_values.csv")
    print('Estimating optimal k for kNN regression...\n')
    k_opt, r2_kfold, loss_kfold = knn.compute_optimal_k()
    print('\nOptimal k =', k_opt)
    print("average 10-fold validation loss =", loss_kfold)
    print("r^2 10-fold validation =", r2_kfold)
    print("-----------------------------------------------------------------------------------")
    print('Computing training loss...')
    loss_training, r2_training = knn.compute_training_loss(k_opt)
    print("training loss =", loss_training)
    print("r^2 for training set =", r2_training)
    print("-----------------------------------------------------------------------------------")
    loss_test, r2_test, Yhat_knn = knn.run_test_cases(k_opt)
    print("test loss =", loss_test)
    print("r^2 for test set =", r2_test)
    print("-----------------------------------------------------------------------------------")
    
    # kNN predications for the test case data is stored in the variable Yhat_knn

