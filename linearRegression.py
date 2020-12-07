import math
import numpy as np
import random as rand
import scipy.stats


class LinearRegression:

    def __init__(self, x, y, bias=True):
        if bias:
            # Including intercept, shape is nxd+1
            self.x = self.__add_intercept(x)
        else:
            # Shape is nxd
            self.x = x

        # Shape is n,
        self.y = y
        # These are usefull
        (self.n, self.d) = self.x.shape
        # Learn the coefficients that minimize the MSE
        self.theta_hat = self.__learn()

    def get_theta_hat(self):
        return self.theta_hat

    def __add_intercept(self, x):
        if len(np.shape(x)) == 1:
            x = np.append(x, 1)
        else:
            x = np.append(
                x,
                np.ones((x.shape[0], 1)),
                axis=1
            )
        return x

    def __learn(self):
        # MSE estimation of theta_hat
        return np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y

    def predict(self, x, bias=True):
        if bias:
            x = self.__add_intercept(x)
        return x @ self.theta_hat

    def kfold(self, k=10, seed=None):
        # Set seed for repeatibility.
        if seed is not None:
            rand.seed(seed)
        # Randomize order of indicies
        rand_indicies = rand.sample(range(self.n), k=self.n)
        # Unecessary but gets the buckets as similar a size as possible
        bucket_size = round(len(rand_indicies)/k)

        total_mse = 0
        total_r_squared = 0
        for i in range(k):
            if i == k-1:
                train_bucket_indicies = rand_indicies[:i*bucket_size]
                test_bucket_indicies = rand_indicies[i*bucket_size:]
            else:
                train_bucket_indicies = \
                    rand_indicies[:i * bucket_size] + \
                    rand_indicies[(i+1) * bucket_size:]
                test_bucket_indicies = \
                    rand_indicies[i * bucket_size:(i+1) * bucket_size]

            train_x = self.x[train_bucket_indicies, :]
            train_y = self.y[train_bucket_indicies]
            test_x = self.x[test_bucket_indicies, :]
            test_y = self.y[test_bucket_indicies]

            temp_model = LinearRegression(train_x, train_y, False)
            sum_squared_error_prediction = sum(
                (test_y-temp_model.predict(test_x, False))**2
            )
            total_mse += sum_squared_error_prediction/test_x.shape[0]

            y_mean = sum(train_y)/len(train_y)
            sum_squared_error_mean = sum((test_y-y_mean)**2)

            total_r_squared += 1 - \
                (sum_squared_error_prediction/sum_squared_error_mean)

        return (total_mse/k, total_r_squared/k)

    def get_significance(self, alpha=0.05):

        half = y-(self.x @ self.theta_hat)
        sigma2 = (half.T @ half)/self.n

        # only 1-alpha to work with this package
        inv_tail = scipy.stats.chi2(1).ppf(1-alpha)

        covar_theta = sigma2*np.linalg.inv(self.x.T@self.x)

        results = []
        for i in range(len(self.theta_hat)):
            results.append(
                ((self.theta_hat[i]**2)/covar_theta[i][i])-inv_tail)

        return np.array(results)


if __name__ == "__main__":
    np.set_printoptions(edgeitems=20, linewidth=100000,
                        precision=3)
    input_file = "Marine_Clean.csv"
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
    print("MSE for dataset: {}\nR Squared for dataset: {}".format(
        mse, r_squared))

    k_fold_mse, k_fold_r2 = lg.kfold(seed=1)
    print("Average MSE for k=10 fold CV: {}\nAverage R Squared for k=10 fold CV: {}".format(
        k_fold_mse, k_fold_r2))

    test_data = np.genfromtxt("TestCase.csv", delimiter=",",
                              skip_header=1)

    # shape is n,
    test_y = test_data[:, 5:6].reshape(-1)
    # shape is nxd
    test_x = test_data[:, :5]
    print("Results for test cases:")
    print("Test case predictions {}".format(lg.predict(test_x)))
    print("MSE for test set {}".format(
        sum((test_y-lg.predict(test_x))**2)/test_x.shape[0]))

    significance = lg.get_significance()
    features = ["Date", "Latitude", "Longitude",
                "Water Temperature", "Sample Volume", "Bias"]
    print("Feature Significance Testing\n< 0 means insignificant, > 0 means significant.")
    for i in range(len(features)):
        print("At {}, feature {} is {}.".format(
            significance[i], features[i], "significant" if significance[i] > 0 else "insignificant"))
