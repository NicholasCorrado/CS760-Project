import math
import numpy as np
import random as rand
import scipy.stats


class LinearRegression:

    def __init__(self, x, y):
        # Including intercept, shape is nxd+1
        self.x = self.__add_intercept(x)
        self.y = y
        (self.n, self.d) = self.x.shape
        self.theta_hat = self.learn()

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

    def learn(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        # MSE estimation of theta_hat
        return np.linalg.inv(x.T @ x) @ x.T @ y

    def mean_squared_error(self, x=None, y=None, theta_hat=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if theta_hat is None:
            theta_hat = self.theta_hat
        return np.sum((y-self.predict(x, self.theta_hat))**2)/x.shape[0]

    def predict(self, x, theta_hat=None):
        if theta_hat is None:
            theta_hat = self.theta_hat
            x = self.__add_intercept(x)
        return x @ theta_hat

    def kfold(self, k=10, seed=None):
        # Set seed for repeatibility.
        if seed is not None:
            rand.seed(seed)
        # Randomize order of indicies
        rand_indicies = rand.sample(range(self.n), k=self.n)
        # Unecessary but gets the buckets as similar a size as possible
        bucket_size = round(len(rand_indicies)/k)

        total_mse = 0
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

            theta_hat = self.learn(train_x, train_y)
            total_mse += self.mean_squared_error(test_x, test_y, theta_hat)

        return total_mse/k

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

    print("MSE for dataset: {0}".format(
        lg.mean_squared_error()))

    print("Average MSE for k=10 fold CV: {0}".format(lg.kfold(seed=1)))

    significance = lg.get_significance()

    features = ["Date", "Latitude", "Longitude",
                "Water Temperature", "Sample Volume", "Bias"]
    print("Feature Significance Testing\n< 0 means insignificant, > 0 means significant.")
    for i in range(len(features)):
        print("At {}, feature {} is {}.".format(
            significance[i], features[i], "significant" if significance[i] > 0 else "insignificant"))
