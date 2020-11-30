import math
import numpy as np
import random as rand


class LinearRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n, self.d = self.x.shape
        self.theta_hat = self.learn(self.x, self.y)

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

    def learn(self, x, y):

        # Including intercept, shape is nxd+1
        x = self.__add_intercept(x)

        # MSE estimation of theta_hat
        return np.linalg.inv(x.T @ x) @ x.T @ y

    def mean_squared_error(self, x, y, theta_hat):
        return np.sum((y-self.predict(x, theta_hat))**2)/x.shape[0]

    def predict(self, x, theta_hat):
        return self.__add_intercept(x) @ theta_hat

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
    theta_hat = lg.theta_hat
    print("Theta hat {0}".format(theta_hat))

    new_x = np.array([1.43683E+12, 73.62611667, -81.36758333, 0, 1.55])
    print(lg.predict(new_x, theta_hat))

    new_x = np.array([[1.43683E+12, 73.62611667, -81.36758333, 0, 1.55],
                      [1.43683E+12, 73.62611667, -81.36758333, 0, 1.6]])
    print(lg.predict(new_x, theta_hat))

    predictions = lg.predict(x, theta_hat)
    min_index = np.argmin(abs(y-predictions))
    print(min_index)
    print(x[min_index, :])

    print(lg.mean_squared_error(x, y, theta_hat))

    print(lg.kfold(seed=1))
