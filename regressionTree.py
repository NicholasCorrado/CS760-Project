"""
@author: John Li, zli769@wisc.edu
@regression Tree marine microplastic data
 """

from copy import copy
import numpy as np
from numpy import ndarray
from numpy.random import choice, seed
import matplotlib.pyplot as plt

# Node is used in regression tree which helps its built
class Node:
    attr_names = ("avg", "left", "right", "feature", "split", "mse")

    def __init__(self, avg=None, left=None, right=None, feature=None, split=None, mse=None):
        self.avg = avg
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.mse = mse

    def __str__(self):
        ret = []
        for attr_name in self.attr_names:
            attr = getattr(self, attr_name)
            # Describe the attribute of Node.
            if attr is None:
                continue
            if isinstance(attr, Node):
                des = "%s: Node object." % attr_name
            else:
                des = "%s: %s" % (attr_name, attr)
            ret.append(des)

        return "\n".join(ret) + "\n"

    def copy(self, node):
        """Copy the attributes of another Node.
        """
        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)

# regression tree has root, depth and rules attributes
class RegressionTree:
    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None

    def __str__(self):
        ret = []
        for i, rule in enumerate(self._rules):
            literals, avg = rule
            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % avg)
        return "\n".join(ret)

    @staticmethod
    def _expr2literal(expr: list) -> str:
        feature, operation, split = expr
        operation = ">=" if operation == 1 else "<"
        return "Feature%d %s %.4f" % (feature, operation, split)

    # to search for all split rules, use bfs to do level travese
    def get_rules(self):
        # Breadth-First Search.
        queue = [[self.root, []]]
        self._rules = []

        while queue:
            node, exprs = queue.pop(0)

            # Generate a rule when the current node is leaf node.
            if not (node.left or node.right):
                # Convert expression to text.
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.avg])

            # Expand when the current node has left child.
            if node.left:
                rule_left = copy(exprs)
                rule_left.append([node.feature, -1, node.split])
                queue.append([node.left, rule_left])

            # Expand when the current node has right child.
            if node.right:
                rule_right = copy(exprs)
                rule_right.append([node.feature, 1, node.split])
                queue.append([node.right, rule_right])

    @staticmethod
    # to get all mse of label and do split
    def _get_split_mse(col: ndarray, label: ndarray, split: float) -> Node:
        # Split label.
        label_left = label[col < split]
        label_right = label[col >= split]
        # Calculate the means of label.
        avg_left = label_left.mean()
        avg_right = label_right.mean()

        # Calculate the mse of label.
        mse = (((label_left - avg_left) ** 2).sum() +
               ((label_right - avg_right) ** 2).sum()) / len(label)

        # Create nodes to store result.
        node = Node(split=split, mse=mse)
        node.left = Node(avg_left)
        node.right = Node(avg_right)
        return node
    # to choose the best split feature and cutoff
    def _choose_split(self, col: ndarray, label: ndarray) -> Node:
        # Feature cannot be splitted if there's only one unique element.
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node

        # In case of empty split.
        unique.remove(min(unique))

        # Get split point which has min mse.
        item = map(lambda x: self._get_split_mse(col, label, x), unique)
        node = min(item, key=lambda x: x.mse)
        return node

    # choose the best feature
    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:
        # Compare the mse of each feature and choose best one.
        item = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        item = filter(lambda x: x[0].split is not None, item)

        node, feature = min(item, key=lambda x: x[0].mse, default=(Node(), None))
        node.feature = feature
        return node

    # this is the main function to put data into the tree, and get tree split
    # default depth is 10
    def fit(self, data: ndarray, label: ndarray, max_depth=10, min_samples_split=2):

        # Initialize with depth, node, indexes.
        self.root.avg = label.mean()
        queue = [(self.depth + 1, self.root, data, label)]

        # Breadth-First Search.
        while queue:
            depth, node, _data, _label = queue.pop(0)

            # Terminate loop if tree depth is more than max_depth.
            if depth > max_depth:
                depth -= 1
                break
            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements.
            nnode = self._choose_feature(_data, _label)
            if nnode.split is None:
                continue

            # Copy the attributes of _node to node.
            node.copy(nnode)

            # Put children of current node in que.
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            queue.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            queue.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        # Update tree depth and rules.
        self.depth = depth
        self.get_rules()

    def toString(decisionTree, indent=''):
        if decisionTree.depth == 1:  # leaf node
            sort_list = [[k, v] for k, v in
                         sorted(decisionTree.results.items(), key=lambda item: item[1], reverse=True)]
            best = sort_list[0]
            # print('best is ',best)
            return 'return ' + str(best[0])
        else:
            return
    # when there is only one vector can be predicted
    def predict_one(self, row: ndarray) -> float:
        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right

        return node.avg

    # this is an extension of the predict_one function
    def predict(self, data: ndarray) -> ndarray:
        return np.apply_along_axis(self.predict_one, 1, data)

def load_data(filename):
    data = np.loadtxt(filename,skiprows=1,delimiter=',')
    X,y = data[:, 0:5], data[:,5]
    return X,y

def regressionTreeConstruct(filename,depth = 8, split=False,printTree = False,random_state =200):
    """Tesing the performance of RegressionTree
    """
    # print("Tesing the performance of RegressionTree...")
    # Load data
    if split == False:
        data_train, label_train = load_data(filename)
        data_test = None
        label_test = None
    else:
        X, y = load_data(filename)
        data_train, data_test, label_train, label_test = train_test_split(
            X, y, random_state=random_state)
    # Train model
    tree = RegressionTree()
    tree.fit(data=data_train, label=label_train, max_depth=depth)
    # Show rules
    if printTree == True:
        print(tree)
    return tree

def train_test_split(data, label=None, prob=0.9, random_state=None):
    # Set random state.
    if random_state is not None:
        seed(random_state)
    # Split data
    n_rows, _ = data.shape
    k = int(n_rows * prob)
    train_indexes = choice(range(n_rows), size=k, replace=False)
    test_indexes = np.array([i for i in range(n_rows) if i not in train_indexes])
    data_train = data[train_indexes]
    data_test = data[test_indexes]

    # Split label.
    if label is not None:
        label_train = label[train_indexes]
        label_test = label[test_indexes]
        ret = (data_train, data_test, label_train, label_test)
    else:
        ret = (data_train, data_test)
    # Cancel random state.
    if random_state is not None:
        seed(None)
    return ret

def calGoodnessFit(reg, X, y):
    if isinstance(y, list):
        y = np.array(y)
    y_hat = reg.predict(X)
    if isinstance(y_hat, list):
        y_hat = np.array(y_hat)
    sse = ((y - y_hat) ** 2).mean()
    sst = y.var()
    r2 = 1 - sse / sst
    return r2,sse,sst

def _evalTree(filename,maxdepth):
    X, y = load_data(filename)
    data_train, data_test, y_train, y_test = train_test_split(
        X, y, random_state=maxdepth*10)
    regTree = regressionTreeConstruct('Marine_Clean.csv', depth = maxdepth,split=True, printTree = False,random_state = maxdepth*10)
    r2_train, sse_train, sst_train = calGoodnessFit(regTree, data_train, y_train)
    r2_test, sse_test, sst_test = calGoodnessFit(regTree,data_test,y_test)
    # print('At maxdepth %d, the goodness of fit for training data is %.2f' % (maxdepth,r2_train))
    # print('At maxdepth %d, the goodness of fit for resting data is %.2f' % (maxdepth,r2_test))
    y_hat_train = []
    x_list_train = list(data_train)
    for x in x_list_train:
        y_hat_train.append(regTree.predict_one(x))
    y_hat_test = []
    x_list_test = list(data_test)
    for x in x_list_test:
        y_hat_test.append(regTree.predict_one(x))
    MSE_train = calMse(y_train, y_hat_train)
    MSE_test = calMse(y_test, y_hat_test)
    return r2_train, r2_test, MSE_train,MSE_test

def evalTree(filename):
    r2_train_list = []
    r2_test_list = []
    for depth in range(1,16):
        r2_train,r2_test,MSE_train,MSE_test = _evalTree(filename,depth)
        r2_train_list.append(r2_train)
        r2_test_list.append(r2_test)
    depth_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    plt.plot(depth_list, r2_train_list, marker='o', label = 'training fitness', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    plt.plot(depth_list, r2_test_list, marker='o', label = 'testing fitness', markerfacecolor='red', markersize=12, color='orange', linewidth=4)
    plt.legend(loc="upper left")
    plt.xlabel("Max Depth of Regression Tree")
    plt.ylabel("Goodness of Fitness (R2)")
    plt.show()
    return

def getTreeMSE(filename,depth):
    regTree = regressionTreeConstruct('Marine_Clean.csv', depth = depth,split=False, printTree = False,random_state = 200)
    X, y = load_data(filename)
    y_hat = []
    x_list = list(X)
    for x in x_list:
        y_hat.append(regTree.predict_one(x))
    MSE_train = calMse(y, y_hat)
    return MSE_train

def crossValidation(filename, kfold,opt_depth = 8):
    mses = []
    for k in range(kfold):
        X, y = load_data(filename)
        regTree = regressionTreeConstruct('Marine_Clean.csv', depth=opt_depth, split=True, printTree=False,random_state=k)
        y_hat = []
        x_list = list(X)
        for x in x_list:
            y_hat.append(regTree.predict_one(x))
        mse_cur = calMse(y, y_hat)
        mses.append(mse_cur)
    return sum(mses) / len(mses)

def calMse(y, y_bar):
    summation = 0  # variable to store the summation of differences
    n = len(y)  # finding total number of items in list
    for i in range(0, n):  # looping through each element of the list
        difference = y[i] - y_bar[i]  # finding the difference between observed and predicted value
        squared_difference = difference ** 2  # taking square of the differene
        summation = summation + squared_difference  # taking a sum of all the differences
    MSE = summation / n  # dividing summation by total values to obtain average
    return MSE

def testCase(filename,tree):
    predictions = []
    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    X,y = data[:, 0:5],data[:,5]
    x_list = list(X)
    for x in x_list:
        predictions.append(tree.predict_one(x))
    x_axis = []
    for x in range(len(predictions)):
        x_axis.append(x+1)
    plt.plot(x_axis,predictions,'ro',label='prediction')
    plt.plot(x_axis,y,'bs',label='actual')
    plt.legend(loc="upper left")
    plt.xlabel("Sample")
    plt.ylabel("Total Microplastic Pieces")
    plt.show()
    mse_test = calMse(y,predictions)
    return predictions, mse_test

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