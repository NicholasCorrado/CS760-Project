# -*- coding: utf-8 -*-
"""
@author: John Li, zli769@wisc.edu
 """

# -*- coding: utf-8 -*-

from copy import copy
import numpy as np
from numpy import ndarray
from numpy.random import choice, seed
import matplotlib.pyplot as plt

class Node:
    """Node class to build tree leaves.
    Attributes:
        avg {float} -- prediction of label. (default: {None})
        left {Node} -- Left child node.
        right {Node} -- Right child node.
        feature {int} -- Column index.
        split {int} --  Split point.
        mse {float} --  Mean square error.
    """

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
        Arguments:
            node {Node}
        """

        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class RegressionTree:
    """RegressionTree class.
    Attributes:
        root {Node} -- Root node of RegressionTree.
        depth {int} -- Depth of RegressionTree.
        _rules {list} -- Rules of all the tree nodes.
    """

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
        """Auxiliary function of get_rules.
        Arguments:
            expr {list} -- 1D list like [Feature, op, split].
        Returns:
            str
        """

        feature, operation, split = expr
        operation = ">=" if operation == 1 else "<"
        return "Feature%d %s %.4f" % (feature, operation, split)

    def get_rules(self):
        """Get the rules of all the tree nodes.
            Expr: 1D list like [Feature, op, split].
            Rule: 2D list like [[Feature, op, split], label].
            Op: -1 means less than, 1 means equal or more than.
        """

        # Breadth-First Search.
        que = [[self.root, []]]
        self._rules = []

        while que:
            node, exprs = que.pop(0)

            # Generate a rule when the current node is leaf node.
            if not (node.left or node.right):
                # Convert expression to text.
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.avg])

            # Expand when the current node has left child.
            if node.left:
                rule_left = copy(exprs)
                rule_left.append([node.feature, -1, node.split])
                que.append([node.left, rule_left])

            # Expand when the current node has right child.
            if node.right:
                rule_right = copy(exprs)
                rule_right.append([node.feature, 1, node.split])
                que.append([node.right, rule_right])

    @staticmethod
    def _get_split_mse(col: ndarray, label: ndarray, split: float) -> Node:
        """Calculate the mse of label when col is splitted into two pieces.
        MSE as Loss fuction:
        y_hat = Sum(y_i) / n, i <- [1, n]
        Loss(y_hat, y) = Sum((y_hat - y_i) ^ 2), i <- [1, n]
        --------------------------------------------------------------------
        Arguments:
            col {ndarray} -- A feature of training data.
            label {ndarray} -- Target values.
            split {float} -- Split point of column.
        Returns:
            Node -- MSE of label and average of splitted x
        """

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

    def _choose_split(self, col: ndarray, label: ndarray) -> Node:
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get minimum mse.
        Arguments:
            col {ndarray} -- A feature of training data.
            label {ndarray} -- Target values.
        Returns:
            Node -- The best choice of mse, split point and average.
        """

        # Feature cannot be splitted if there's only one unique element.
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node

        # In case of empty split.
        unique.remove(min(unique))

        # Get split point which has min mse.
        ite = map(lambda x: self._get_split_mse(col, label, x), unique)
        node = min(ite, key=lambda x: x.mse)

        return node

    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:
        """Choose the feature which has minimum mse.
        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.
        Returns:
            Node -- feature number, split point, average.
        """

        # Compare the mse of each feature and choose best one.
        _ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, _ite)

        # Return None if no feature can be splitted.
        node, feature = min(
            ite, key=lambda x: x[0].mse, default=(Node(), None))
        node.feature = feature

        return node

    def fit(self, data: ndarray, label: ndarray, max_depth=5, min_samples_split=2):
        """Build a regression decision tree.
        Note:
            At least there's one column in data has more than 2 unique elements,
            and label cannot be all the same value.
        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.
        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {5})
            min_samples_split {int} -- The minimum number of samples required
            to split an internal node. (default: {2})
        """

        # Initialize with depth, node, indexes.
        self.root.avg = label.mean()
        que = [(self.depth + 1, self.root, data, label)]

        # Breadth-First Search.
        while que:
            depth, node, _data, _label = que.pop(0)

            # Terminate loop if tree depth is more than max_depth.
            if depth > max_depth:
                depth -= 1
                break

            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements.
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue

            # Copy the attributes of _node to node.
            node.copy(_node)

            # Put children of current node in que.
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        # Update tree depth and rules.
        self.depth = depth
        self.get_rules()

    def predict_one(self, row: ndarray) -> float:
        """Auxiliary function of predict.
        Arguments:
            row {ndarray} -- A sample of testing data.
        Returns:
            float -- Prediction of label.
        """
        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right

        return node.avg

    def predict(self, data: ndarray) -> ndarray:
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Prediction of label.
        """

        return np.apply_along_axis(self.predict_one, 1, data)

def load_data(filename):
    data = np.loadtxt(filename,skiprows=1,delimiter=',')
    X,y = data[:, 0:4], data[:,5]
    return X,y

def regressionTreeConstruct(filename,depth = 8, split=False,printTree = True):
    """Tesing the performance of RegressionTree
    """
    print("Tesing the performance of RegressionTree...")
    # Load data
    if split == False:
        data_train, label_train = load_data(filename)
        data_test = None
        label_test = None
    else:
        X, y = load_data(filename)
        data_train, data_test, label_train, label_test = train_test_split(
            X, y, random_state=200)
    # Train model
    tree = RegressionTree()
    tree.fit(data=data_train, label=label_train, max_depth=depth)
    # Show rules
    if printTree == True:
        print(tree)
    return tree

def train_test_split(data, label=None, prob=0.95, random_state=None):
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
    return r2

def _evalTree(filename,maxdepth):
    X, y = load_data(filename)
    data_train, data_test, y_train, y_test = train_test_split(
        X, y, random_state=200)
    regTree = regressionTreeConstruct('Marine_Clean.csv', depth = maxdepth,split=True, printTree = False)
    r2_train = calGoodnessFit(regTree, data_train, y_train)
    r2_test = calGoodnessFit(regTree,data_test,y_test)
    print('At maxdepth %d, the goodness of fit for training data is %.2f' % (maxdepth,r2_train))
    print('At maxdepth %d, the goodness of fit for resting data is %.2f' % (maxdepth,r2_test))
    return r2_train, r2_test

def evalTree(filename):
    x1,y1 = _evalTree(filename,7)
    x2,y2 = _evalTree(filename,8)
    x3,y3 = _evalTree(filename,9)
    x4,y4 = _evalTree(filename,10)
    x5,y5 = _evalTree(filename,11)
    x6,y6 = _evalTree(filename,12)

    depth = [7,8,9,10,11,12]
    r2_train = [x1,x2,x3,x4,x5,x6]
    r2_test = [y1,y2,y3,y4,y5,y6]
    plt.plot(depth, r2_train, marker='o', label = 'training fitness', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    plt.plot(depth, r2_test, marker='o', label = 'testing fitness', markerfacecolor='red', markersize=12, color='orange', linewidth=4)
    plt.legend(loc="upper left")
    plt.xlabel("Max Depth of Regression Tree")
    plt.ylabel("Goodness of Fitness (R2)")
    plt.show()
    return
def testCase(filename,tree):
    predictions = []
    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    X,y = data[:, 0:4],data[:,-1]
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
    return predictions

if __name__ == "__main__":
    regTree = regressionTreeConstruct('Marine_Clean.csv', 10)

    evalTree('Marine_Clean.csv')

    predictions = testCase('TestCase.csv',regTree)

