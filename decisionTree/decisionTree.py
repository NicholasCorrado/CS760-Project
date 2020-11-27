# -*- coding: utf-8 -*-
"""
@author: John Li, zli769@wisc.edu
 """

import numpy as np
from ppbtree import *  # The pretty print package for decision tree
Labels = {0: 'Date', 1: 'Latitude', 2: 'Longitude', 3: 'Temperature'}
OutcomeLabels = {0: 'No/Few Microplastic', 1: 'More Microplstic'}

class DecisionTree:
    def __init__(self, feature=-1, trueBranch=None, falseBranch=None, results=None, value=0.5, label=None):
        self.feature = feature
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results  # None for non-leave nodes, not-None for leave node (with all information we need), update results when grow the tree
        self.value = value
        self.label = label
class KFolds:
    """
    Parameters
    ----------
    n_splits : int
        number of folds. Must be at least 2

    shuffle : bool, default True
        whether to shuffle the data before splitting into batches

    seed : int, default 6666
        When shuffle = True, pseudo-random number generator state used for
        shuffling; this ensures reproducibility
    """

    def __init__(self, n_splits, shuffle=True, seed=6666):
        self.seed = seed
        self.shuffle = shuffle
        self.n_splits = n_splits

    def split(self, X):
        """pass in the data to create train/test split for k fold"""
        # shuffle modifies indices inplace
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            rstate = np.random.RandomState(self.seed)
            rstate.shuffle(indices)

        for test_mask in self._iter_test_masks(n_samples, indices):
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield train_index, test_index

    def _iter_test_masks(self, n_samples, indices):
        """
        create the mask for the test set, then the indices that
        are not in the test set belongs in the training set
        """
        # indicate the number of samples in each fold, and also
        # make sure the ones that are not evenly splitted also
        # gets assigned to a fold (e.g. if we do 2 fold on a
        # dataset that has 5 samples, then 1 will be left out,
        # and has to be assigned to one of the other fold)
        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=np.int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test_mask = np.zeros(n_samples, dtype=np.bool)
            test_mask[test_indices] = True
            yield test_mask
            current = stop

# trees at github.com/clemtoy/pptree
# Read data from titanic_data.txt file
def init_data(filename):
    data = np.loadtxt(filename,skiprows=1,delimiter=",")
    data[:,-1] = (data[:,-1]>2).astype(int) # is plastic is more than 2 per liter, we regard this as positive (more micorplastic)
    data = data[:,[0,1,2,3,6]]
    feature= data[:,0:3]
    label = data[:,-1]
    return data

def compute_entropy(data):
    """Data can be any list of values."""
    # Count frequency of each value for probability distribution
    values, freqs = np.unique(data, return_counts=True)
    # Turn freqs into float type to avoid integer division in Python 2
    freqs = np.array(freqs, dtype=float)
    entropy = 0
    # If there's only 1 value then entropy is 0
    if len(values) > 1:
        for i in range(len(values)):
            if freqs[i] != 0:
                entropy -= freqs[i] / np.sum(freqs) * np.log2(freqs[i] / np.sum(freqs))
    return entropy

def conditional_entropy(x, y):
    """Compute conditional entropy of binary array of independent variables
    x on binary array of dependent variables (y).
    So x is a feature vector, y is an outcome vector."""
    values, freqs = np.unique(y, return_counts=True)
    freqs = np.array(freqs, dtype=float)
    entropy = 0
    for i in range(len(values)):
        if freqs[i] != 0:
            entropy += compute_entropy(x[np.where(y == i)]) * freqs[i] / np.sum(freqs)
    return entropy

def mutual_information(x, y):
    return compute_entropy(x) - conditional_entropy(x, y)

def process_data(filename):
    data = init_data(filename)
    N, D = data.shape
    outcomes = data[:,-1]
    features = data[:,:D-1]
    initial_mutual_info = []  # will hold mutual info of each feature and outcomes
    cutoff_mutual_info = []  # will hold mutual info after cutting off for comparison
    cutoffs = []  # will hold cutoff values
    for i in range(D - 1):
        initial_mutual_info.append(mutual_information(features[:, i], outcomes))
        # Keep track of mutual info difference.  Initialize to -1
        diff = 0
        # Loop through possible cutoffs
        values = np.unique(features[:, i])
        # Initialize best cut-off to lowest value
        best_cutoff = values[0]
        for value in values:
            # Create binary array with 0's where >= value, 0's where < value
            binary = 1 * np.greater_equal(features[:, i], value)
            # Compute mutual info with outcome
            mut_inf = mutual_information(binary, outcomes)
            # Check if this is the best yet:
            if mut_inf > diff:
                diff = mut_inf
                best_cutoff = value
        cutoffs.append(best_cutoff)
        cutoff_mutual_info.append(diff)

    features_binary = 1 * np.greater_equal(features, cutoffs)
    data_binary = np.append(features_binary, np.array([outcomes]).T, axis=1)
    return cutoffs,features_binary,data_binary

# Rewrote split data to keep type as numpy.array:
def splitData(records, feature, value):
    trueData = records[records[:, feature] > value]
    falseData = records[records[:, feature] <= value]
    return trueData, falseData

def growTree(data, Labels, OutcomeLabels, cutoffs, sufficientData=1):
    '''data is an array of all measurements, meaning features and outcome
    for some subset of passengers on titanic.
    Here the stopping condition is that at least some portion (like 5% or 0.1%) of the
    original data is still being used.  This condition is passed in as the argument
    sufficientData.  If sufficientData is not specified then the stopping condition
    is simply that all features at a node have 0 mutual information with the
    remaining outcomes.'''
    N, D = data.shape
    features = data[:, :D - 1]  # the features only
    outcomes = data[:, -1]  # the outcomes only
    bestGain = 0.0  # track mutual information of each feature
    bestFeature = None  # track feature with best mutual info
    bestDataSets = None  # trueSet, falseSet for best feature
    # Check stopping condition:
    if N < sufficientData:
        # Take whichever outcome is most frequent for data at this stage of tree
        vals, freqs = np.unique(outcomes, return_counts=True)
        result = vals[np.argmax(freqs)]
        return DecisionTree(results=result, label=OutcomeLabels[int(result)])

    # If there is sufficient data, find feature with highest mutual info:
    for feature in range(D - 1):
        if mutual_information(features[:, feature], outcomes) > bestGain:
            bestGain = mutual_information(features[:, feature], outcomes)
            bestFeature = feature
            bestDataSets = splitData(data, feature, 0.5)
    if bestGain > 0:
        trueBranch = growTree(bestDataSets[0], Labels, OutcomeLabels, cutoffs, sufficientData)
        falseBranch = growTree(bestDataSets[1], Labels, OutcomeLabels, cutoffs, sufficientData)
        # Make an informative label for the pretty print display later:
        label = Labels[bestFeature] + ' over/under ' + str(cutoffs[bestFeature])
        return DecisionTree(feature=bestFeature, trueBranch=trueBranch, falseBranch=falseBranch, label=label)
    else:  # when no best variable is found return original tree without branch and with result
        # figure out what that result is:
        # Take whichever outcome is most frequent for data at this stage of tree
        vals, freqs = np.unique(outcomes, return_counts=True)
        result = vals[np.argmax(freqs)]
        return DecisionTree(results=result, label=OutcomeLabels[int(result)])

def classify(testData, tree):
    if tree == None:
        return
    if tree.results != None:  # leaf node
        return tree.results
    else:
        inputValue = testData[tree.feature]
        if inputValue > tree.value:
            branch = tree.trueBranch
        else:
            branch = tree.falseBranch
    return classify(testData, branch)

def predictByTree (myvector,filename):
    tree = initial_tree(filename)
    cutoffs, feaures_binary, data_binary = process_data(filename)

    my_vector = np.array(myvector)
    # change vectors to binary:
    my_binary = 1 * np.greater_equal(my_vector, cutoffs)

    my_result = classify(my_binary, tree)
    return my_result,OutcomeLabels[my_result]


def initial_tree(filename, printTree = False, stopcreterial = 0.001):
    cutoffs, feaures_binary, data_binary = process_data(filename)
    stopping_creteria = 0.001 * len(data_binary)
    myTree = growTree(data_binary, Labels,OutcomeLabels, cutoffs, sufficientData=stopping_creteria)
    # Use print_tree from the pptree package imported above
    if printTree == True:
        print_tree(myTree, nameattr='label', left_child='trueBranch', right_child='falseBranch')
    return myTree


# if selection = 1, generate random forest by deleting one feature for each tree
# else if selection = 2, generate random forest by selecting 80% data for each tree
def growForest(filename, selection = 1, print_forest_Tree=False):
    cutoffs, feaures_binary, data = process_data(filename)
    # Assumes data has outcomes as the last column, so in
    N, D = data.shape
    # the number of features is D-1.
    forest = []
    if selection == 1:
        for i in range(D - 1):
            # set feature i to 0:
            tempData = np.array(data)
            tempData[:, i] = np.zeros(N)
            # grow a tree:
            tree = growTree(tempData,Labels,OutcomeLabels,cutoffs)
            forest.append(tree)
    else:
        kf5 = KFolds(n_splits=5)
        for train_index, test_index in kf5.split(data):
            tree = growTree(data[train_index, :],Labels,OutcomeLabels,cutoffs)
            forest.append(tree)
    return forest

def predictByForest(myvector,filename,selection=1):
    cutoffs, feaures_binary, data_binary = process_data(filename)
    my_vector = np.array(myvector)
    my_binary = 1 * np.greater_equal(my_vector, cutoffs)
    forest = growForest(filename,selection)
    lengthOfForest = len(forest)

    forest_prediction = 0.0
    for tree in forest:
        forest_prediction += classify(my_binary,tree)

    my_result = 0.0
    if forest_prediction > lengthOfForest/2:
        my_result = 1.0
        return my_result, OutcomeLabels[my_result]
    else:
        return my_result, OutcomeLabels[my_result]


k = 10
kf = KFolds(n_splits=k)
nPredictions = 0
nIncorrect = 0
cutoffs, feature_binary, data_binary = process_data('Marine_Clean.csv')

for train_index, test_index in kf.split(data_binary):
    tree = growTree(data_binary[train_index, :],Labels,OutcomeLabels,cutoffs)
    for index in test_index:
        prediction = classify(data_binary[index, :-1], tree)
        nPredictions += 1
        nIncorrect += (prediction - data_binary[index, -1]) ** 2
print('Accuracy in {}-fold verification is {:.2f} percent'.format(k, 100 - nIncorrect * 100.0 / nPredictions))
