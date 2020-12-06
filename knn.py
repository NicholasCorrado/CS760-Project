import numpy as np
import matplotlib.pyplot as plt

def load():
    
    data = np.loadtxt(
            "C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\Marine_Clean_no_missing_values.csv", 
            delimiter=',', usecols=(0,1,2,3,4,5), skiprows=1)
    
    data = data[:200,:]
    
    # Normalize features and target
    ranges = [np.max(col)-np.min(col) for col in data.T]
    mins = [np.min(col) for col in data.T]
    ranges[-1] = 1

    data = np.array([(data.T[i]-mins[i])/ranges[i] for i in range(len(data.T))]).T
    
    
    return data
    
# Distance function: l2 norm
def d(x1, x2):
    '''
    Compute the l2 distance between x1 and x2
    
    Parameters:
    x1 : 1D array
        a feature vector
    x2 : 1D array
        a feature vector
        
    Returns:
    int : l2 distance between x1 and x2     
    '''
    return np.linalg.norm(x1 - x2, 2)


def plot(x, y):
    
    plt.rcParams.update({'font.size': 28})
    plt.figure(figsize=(16,16))
    plt.plot(x, y)
    plt.ylabel("Average Loss (averaged over 5 runs of 10-fold validation)")
    plt.xlabel("k (number of neighbors used to predict target)")
    plt.title("kNN Average Loss vs. k")
    
      
def distance_matrix(X):
    '''
    Compute a distance matrix D where D[i,j] corresponds to the distance
    between X[i] and X[j]
    
    Parameters:
        X : 2D array
            all feature vectors in the dataset.
    
    Returns:
    2D array : distance matrix as described above  
    '''
    
    n = len(X)
    distances = np.zeros(shape=(n,n))

    for i in range(n):
        for j in range(n):
            distances[i, j] = d(X[i], X[j])
            
    return distances

def compute_loss(D, Y, K, offset):
    '''
    Compute the mean squared error loss using k 
    
    D : 2D array
        distance matrix 
    Y : 1D array
        true target values
    K : int 
        number of neighbors used to compute our target estimate yhat
    offset : int
        index corresponding to the "start" index used to create the D parameter
        (see the docstring for get_block_of_distance_matrix). This is just an 
        implementation detail.
        
    Returns:
    float : mean squared error loss 
    '''
    
    sorted_indices = np.argsort(D, axis=1)
    knn_indices = sorted_indices[:,:K]
    
    loss = 0
    
    for i in range(len(knn_indices)):
        indices = knn_indices[i]
        yhat = np.average([Y[j] for j in indices])
        y = Y[i+offset]
        loss += (yhat-y)**2
    return loss

def get_block_of_distance_matrix(D, start, stop, n_subsets):
    '''
    Return a block/slice of the distance matrix D containing only the rows 
    from start to stop (including start, excluding stop), excluding the columns 
    from start to stop. 
    
    For example, say D is a 6x6 matrix, and let start=2 and stop=4. Then a 2x4
    matrix will be returned. The matrix below has 1 in each entry that is 
    included in the returned matrix and a 0 for all excluded entries. 
    
    0 0 0 0 0 0
    0 0 0 0 0 0
    1 1 0 0 1 1
    1 1 0 0 1 1
    0 0 0 0 0 0
    0 0 0 0 0 0
    
    D : 2D array
        distance matrix 
    start : int
        index of the first feature vector we want to include in our matrix 
        block
    stop : int
        index of the first feature vector for which we want to exclude from our
        matrix block.
    n_subsets : int
        number of subsets used in k-fold validiation (fixed to 10)
    
    Returns:
    2D array: a block of the inputted D matrix as specified above.
    '''
    
    Dsub = D[start:stop, :]
    if n_subsets > 1:
        Dsub = np.delete(Dsub, slice(start, stop), axis=1)
    return Dsub
    


def k_fold_validation(D, X, Y):
    '''
    Run 10-fold validation for all k between 1 and 200.
    
    Parameters:
    
    D : 2D array
        distance matrix 
    X : 2D array
        matrix containing all feature vectors in the dataset
    Y : 1D array
        all true target values in the dataset
    Returns:
    1D int array : an array of all k values for which the loss was computed
    1D float array: mean square error loss averaged over 10 test subsets for 
    all values of k between 1 and 200
    '''
    n = len(X)
    n_subsets = 10
    
    Ks = np.arange(1,200, 1)
    losses = np.zeros(len(Ks))
    
    subset_size = int(np.around(n/n_subsets))

    subset_boundaries = [subset_size*k for k in range(n_subsets+1)]
    subset_boundaries[-1] = n #adjust last index

    for j in range(len(Ks)):
        k = Ks[j]
        loss = 0
        for i in range(n_subsets):
            start = subset_boundaries[i]
            stop = subset_boundaries[i+1]
            
            Dsub = get_block_of_distance_matrix(D, start, stop, n_subsets)
        

                
            loss += compute_loss(Dsub, Y, k, start)/(stop-start)
        losses[j] = loss/n_subsets
            
    return Ks, losses
    
def compute_yhat(x, K, X, Y):
    '''
    Compute the target estimate for input vector x as the average of the the 
    true target values for the k nearest neighbors.

    Parameters:
    x : 1D array
        input feature vector for which we are estimating the target
    k : int
        number of neighbors used to compute our target estimate
    X : 2D array
        matrix containing all feature vectors in the dataset
    Y : 1D array
        all true target values in the dataset
        
    Returns: 
    float: target estimate for input vector x
    '''
    
    n = len(X)
    distances = np.zeros(len(X))

    for j in range(n):
        distances[j] = d(x, X[j])

    sorted_indices = np.argsort(distances)
    knn_indices = sorted_indices[:K]
    
    return np.average([Y[i] for i in knn_indices])    

def test(k):
    
    data = np.loadtxt(
        "C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\TestCase.csv", 
        delimiter=',', usecols=(0,1,2,3,4,5), skiprows=1)
    
    ranges = [np.max(col)-np.min(col) for col in data.T]
    mins = [np.min(col) for col in data.T]
    ranges[-1] = 1

    data = np.array([(data.T[i]-mins[i])/ranges[i] for i in range(len(data.T))]).T
    
    Xtest = data[:,:-1]
    Ytest = data[:,-1]
    
    data = load()
    X = data[:,:-1]
    Y = data[:,-1]
    
    Yhat = np.zeros(len(Ytest))
    
    for i in range(len(Xtest)):
        x = Xtest[i]
        y = Ytest[i]        
        yhat = compute_yhat(x, k, X, Y)
        Yhat[i] = yhat
        
    plt.figure(figsize=(16,16))
    plt.plot(Ytest)
    plt.plot(Yhat)
    plt.ylabel("value")
    plt.xlabel("")
    plt.title("asdf")
        
    
    
def compute_training_loss(k, X, Y):
    '''
    Compute the empircal loss on the training data. The optimal k value 
    should be used.
    
    Parameters:
    k : int
        number of neighbors used to compute our target estimate
    X : 2D array
        matrix containing all feature vectors in the dataset
    Y : 1D array
        all true target values in the dataset
        
    Returns:
    float : mean squared error loss
    '''
    n = len(X)
    
    loss = 0
    for i in range(n):
        yhat = compute_yhat(X[i], k, X, Y)
        loss += (Y[i] - yhat)**2
        
    return loss/n
    

if __name__ == "__main__":
    data = load()
    n = len(data)
    print(n)
    test(21)
    
    X = data[:,:-1]
    Y = data[:,-1]
    l = compute_training_loss(28, X, Y)
    print("training loss =", l)
#    for k in range(1,50,5):
#        test(k)

    n_trials = 1
    all_losses = []
    
    for i in range(n_trials):
        print("running trial", i+1,"...")
        # shuffle data so that each trial of k-fold validation produces
        # different data subsets.
        np.random.shuffle(data)
        X = data[:,:-1]
        Y = data[:,-1]
        #@TODO: avoid recomputing distance matrix for each trial.  
        D = distance_matrix(X)
        Ks, losses = k_fold_validation(D,X,Y)
        all_losses.append(losses)
       
    avg_losses = np.average(all_losses, axis=0)
    plot(Ks, avg_losses)
    print(Ks)
    print(avg_losses)
    print(np.argmin(avg_losses)+1)
