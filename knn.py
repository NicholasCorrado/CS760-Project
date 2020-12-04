import numpy as np
import matplotlib.pyplot as plt

def load():
    
    data = np.loadtxt(
            "C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\Marine_Clean_no_missing_values.csv", 
            delimiter=',', usecols=(0,1,2,3,4,5), skiprows=1)
    
    # Normalize features and target
    ranges = [np.max(col)-np.min(col) for col in data.T]
    mins = [np.min(col) for col in data.T]
    ranges[-1] = 1

    data = np.array([(data.T[i]-mins[i])/ranges[i] for i in range(len(data.T))]).T
    
    return data
    
# Distance function: l1 norm
def d(x1, x2):
    return np.linalg.norm(x1 - x2, 1)


def plot(x, y):
    
    plt.figure(figsize=(16,16))
    plt.plot(x, y)
    plt.ylabel("average loss")
    plt.xlabel("k")
    plt.title("kNN average loss vs. k")
    
      
def distance_matrix(X):
    
    n = len(X)
    distances = np.zeros(shape=(n,n))

    for i in range(n):
        for j in range(n):
            distances[i, j] = d(X[i], X[j])
            
    return distances

def compute_loss(D, Y, K, offset):
    
    sorted_indices = np.argsort(D, axis=1)
    knn_indices = sorted_indices[:,:K]
    
    loss = 0
    
    for i in range(len(knn_indices)):
        indices = knn_indices[i]
        yhat = np.average([Y[j] for j in indices])
        y = Y[i+offset]
        loss += (yhat-y)**2
    return loss


def k_fold_validation(D, X, Y):
    
    n = len(X)
    n_subsets = 10
    
    Ks = np.arange(1,100, 1)
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
        
            Dsub = D[start:stop, :]
            if n_subsets > 1:
                Dsub = np.delete(Dsub, slice(start, stop), axis=1)
                
            loss += compute_loss(Dsub, Y, k, start)
        losses[j] = loss/n_subsets
            
    return Ks, losses
    
def compute_yhat(x, K, X, Y):
    
    n = len(X)
    distances = np.zeros(len(X))

    for j in range(n):
        distances[j] = d(x, X[j])

    sorted_indices = np.argsort(distances)
    knn_indices = sorted_indices[:K]
    
    return np.average([Y[i] for i in knn_indices])    

def test():
    
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
        yhat = compute_yhat(x, 27, X, Y)
        Yhat[i] = yhat
        
    plt.figure(figsize=(16,16))
    plt.plot(Ytest)
    plt.plot(Yhat)
    plt.ylabel("value")
    plt.xlabel("")
    plt.title("asdf")
        
    

if __name__ == "__main__":
    data = load()
    n = len(data)
    test()

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
    print(np.argmin(avg_losses)+1)
    
    
    plot(Ks, avg_losses)
