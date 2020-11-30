import numpy as np
import matplotlib.pyplot as plt

def load():
    
    data = np.loadtxt(
            "C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\Marine_Clean.csv", 
            delimiter=',', usecols=(1,2,4,6), skiprows=1)
    
    # The data is currently sorted by pieces/L. Shuffling the data now will 
    # help us whenever we do k-fold validation.
    np.random.shuffle(data)
    #data = data[:100]
    
    # Normalize features and target
    ranges = [np.max(col)-np.min(col) for col in data.T]
    data = np.array([data.T[i]/ranges[i] for i in range(len(data.T))]).T

    Xdata = data[:,:3]
    Ydata = data[:,3]
    
    return Xdata, Ydata
    
# Distance function: l1 norm
def d(x1, x2):
    return np.linalg.norm(x1 - x2, 1)

# Output an estimate of pieces/L using an average of the k nearest neighbors.
def knn(x, K,  X, Y):
    
    n = len(X)
    distances = np.zeros(len(X))

    for j in range(n):
    
        distances[j] = d(x, X[j])

    sorted_indices = np.argsort(distances)
    knn_indices = sorted_indices[:K]
    
    s = np.sum([Y[i] for i in knn_indices])
    return s/K
    
# Compute the loss for a data set using a test data set
def compute_loss(K, X, Y, Xtest, Ytest):
    loss = 0

    for x,y in zip(Xtest,Ytest):
        loss += (knn(x, K, X, Y)-y)**2
            
    return loss

def k_fold_validation(Xdata, Ydata, N_folds):
    
    Ks = [1,5,10,15,20,25,30,35,40,45,50,60,70,80,100,150,200]
    #Ks = [1,2]
    losses = np.zeros(len(Ks))
    
    N = len(Ydata)

    subset_size = int(np.around(len(Xdata)/N_folds))
    
    Xj = [Xdata[i:i + subset_size] for i in range(0, N, subset_size)]
    Yj = [Ydata[i:i + subset_size] for i in range(0, N, subset_size)]
    
    for s in range(len(Ks)):
        k = Ks[s]
        print("computing loss for k =", k)
        loss = 0
        for i in range(N_folds):
    
            if i == 0:
                X = Xj[1]
                Y = Yj[1]
            else:
                X = Xj[0]
                Y = Yj[0]
            for j in range(1, N_folds):
                if (i != j):
                    X = np.concatenate((X, Xj[j]))
                    Y = np.concatenate((Y, Yj[j]))
                
    
            loss += compute_loss(k, X, Y, Xj[i], Yj[i])
    
        losses[s] = loss/N_folds
        #print("avg loss for k =",k, loss/N_folds)
        
    return Ks, losses

def plot(x, y):
    
    plt.figure(figsize=(16,16))
    plt.plot(x, y)
    plt.ylabel("average loss")
    plt.xlabel("k")
    plt.title("kNN average loss vs. k")

if __name__ == "__main__":
    X, Y = load()
    Ks, losses = k_fold_validation(X, Y, 10)
    plot(Ks, losses)
    #print(Ks)
    #print(losses)   
    