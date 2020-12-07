import numpy as np
import matplotlib.pyplot as plt



class kNN:
    
    def __init__(self, path):
        
        self.data = self.__load()
        
    def __load(self):
        
        data = np.loadtxt("Marine_Clean.csv", 
                delimiter=',', usecols=(0,1,2,3,4,5), skiprows=1)
        
        # Normalize features and target
        ranges = [np.max(col)-np.min(col) for col in data.T]
        mins = [np.min(col) for col in data.T]
        ranges[-1] = 1
        data[:300,:]
        data = np.array([(data.T[i]-mins[i])/ranges[i] for i in range(len(data.T))]).T
        
        return data
        
    def __d(self, x1, x2):
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
    
    def __distance_matrix(self, X):
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
                distances[i, j] = self.__d(X[i], X[j])
                
        return distances
    
    def __compute_loss(self, D, Y, K, offset):
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
    
    def __get_block_of_distance_matrix(self, D, start, stop, n_subsets):
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

    def __run_kfold(self, D, X, Y):
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
        
        # an array of k values for which we want to compute losses
        Ks = np.arange(1,200,1)
        losses = np.zeros(len(Ks))
        r2s = np.zeros(len(Ks))
        
        subset_size = int(np.around(n/n_subsets))
    
        # index boundaries to partition our dataset. For example,
        # subset_boundaries[0]=0 is the first index of the first subset, and 
        # subset_boundaries[0]=subset_size is the last index of the first subset.
        subset_boundaries = [subset_size*k for k in range(n_subsets+1)]
        subset_boundaries[-1] = n # since n is usually not divisible by subset_size
    
        for i in range(len(Ks)):
            
            k = Ks[i]
            loss = 0
            
            for j in range(n_subsets):
                start = subset_boundaries[j]
                stop = subset_boundaries[j+1]
                
                # a block of the distance matrix containing only the distances
                # we need to test the current subset.
                Dsub = self.__get_block_of_distance_matrix(D, start, stop, n_subsets)
            
                loss += self.__compute_loss(Dsub, Y, k, start)/(stop-start)
            losses[i] = loss/n_subsets
    #        r2s[i] = 1 - np.sum(loss)/np.sum()
            
                
        return Ks, losses
        
    def __compute_yhat(self, x, K, X, Y):
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
            distances[j] = self.__d(x, X[j])
    
        # find k nearest neighbors
        sorted_indices = np.argsort(distances)
        knn_indices = sorted_indices[:K]
        
        return np.average([Y[i] for i in knn_indices])    
    
    def run_test_cases(self, k):
        '''
        Compute the target estimation for our test cases.
        
        Parameters:
        k : int
            number of neighbors used to compute our target estimate
        '''
        
        data = np.loadtxt("TestCase.csv", 
            delimiter=',', usecols=(0,1,2,3,4,5), skiprows=1)
        
        ranges = [np.max(col)-np.min(col) for col in data.T]
        mins = [np.min(col) for col in data.T]
        
        # ranges[-1] corresponds to the range of the target variable. We do not 
        # want to normalize the target, so manually set its range to 1.
        ranges[-1] = 1
        
        # normalize test case data
        data = np.array([(data.T[i]-mins[i])/ranges[i] for i in range(len(data.T))]).T
        n = len(data)
        
        Xtest = data[:,:-1]
        Ytest = data[:,-1]
        
        # load training data
        data = self.__load()
        
        X = data[:,:-1]
        Y = data[:,-1]
        
        Yhat = np.zeros(n)
        
        loss = 0
        for i in range(n):
            x = Xtest[i]    
            y = Ytest[i]
            yhat = self.__compute_yhat(x, k, X, Y)
            Yhat[i] = yhat
            loss += (y-yhat)**2
        
        r2 = 1 - loss/np.sum(Ytest**2)
        
        return loss/n, r2, Yhat
            
        
        
    def compute_training_loss(self, k):
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
        
        X = self.data[:,:-1]
        Y = self.data[:,-1]
        
        n = len(X)
        
        loss = 0
        for i in range(n):
            yhat = self.__compute_yhat(X[i], k, X, Y)
            loss += (Y[i] - yhat)**2
            
        r2 = 1 - loss/np.sum(Y**2)
            
        return loss/n, r2
    
    def compute_optimal_k(self, n_trials=1):
        
        all_losses = []
        
        for i in range(n_trials):
            
            print("running trial", i+1,"...")
            
            # shuffle data so that each trial of k-fold validation produces
            # different data subsets.
            np.random.shuffle(self.data)
            X = self.data[:,:-1]
            Y = self.data[:,-1]
            
            #@TODO: avoid recomputing distance matrix for each trial.  
            D = self.__distance_matrix(X)
            Ks, losses = self.__run_kfold(D,X,Y)
            all_losses.append(losses)
            
        avg_losses = np.average(all_losses, axis=0)
        
        # compute r squared
        Y = self.data[:,-1]
        r2 = 1 - np.sum(avg_losses)/np.sum(Y**2)
    
        
        plt.rcParams.update({'font.size': 28})
        plt.figure(figsize=(16,16))
        plt.plot(Ks, avg_losses)
        plt.ylabel("Average Loss (averaged over 5 runs of 10-fold validation)")
        plt.xlabel("k (number of neighbors used to predict target)")
        plt.title("kNN Average Loss vs. k")
        plt.savefig('loss-vs-k.pdf')
        
        k_opt = np.argmin(avg_losses)+1        
        return k_opt, r2, avg_losses[k_opt-1]


    

       

