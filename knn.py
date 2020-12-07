import numpy as np
import matplotlib.pyplot as plt



class kNN:
    
    def __init__(self, data):
        
        ranges = [np.max(col)-np.min(col) for col in data.T]
        mins = [np.min(col) for col in data.T]
        ranges[-1] = 1
        
        self.data = np.array([(data.T[i]-mins[i])/ranges[i] for i in range(len(data.T))]).T
        
    def d(self, x1, x2):
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
    
    def distance_matrix(self, X):
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
                distances[i, j] = self.d(X[i], X[j])
                
        return distances
    
    def compute_loss(self, D, Y, K, offset):
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
    
    def get_block_of_distance_matrix(self, D, start, stop, n_subsets):
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

    def k_fold_validation(self, D, X, Y):
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
                Dsub = self.get_block_of_distance_matrix(D, start, stop, n_subsets)
            
                loss += self.compute_loss(Dsub, Y, k, start)/(stop-start)
            losses[i] = loss/n_subsets
    #        r2s[i] = 1 - np.sum(loss)/np.sum()
            
                
        return Ks, losses
        
    def compute_yhat(self, x, K, X, Y):
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
            distances[j] = self.d(x, X[j])
    
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
        
        data = np.loadtxt(
            "C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\TestCase.csv", 
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
        data = load()
        
        X = data[:,:-1]
        Y = data[:,-1]
        
        Yhat = np.zeros(n)
        
        loss = 0
        for i in range(n):
            x = Xtest[i]    
            y = Ytest[i]
            yhat = self.compute_yhat(x, k, X, Y)
            Yhat[i] = yhat
            loss += (y-yhat)**2
        
        r2 = 1 - loss/np.sum(Ytest**2)
            
        plt.figure(figsize=(16,16))
        plt.plot(Ytest, label="true")
        plt.plot(Yhat, label="estimated")
        plt.ylabel("Target Value")
        plt.xlabel("Test Vector Index")
        plt.title("Target Estimations and True Target Values for Test Vectors")
        plt.legend()
        plt.show()
        
        return loss/n, r2
            
        
        
    def compute_training_loss(self, k, X, Y):
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
            yhat = self.compute_yhat(X[i], k, X, Y)
            loss += (Y[i] - yhat)**2
            
        r2 = 1 - loss/np.sum(Y**2)
            
        return loss/n, r2
    
    def compute_optimal_k(self, data):
        
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
            D = self.distance_matrix(X)
            Ks, losses = self.k_fold_validation(D,X,Y)
            all_losses.append(losses)
            
        avg_losses = np.average(all_losses, axis=0)
        
        # compute r squared
        Y = data[:,-1]
        r2 = 1 - np.sum(avg_losses)/np.sum(Y**2)
    
        
        plt.rcParams.update({'font.size': 28})
        plt.figure(figsize=(16,16))
        plt.plot(Ks, avg_losses)
        plt.ylabel("Average Loss (averaged over 5 runs of 10-fold validation)")
        plt.xlabel("k (number of neighbors used to predict target)")
        plt.title("kNN Average Loss vs. k")
        plt.show()
        
        k_opt = np.argmin(avg_losses)+1        
        return k_opt, r2, avg_losses[k_opt-1]
    
    
def load():
    
    data = np.loadtxt(
            "C:\\Nicholas\\Graduate\\Courses\\cs760\\project\\Marine_Clean_no_missing_values.csv", 
            delimiter=',', usecols=(0,1,2,3,4,5), skiprows=1)
    
    data = data[:300,:]
    
    # Normalize features and target
    ranges = [np.max(col)-np.min(col) for col in data.T]
    mins = [np.min(col) for col in data.T]
    ranges[-1] = 1

    data = np.array([(data.T[i]-mins[i])/ranges[i] for i in range(len(data.T))]).T
    
    return data
    
# Distance function: l2 norm
 
      




    



def offline_plotting():
        
    x = np.arange(1,200)
    y = [1694.16174757, 1261.70522577, 1090.44315782, 1003.31907035,  982.49041775,
  957.75387151,  926.17332056,  908.27283595,  905.48626203,  901.3321792 ,
  893.77598178,  898.80745447,  899.36321124,  891.28761779,  888.76273562,
  889.08363043,  886.76903875,  887.09294545,  886.55858719,  882.06248852,
  882.49377824,  880.07653894,  880.08082252,  880.17480504,  878.82811865,
  878.51476542,  876.96815022,  874.46550405,  874.59057317,  876.76508653,
  878.07110682,  879.69217008,  881.62843966,  882.64528892,  882.2241835 ,
  881.15002724,  880.19010837,  881.01727451,  881.38174772,  883.42456541,
  885.18756049,  885.19050725,  885.85102556,  885.84002503,  887.38141673,
  888.03355228,  888.78159839,  887.88924628,  887.08199447,  886.87783434,
  886.34344533,  886.27115893,  886.29321419,  886.47635649,  887.13539667,
  887.58503881,  886.96445509,  887.46298245,  886.86170628,  887.32273015,
  887.31628323,  886.66318956,  885.29724096,  885.76272996,  886.24910697,
  884.71911396,  884.25257552,  884.32356645,  884.46415761,  884.23584082,
  884.91754426,  884.28438498,  883.64274958,  883.99489805,  884.21089093,
  883.90736391,  883.96509744,  883.10376245,  883.3486938 ,  883.78937916,
  884.75405769,  884.89812423,  884.58729356,  884.8123255 ,  884.72511533,
  884.79334739,  885.36234436,  885.30383477,  885.68587263,  886.26442709,
  886.48144508,  886.43204701,  886.50706052,  886.02994939,  886.33895973,
  886.62577939,  887.04520897,  887.64604782,  887.39046002,  887.38873633,
  887.89233572,  888.27399963,  888.58526768,  888.52155288,  887.86244488,
  887.7456184 ,  887.78869449,  887.7886995 ,  887.98691628,  887.75284128,
  887.7559329 ,  887.74811189,  887.94735859,  887.92650029,  887.99004944,
  887.90101394,  888.12435591,  888.02499097,  888.03964057,  888.06068841,
  887.65822993,  887.81310625,  888.09787335,  887.99528382,  887.91543748,
  888.29259808,  888.79952644,  889.31159713,  889.40455429,  889.68510688,
  889.84982066,  890.01194375,  890.04505275,  890.24077016,  890.61367201,
  890.60859403,  891.17400163,  891.46454686,  891.66952047,  892.0982862 ,
  892.38519883,  892.76201321,  892.98253532,  893.40144441,  893.50577707,
  893.82431735,  893.41187265,  893.82287791,  893.37074972,  893.56820195,
  893.76542759,  893.71809967,  893.89737223,  893.57236169,  893.68868158,
  893.73112593,  893.11009691,  893.17513451,  892.97292056,  893.2832108 ,
  893.39017599,  893.41044614,  893.6967711 ,  893.75791293,  893.91042   ,
  893.7601543 ,  894.08173927,  894.09849597,  894.32525403,  894.24365061,
  894.56075757,  894.37216349,  894.57760624,  894.52609821,  894.75045419,
  894.69181878,  894.87373669,  894.89731263,  895.13161695,  894.66546693,
  894.84471779,  895.14000933,  895.10675327,  895.00837066,  895.2155527 ,
  895.27802899,  895.76241021,  895.64989305,  895.44607354,  895.50561068,
  895.57082866,  895.73160551,  895.65807289,  895.77902484,  895.36922232,
  895.54441751,  895.59280999,  895.47669617,  895.9146615 ]
    print(np.argmin(y)+1)
    plt.figure(figsize=(16,16))
    plt.plot(x, y)
    plt.ylabel("Average Loss (averaged over 5 runs of 10-fold validation)")
    plt.xlabel("k (number of neighbors used to predict target)")
    plt.title("kNN Average Loss vs. k")
    plt.legend()
    plt.ylim(850,1000)
    plt.show()
    
    plt.savefig('loss-vs-k.pdf')  

if __name__ == "__main__":    
    
#    offline_plotting()
    

    data = load()
    
    knn = kNN(data)
    
#    loss_test, r2_test = knn.run_test_cases(28)
##    loss_test, r2_test = run_test_cases(28)
#    print("test loss =", loss_test)
#    print("r^2 for test set =", r2_test)

    # k_opt is usually 28
    k_opt, r2_kfold, loss_kfold = knn.compute_optimal_k(data)
    loss_test, r2_test = knn.run_test_cases(k_opt)
#    k_opt, r2_kfold, loss_kfold = compute_optimal_k(data)
#    loss_test, r2_test = run_test_cases(k_opt)
    
    X = data[:,:-1]
    Y = data[:,-1]
    loss_training, r2_training = knn.compute_training_loss(k_opt, X, Y)
#    loss_training, r2_training = compute_training_loss(k_opt, X, Y)
    
    print("Optimal k =", k_opt)
    print("avarage 10-fold validation loss =", loss_kfold)
    print("r^2 10-fold validation =", r2_kfold)
    print("training loss =", loss_training)
    print("r^2 for training set =", r2_training)
    print("test loss =", loss_test)
    print("r^2 for test set =", r2_test)


    

       

