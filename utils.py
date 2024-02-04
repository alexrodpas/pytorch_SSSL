import numpy as np
import torch

# Calculates the spectral time series similarity of the given data and sigma
def spectral_timeseries_similarity(X, sigma):
    n, _ = X.shape                              # For looping through X
    D_G = torch.zeros((n, n))                   # Initializing the diagonal
    G = torch.zeros((n, n))                     # Initializing similarity matrix
    
    # Loops through X to calculate norm, similarity matrices
    for i in range(n):
        for j in range(i, n):
            g = torch.linalg.norm(X[i] - X[j])          # 2-norm between timeseries
            G[i, j] = torch.exp(-g**2 / sigma**2)       # Calculates G entry
            G[j, i] = G[i, j]                           # Ensures symmetry
        
        D_G[i, i] = torch.sum(G[i])             # Calculates diagonal value
    
    L_G = D_G - G                               # Calculates similarity matrix
    
    # Returns similarity matrix
    return L_G

# Calculates the distance between a long series and a short series, using alpha
def distance_longseries_shortseries(series_long, series_short, alpha):
    l = len(series_short)                   # for looping through short series
    num_seg = len(series_long) - l + 1      # for looping through long series
    dijq_skp = torch.zeros((num_seg, l))    # for storing dijq_skp
    dijq = torch.zeros(num_seg)             # for storing d_i,j,q

    # Loops through all possible segments and stores distances, norms
    for q in range(num_seg):
        segment = series_long[q:q+l]  # the q-th segment of the long series
        dijq_skp[q] = (2/l * (series_short - segment))                  # stores dijq_skp
        dijq[q] = (1/l * torch.linalg.norm(series_short - segment)**2)  # stores d_i,j,q
    
    big_lambda = torch.sum(torch.mul(dijq, torch.exp(alpha * dijq)))    # /\
    big_theta = torch.sum(torch.exp(alpha * dijq))                      # (-)
    X = big_lambda/big_theta    # the minimum distance between the series_long and series_short

    #exp_sum = torch.mul(torch.exp(alpha * dijq), (1 + alpha * dijq) * big_theta - alpha * big_lambda)
    #Xij_skp = (1/(big_theta**2) * torch.sum(torch.dot(exp_sum, dijq_skp)))  # for storing Xij_skp

    # Returns distance between series, derivative array
    return X#, Xij_skp

# Calculates the distances between given time series T and shapelets S, using alpha
def distance_timeseries_shapelet(T, lengths, S, alpha):
    mT, _ = T.shape                         # for looping through T
    mS, nS = S.shape                        # for looping through S
    #Xij_skp = torch.zeros((mS, mT, nS-1))   # for storing derivatives
    X = torch.zeros((mT, mS))               # for storing distances
    
    # Loops through T and S and calculates distances
    for i in range(mT): # the j-th time series
        TS = T[i]                               # time series to compare
        for j in range(mS):
            shapelet = S[j, 0:int(lengths[j])]  # shapelet to compare

            # Calculates the distance between the time series and the shapelet
            X[i, j] = distance_longseries_shortseries(TS, shapelet, alpha)
            #Xij_skp[i, j, 0:int(S[i, 0])] = Xij_sk  # stores derivative
    
    # Returns distances
    return X#, Xij_skp

# Calculates the similarity between the given shapelets S, using alpha and sigma
def shapelet_similarity(lengths, S, alpha, sigma):
    m, n = S.shape                      # for looping through S
    # Hij_skp = np.zeros((m, m, n))       # for storing derivatives
    H = torch.zeros((m, m))             # for storing similarity
    XS = torch.zeros((m, m))            # for storing distances
    
    # Loops through shapelets and calculates their similarities
    for i in range(m):
        length_s = lengths[i]
        sh_s = S[i, :length_s]          # the i-th shapelet
        for j in range(i, m):
            length_l = lengths[j]
            sh_l = S[j, :length_l]      # the j-th shapelet
            XS[i, j] = distance_longseries_shortseries(sh_l, sh_s, alpha) # distance between the i-th shapelet and the j-th shapelet
            XS[j, i] = XS[i, j]         # ensures symmetry

            # Calculates the similarity matrix of shapelets
            H[i, j] = torch.exp(-(XS[i, j]/sigma)**2)
            H[j, i] = H[i, j]           # ensures symmetry

            # Calculates the derivative of H_(ij) on S_(il)
            #Hij_skp[i, j, :length_s] = H[i, j] * (-1/sigma**2 * XS[i, j]) *  XSij_si
            #Hij_skp[j, i, :length_s] = Hij_skp[i, j, :length_s]    # ensures symmetry
    
    # Returns similarity matrix
    return H# , XS, Hij_skp

# Clusters the data into C centroids using the given epsilon
def EM(Data, C, epsilon=0.1):
    m, n = Data.shape                   # for looping through data
    
    # Initializes centroids
    C_t = np.max(Data) * np.random.rand(m, C)   # random centroid initialization
    C_tp1 = np.zeros((m, C))                    # update variable
    v = 2 * epsilon                             # loop termination variable
    
    # Loops through the update script until the target value is reached
    while v > epsilon:
        Y = np.zeros((C, n))            # cluster assignment matrix
        
        # Calculates distances between data and centroids and assigns clusters
        for i in range(n):
            distance = np.zeros(C)      # for calculating data-centroid distances
            
            for j in range(C):
                distance[j] = np.linalg.norm(Data[:, i] - C_t[:, j])    # calculates distance to each centroid
            
            # Assigns each data point to the centroid that has the smallest distance to it
            index = np.argmin(distance)
            Y[index, i] = 1
        
        # Calculates the optimal centroids based on the current clusters
        for i in range(C):
            # Failsafe for faulty cluster assignment
            if np.size((np.where(Y[i, :] == 1))) > 0:
                #index = np.random.randint(n)
            #else:
                index = np.where(Y[i, :] == 1)[0]
            
                # Calculates the mean of the data on the selected cluster
                C_tp1[:, i] = np.mean(Data[:, index], axis=1)
        
        # Calculates the trace if the difference between the current centroids and optimal centroids
        v = np.trace(np.dot((C_tp1 - C_t).T, C_tp1 - C_t))
        C_t = C_tp1                     # updates centroids
    
    # Returns the calculated centroids and cluster assignments
    return C_t, Y

# Clusters the data into C centroids using the given epsilon
def kmeans(Data, C, epsilon=0.001):
    m, n = Data.shape                   # for looping through data
    
    # Initializes centroids
    C_t = np.max(Data) * np.random.rand(m, C)   # random centroid initialization
    C_tp1 = np.zeros((m, C))                    # update variable
    v = 2 * epsilon                             # loop termination variable
    
    while v > epsilon:
        # Cluster data based on current centroids
        Y = np.zeros((C, n))            # cluster assignment matrix
        for i in range(n):
            distance = np.zeros(C)      # for calculating data-centroid distances
            for j in range(C):
                distance[j] = np.linalg.norm(Data[:, i] - C_t[:, j])    # calculates distance to each centroid
            
            # Assigns each data point to the centroid that has the smallest distance to it
            index = np.argmin(distance)
            Y[index, i] = 1
        
        # Calculates the optimal centroids based on the current clusters
        for i in range(C):
            # Failsafe for faulty cluster assignment
            if len(np.where(Y[i, :] == 1)) == 0:
                index = np.random.randint(n)
            else:
                index = np.where(Y[i, :] == 1)[0]
            
            # Calculates the mean of the data on the selected cluster
            C_tp1[:, i] = np.mean(Data[:, index], axis=1)
        
        # Calculates the trace if the difference between the current centroids and optimal centroids
        v = np.trace(np.dot((C_tp1 - C_t).T, C_tp1 - C_t))
        C_t = C_tp1                     # updates centroids
    
    # Returns the calculated centroids and cluster assignments
    return C_t, Y

# Obtains all segments of T with a specific length
def obtain_segments(T, length):
    L = T[:, 0].astype(int)     # length column
    DT = T[:, 1:]               # time series data
    m, _ = DT.shape             # for looping through DT
    segment_matrix = []         # for storing the segments

    # Loops through L
    for j in range(m):
        len = L[j] - length + 1
        # Loops through each possible segment in DT[j]
        for k in range(len):
            segment = DT[j, k:k + length]
            segment_matrix.append(segment)  # records segment

    # Returns a matrix where each row is a segment of T
    return np.array(segment_matrix)

# Initialize shapelets
def s_initialization(parameters):
    S = np.zeros((parameters['k'] * parameters['R'], parameters['R'] * parameters['Lmin'])) # shapelets
    L = np.zeros(parameters['k'] * parameters['R']).astype(int)

    # Creates k shapelets for each of R lengths and saves them to S
    for j in range(parameters['R']):
        length = (j + 1) * parameters['Lmin']       # shapelet length
        L[j * parameters['k']:(j + 1) * parameters['k']] = length   # records shapelet length
    
    # Returns the saved shapelets with their lengths
    return L, S

# Reshapes the Y_true matrix into a one-hot encoding
def reshape_y_true(Y_true, C):
    Y_true[np.where(Y_true < 0)] = C-1      # corrects invalid values
    m = len(Y_true)                         # for looping through Y_true
    Y_true_matrix = np.zeros((m, C), dtype=int)        # for storing one-hot encoding

    # Loops through Y_true values
    for i in range(m):
        Y_true_matrix[i][Y_true[i]] = 1     # one-hot encoding of Y_true[i]
    
    # Returns one-hot encoding of Y_true
    return Y_true_matrix

# Applies z-regularization to the given data
def z_regularization(Data):
    # Returns regularized data
    return (Data - np.min(Data)) / (np.max(Data) - np.min(Data))