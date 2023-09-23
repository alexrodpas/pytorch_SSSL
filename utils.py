import numpy as np

# Calculates the spectral time series similarity of the given data and sigma
def spectral_timeseries_similarity(X, sigma):
    _, n = X.shape                              # For looping through X
    D_G = np.zeros((n, n))                      # Initializing the diagonal
    G = np.zeros((n, n))                        # Initializing norm matrix
    
    # Loops through X to calculate norm, similarity matrices
    for j in range(n):
        for h in range(j, n):
            g = np.linalg.norm(X[:, j] - X[:, h])   # 2-norm between columns
            G[j, h] = np.exp(-g**2 / sigma**2)      # Calculates G entry
            G[h, j] = G[j, h]                       # Ensures symmetry
        
        D_G[j, j] = np.sum(G[j, :])             # Calculates diagonal value
    
    L_G = D_G - G                               # Calculates similarity matrix
    
    # Returns similarity, norm matrices
    return L_G, G

# Calculates the distance between a long series and a short series, using alpha
def distance_longseries_shortseries(series_long, series_short, alpha):
    m = len(series_short)                   # for looping through short series
    num_seg = len(series_long) - m + 1      # for looping through long series
    D1 = []                                 # for storing distances
    D2 = []                                 # for storing squared distance norms

    # Loops through all possible segments and stores distances, norms
    for q in range(num_seg):
        segment = series_long[q:q+m]  # the q-th segment of the long series
        D2.append(1/m * np.linalg.norm(series_short - segment)**2)  # stores average |v|_2^2
        D1.append(1/m * (series_short - segment))                   # stores distance
    
    D1 = np.array(D1)                       # numpy array conversion
    D2 = np.array(D2)                       # numpy array conversion
    X_1 = np.sum(np.multiply(D2, np.exp(alpha * D2)))   # weighted sum of norms
    X_2 = np.sum(np.exp(alpha * D2))        # exponential sum of norms
    X = X_1/X_2  # the distance between the series_long and series_short
    Xkj_sk = []                             # for storing derivatives

    # Loops through the short sequence to calculate distances, derivatives
    for l in range(m):
        part1 = 1/X_2**2                    # inverse of squared exponential sum of norms
        part2 = D1[:, l]                    # l moment in each segment
        part3 = np.multiply(np.exp(alpha * D2), X_2 * (1 + alpha * D2) - alpha * X_1) # coefficient
        Xkj_sk.append(part1 * np.sum(np.multiply(part2, part3)))  # derivative of X_(kj) on S_(kl)

    # Returns distance between series, derivative array
    return X, np.array(Xkj_sk)

# Calculates the distances between given time series T and shapelets S, using alpha
def distance_timeseries_shapelet(T, S, alpha):
    mT, _ = T.shape                     # for looping through T
    DT = T[:, 1:]                       # all but the first column of T
    mS, nS = S.shape                    # for looping through S
    DS = S[:, 1:]                       # all but the first column of S
    Xkj_skl = np.zeros((mS, mT, nS-1))  # for storing derivatives
    X = np.zeros((mS, mT))              # for storing distances
    
    # Loops through T and S and calculates distances
    for j in range(mT): # the j-th time series
        for k in range(mS):
            shapelet = DS[k, 0:int(S[k, 0])]        # shapelet to compare
            time_series = DT[j, 0:int(T[j, 0])]     # time series to compare

            # Calculates the distance between the time series and the shapelet
            X[k, j], Xkj_sk = distance_longseries_shortseries(time_series, shapelet, alpha)
            Xkj_skl[k, j, 0:int(S[k, 0])] = Xkj_sk  # stores derivative
    
    # Returns distance and derivative matrices
    return X, Xkj_skl

# Calculates the similarity between the given shapelets S, using alpha and sigma
def shapelet_similarity(S, alpha, sigma):
    m, n = S.shape                      # for looping through S
    DS = S[:, 1:]                       # all but the first column of S
    Hij_sil = np.zeros((m, m, n-1))     # for storing derivatives
    H = np.zeros((m, m))                # for storing similarity
    XS = np.zeros((m, m))               # for storing distances
    
    # Loops through shapelets and calculates their similarities
    for i in range(m):
        length_s = S[i, 0].astype(int)
        sh_s = DS[i, :length_s]         # the i-th shapelet
        for j in range(i, m):
            length_l = S[j, 0].astype(int)
            sh_l = DS[j, :length_l]     # the j-th shapelet
            XS[i, j], XSij_si = distance_longseries_shortseries(sh_l, sh_s, alpha) # distance between the i-th shapelet and the j-th shapelet
            XS[j, i] = XS[i, j]         # ensures symmetry

            # Calculates the similarity matrix of shapelets
            H[i, j] = np.exp(-(XS[i, j]/sigma)**2)
            H[j, i] = H[i, j]           # ensures symmetry

            # Calculates the derivative of H_(ij) on S_(il)
            Hij_sil[i, j, :length_s] = H[i, j] * (-2/sigma**2 * XS[i, j]) *  XSij_si
            Hij_sil[j, i, :length_s] = Hij_sil[i, j, :length_s]    # ensures symmetry
    
    # Returns similarity, distance and derivative matrices
    return H, XS, Hij_sil

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

# Obtains a segment of T with a specific length
def obtain_segment(T, length):
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

def s_initialization(T, parameters):
    S = np.zeros((parameters['k'] * parameters['R'], 1 + parameters['R'] * parameters['Lmin'])) # shapelets

    # Creates k shapelets for each of R lengths and saves them to S
    for j in range(parameters['R']):
        length = (j + 1) * parameters['Lmin']       # shapelet length
        segment_matrix = obtain_segment(T, length)  # time series segments
        DS = np.dot(np.ones((parameters['k'], 1)), np.mean(segment_matrix, axis=0).reshape(1, segment_matrix.shape[1])) # initializes shapelets as mean time series values
        # DS, _ = EM(segment_matrix.T, parameters['k'])
        S[j * parameters['k']:(j + 1) * parameters['k'], 0] = length            # records shapelet length
        S[j * parameters['k']:(j + 1) * parameters['k'], 1:length + 1] = DS     # saves shapelets
    
    # Returns the saved shapelets with their lengths
    return S

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