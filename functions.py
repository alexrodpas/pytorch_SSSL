import numpy as np

# Calculates f and ff values of the given data and W
def ff_function(unlabeled_X, labeled_X, W, unlabeled_Y, labeled_Y):
    # Initializes variables
    M1 = np.dot(unlabeled_X.T, W) - unlabeled_Y.T
    M2 = np.dot(labeled_X.T, W) - labeled_Y

    # Calculates f and ff values
    f_W = 50000 * np.linalg.norm(M1, 'fro')**2 + 5000 * np.linalg.norm(W, 'fro')**2 + 50000 * np.linalg.norm(M2, 'fro')**2
    ff_W = 10000 * np.dot(labeled_X, labeled_X.T) * W - np.dot(labeled_X, labeled_Y) + 10000 * np.dot(unlabeled_X, unlabeled_X.T) * W - np.dot(unlabeled_X, unlabeled_Y.T) + W

    # Returns calculated values
    return f_W, ff_W

# Calculates the L-21 norm of input tensor W
def L21_norm(W):
    m, _ = W.shape                          # For looping through W
    s = []                                  # For storing row norms

    for j in range(m):
        # Calculates 2-norm of each row of W, stores them in s
        s.append(np.linalg.norm(W[j, :]))

    # Calculates the sum of all 2-norms and returns it
    return sum(s)                         

# Helper function for calculating ff with normalization on the given data
def F(unlabeled_X, labeled_X, W, unlabeled_Y, labeled_Y, norm_coeff=0.01):
    # Calculates the f value of the given data using W
    f_W, _ = ff_function(unlabeled_X, labeled_X, W, unlabeled_Y, labeled_Y)

    # Applies the 21-norm to the result and returns it
    return f_W + norm_coeff * L21_norm(W)

# Calculates the phi value of the given data, W and eita
def phi(unlabled_X, labled_X, unlabled_Y, labled_Y, W_t, eita_t):
    # Calculates the ff value of the given data using W at time t
    _, ff_Wt = ff_function(unlabled_X, labled_X, W_t, unlabled_Y, labled_Y)
    U_t = W_t - 1/eita_t * ff_Wt                        # Calculates U
    W = []                                              # Stores rows
    for j in range(W_t.shape[0]):
        if np.linalg.norm(U_t[j, :]) > 0.01/eita_t:
            co = 1 - 1/(eita_t * np.linalg.norm(U_t[j, :])) # Weight coefficient
            W_i = co * U_t[j, :]                            # Weighted U row
            W.append(W_i)
        else:
            W_i = np.zeros(U_t[j, :].shape)             # Zero row
            W.append(W_i)
    
    # Returns the weighted/zeroed out rows of U, phi
    return W

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