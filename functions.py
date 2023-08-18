import numpy as np

# Calculates f and ff values of the given data and W
def ff(unlabeled_X, labeled_X, W, unlabeled_Y, labeled_Y):
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
    m, _ = W.shape                          # for looping through W
    s = []                                  # for storing row norms

    for j in range(m):
        # Calculates 2-norm of each row of W, stores them in s
        s.append(np.linalg.norm(W[j, :]))

    # Calculates the sum of all 2-norms and returns it
    return sum(s)                         

# Calculates ff with normalization on the given data
def F(unlabeled_X, labeled_X, W, unlabeled_Y, labeled_Y, norm_coeff=0.01):
    # Calculates the f value of the given data using W
    f_W, _ = ff(unlabeled_X, labeled_X, W, unlabeled_Y, labeled_Y)

    # Applies the 21-norm to the result and returns it
    return f_W + norm_coeff * L21_norm(W)

# Calculates the phi value of the given data, W and eita
def phi(unlabled_X, labled_X, unlabled_Y, labled_Y, W_t, eita_t):
    # Calculates the ff value of the given data using W at time t
    _, ff_Wt = ff(unlabled_X, labled_X, W_t, unlabled_Y, labled_Y)
    U_t = W_t - 1/eita_t * ff_Wt                        # calculates U
    W = []                                              # stores rows
    for j in range(W_t.shape[0]):
        if np.linalg.norm(U_t[j, :]) > 0.01/eita_t:
            co = 1 - 1/(eita_t * np.linalg.norm(U_t[j, :])) # weight coefficient
            W_i = co * U_t[j, :]                            # weighted U row
            W.append(W_i)
        else:
            W_i = np.zeros(U_t[j, :].shape)             # zero row
            W.append(W_i)
    
    # Returns the weighted/zeroed out rows of U, phi
    return W

# Calculates the combined trace value of the given data, L_G, H and W
def value(labeled_X, unlabeled_X, unlabeled_Y, labeled_Y, L_G, H, W, Parameter):
    # trace of Y_u*L_G*Y_u^T
    part1 = 5000000 * np.trace(np.dot(np.dot(unlabeled_Y, L_G), unlabeled_Y.T))

    # trace of H^T*H
    part2 = 0.5 * Parameter.lambda_1 * np.trace(np.dot(H.T, H))

    # trace of loss_u^T*loss_u, where loss_u is W^T*X_u-Y_u
    part3 = 0.5 * Parameter.lambda_2 * np.trace(np.dot(np.dot((W.T, unlabeled_X) - unlabeled_Y).T, (W.T, unlabeled_X) - unlabeled_Y))

    # trace of W^T*W
    part4 = 0.5 * Parameter.lambda_3 * np.trace(np.dot(W.T, W))

    # trace of loss_l^T*loss_l, where loss_l is W^T*X_l-Y_l
    part5 = 0.5 * Parameter.lambda_4 * np.trace(np.dot(np.dot((W.T, labeled_X) - labeled_Y.T).T, (W.T, labeled_X) - labeled_Y.T))

    # Returns the weighted sum of the different trace values
    return part1 + part2 + part3 + part4 + part5