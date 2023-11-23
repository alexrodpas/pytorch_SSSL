import numpy as np

# Calculates the S derivative of the given data, W, shapelets, G, H and additional variables and parameters
def S_derivation(Z, Y, un_X, l_X, W, G, S, un_Xij_skp, l_Xij_skp, Hij_skp, H, parameters):
    DShape_t = S[:, 1:]                   # extracts shapelets
    mG, nG = G.shape                            # for looping through G
    mDShape, nDShape = DShape_t.shape           # for looping through shapelets
    mZ, nZ = Z.shape                            # for looping through Y

    Part1 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 1 values
    Part2 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 2 values
    Part3 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 3 values
    Part4 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 3 values

    parameter1 = 0.5 * np.matmul(Z.T, Z)                                            # parameter for weighted calculations
    parameter2 = parameters['lambda_1'] * H                                        # parameter for weighted calculations
    parameter3 = parameters['lambda_3'] * np.matmul(W, (np.matmul(W.T, un_X) - Z))  # parameter for weighted calculations
    parameter4 = parameters['lambda_4'] * np.matmul(W, (np.matmul(W.T, l_X) - Y))   # parameter for weighted calculations

    Gij_skp = np.zeros((mG, nG, mDShape, nDShape))   # for storing ST_(ij) derivatives with respect to S_(kl)

    # Loops through shapelets and calculates each part
    for k in range(mDShape):
        length = S[k, 0].astype(int)      # unlabeled shapelet length
        for p in range(length):
            for i in range(mG):
                for j in range(nG):
                    # Calculates G_(ij) derivative with respect to S_(kp)
                    Gij_skp[i, j, k, p] = (-2 * G[i, j]/parameters['sigma']**2) * np.sum(np.multiply((un_X[:, i] - un_X[:, j]), (un_Xij_skp[:, i, p] - un_Xij_skp[:, j, p])))
            
            Part1[k, p] = np.sum(np.matmul(parameter1, Gij_skp[:, :, k, p]))    # Part 1: 1/2 Z^TZ * derivative
            Part2[k, p] = np.sum(np.matmul(parameter2[k, :], Hij_skp[k, :, p])) # Part 2: weighted normalized shapelet derivatives
            Part3[k, p] = np.sum(np.matmul(parameter3, un_Xij_skp[k, :, p]))    # Part 3: shapelet derivatives
            Part4[k, p] = np.sum(np.matmul(parameter4, l_Xij_skp[k, :, p]))     # Part 4: shapelet derivatives

    # Returns the combination of all parts
    return Part1 + Part2 + Part3 + Part4

# Calculates the lS derivative with the given X and Y, W, shapelets at time t, and additional variables and parameters
def lS_derivation(Y, X, W, Shape_t, Xkj_sk1, H_skp, SS, parameters):
    DShape_t = Shape_t[:, 1:]                   # extracts shapelets
    mDShape, nDShape = DShape_t.shape           # for looping through shapelets
    mY, nY = Y.shape                            # for looping through Y

    Part1 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 1 values
    Part2 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 2 values

    parameter1 = parameters['lambda_1'] * SS                                        # parameter for weighted calculations
    parameter2 = parameters['lambda_4'] * np.matmul(W, (np.matmul(W.T, X) - Y))     # parameter for weighted calculations

    # Loops through shapelets and calculates Part 1 and Part 2
    for k in range(mDShape):
        length = Shape_t[k, 0].astype(int)      # shapelet length
        for p in range(length):
            Part1[k, p] = np.sum(np.multiply(parameter1[k, :], H_skp[k, :, p])) - parameter1[k, k] * H_skp[k, k, p]   # Part 1: |H|_F_2
            Part2[k, p] = np.sum(np.matmul(parameter2, Xkj_sk1[k, :, p]))            # Part 2: shapelet derivatives

    # Returns the combination of Part 1 and Part 2
    return Part1 + Part2