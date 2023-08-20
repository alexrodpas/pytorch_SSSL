import numpy as np

# Calculates the S derivation of the given data, W, shapelets, ST, SS and additional variables and parameters
def S_derivation(Y, X, W, ST_t, Shape_t, Xkj_skl, SSij_sil, SS, Parameter):
    DShape_t = Shape_t[:, 1:]                   # extracts shapelets
    mST, nST = ST_t.shape                       # for looping through ST
    mDShape, nDShape = DShape_t.shape           # for looping through shapelets
    mY, nY = Y.shape                            # for looping through Y

    Part1 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 1 values
    Part2 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 1 values
    Part3 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 1 values

    parameter1 = 50000000 * np.dot(Y.T, Y)                      # parameter for weighted calculations
    parameter2 = Parameter['lambda_1'] * SS                     # parameter for weighted calculations
    parameter3 = Parameter['lambda_2'] * (np.dot(W.T, X) - Y)   # parameter for weighted calculations

    STij_skl = np.zeros((mST, nST, mDShape, nDShape))   # for storing ST_(ij) derivatives with respect to S_(kl)

    # Loops through shapelets and calculates each part
    for k in range(mDShape):
        len = Shape_t[k, 0]                     # unlabeled shapelet length
        for l in range(len):
            p1 = np.zeros((mST, nST))           # for storing YLY^T elements
            p3 = np.zeros((mY, nY))             # for storing unlabeled shapelet derivatives
            for i in range(mST):
                for j in range(nST):
                    # Calculates ST_(ij) derivative with respect to S_(kl)
                    STij_skl[i, j, k, l] = ST_t[i, j] * (-2/Parameter['sigma']**2) * (X[k, i] - X[k, j]) * (Xkj_skl[k, i, l] - Xkj_skl[k, j, l])
            for i in range(mST):
                for j in range(nST):
                    if j == i:
                        p1[i, j] = parameter1[i, j] * np.sum(STij_skl[i, :, k, l])      # summation for diagonal elements
                    else:
                        p1[i, j] = parameter1[i, j] * STij_skl[i, j, k, l]              # regular calculation for non-diagonal elements
            for i in range(mY):
                for j in range(nY):
                    p3[i, j] = parameter3[i, j] * W[k, i] * Xkj_skl[k, j, l]    # shapelet derivative
            Part1[k, l] = np.sum(p1)            # Part 1: YLY^T
            Part2[k, l] = 2 * np.sum(parameter2[k, :] * SSij_sil[k, :, l]) - parameter2[k, k] * SSij_sil[k, k, l]   # Part 2: weighted normalized shapelet derivatives
            Part3[k, l] = np.sum(p3)            # Part 3: shapelet derivatives

    # Returns the combination of all parts
    return Part1 + Part2 + Part3

# Calculates the SS derivation of the given data, W, shapelets at time t, SS, ST and additional variables and parameters
def SS_derivation(unlabeled_Y, labeled_Y, unlabeled_X, labeled_X, W, ST_t, unlabeled_Shape_t, labeled_Shape_t, unlabeled_Xkj_skl, labeled_Xkj_sk1, unSSij_sil, SS, Parameter):
    DShape_t = unlabeled_Shape_t[:, 1:]         # extracts unlabeled shapelets
    mST, nST = ST_t.shape                       # for looping through ST
    mDShape, nDShape = DShape_t.shape           # for looping through shapelets
    mY, nY = unlabeled_Y.shape                  # for looping through unlabeled Y
    mYT, nYT = labeled_Y.T.shape                # for looping through labeled Y

    Part1 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 1 values
    Part2 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 2 values
    Part3 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 3 values
    Part4 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 4 values

    parameter1 = 1/2 * np.dot(unlabeled_Y.T, unlabeled_Y)                           # parameter for weighted calculations
    parameter2 = Parameter['lambda_1'] * SS                                         # parameter for weighted calculations
    parameter3 = Parameter['lambda_2'] * (np.dot(W.T, unlabeled_X) - unlabeled_Y)   # parameter for weighted calculations
    parameter4 = Parameter['lambda_4'] * (np.dot(W.T, labeled_X) - labeled_Y.T)     # parameter for weighted calculations

    STij_skl = np.zeros((mST, nST, mDShape, nDShape))   # for storing ST_(ij) derivatives with respect to S_(kl)

    # Loops through shapelets and calculates each part
    for k in range(mDShape):
        len = unlabeled_Shape_t[k, 0]           # unlabeled shapelet length
        for l in range(len):
            p1 = np.zeros((mST, nST))           # for storing YLY^T elements
            p3 = np.zeros((mY, nY))             # for storing unlabeled shapelet derivatives
            for i in range(mST):
                for j in range(nST):
                    # Calculates ST_(ij) derivative with respect to S_(kl)
                    STij_skl[i, j, k, l] = ST_t[i, j] * (-2/Parameter['sigma']**2) * (unlabeled_X[k, i] - unlabeled_X[k, j]) * (unlabeled_Xkj_skl[k, i, l] - unlabeled_Xkj_skl[k, j, l])
            for i in range(mST):
                for j in range(nST):
                    if j == i:
                        p1[i, j] = parameter1[i, j] * np.sum(STij_skl[i, :, k, l])      # summation for diagonal elements
                    else:
                        p1[i, j] = parameter1[i, j] * STij_skl[i, j, k, l]              # regular calculation for non-diagonal elements
            for i in range(mY):
                for j in range(nY):
                    p3[i, j] = parameter3[i, j] * W[k, i] * unlabeled_Xkj_skl[k, j, l]  # unlabeled shapelet derivative
            Part1[k, l] = np.sum(p1)            # Part 1: YLY^T
            Part2[k, l] = 2 * np.sum(parameter2[k, :] * unSSij_sil[k, :, l]) - parameter2[k, k] * unSSij_sil[k, k, l]   # Part 2: weighted normalized unlabeled shapelet derivatives
            Part3[k, l] = np.sum(p3)            # Part 3: unlabeled shapelet derivatives
        len = labeled_Shape_t[k, 0]             # labeled shapelet length
        for l in range(len):
            p4 = np.zeros((mYT, nYT))           # for storing labeled shapelet derivatives
            for i in range(mYT):
                for j in range(nYT):
                    p4[i, j] = parameter4[i, j] * W[k, i] * labeled_Xkj_sk1[k, j, l]    # labeled shapelet derivative
            Part4[k, l] = np.sum(p4)            # Part 4: labeled shapelet derivatives

    # Returns the combination of all 4 parts
    return Part1 + Part2 + Part3 + Part4

# Calculates the lS derivation with the given X and Y, W, shapelets at time t, and additional variables and parameters
def lS_derivation(Y, X, W, Shape_t, Xkj_sk1, SSij_sil, SS, Parameter):
    DShape_t = Shape_t[:, 1:]                   # extracts shapelets
    mDShape, nDShape = DShape_t.shape           # for looping through shapelets
    mY, nY = Y.shape                            # for looping through Y

    Part1 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 1 values
    Part2 = np.zeros((mDShape, nDShape))        # initializing matrix to record part 2 values

    parameter1 = Parameter['lambda_1'] * SS                         # parameter for weighted calculations
    parameter2 = Parameter['lambda_4'] * (np.dot(W.T, X) - Y.T)     # parameter for weighted calculations

    # Loops through shapelets and calculates Part 1 and Part 2
    for k in range(mDShape):
        len = Shape_t[k, 0]                     # shapelet length
        for l in range(len):
            p2 = np.zeros((mY, nY))             # for storing shapelet derivatives
            for i in range(mY):
                for j in range(nY):
                    p2[i, j] = parameter2[i, j] * W[k, i] * Xkj_sk1[k, j, l]   # shapelet derivative
            Part1[k, l] = 2 * np.sum(parameter1[k, :] * SSij_sil[k, :, l]) - parameter1[k, k] * SSij_sil[k, k, l]   # Part 1: |H|_F_2
            Part2[k, l] = np.sum(p2)            # Part 2: shapelet derivatives

    # Returns the combination of Part 1 and Part 2
    return Part1 + Part2