import numpy as np
from derivatives import S_derivation, lS_derivation
from utils import z_regularization

# Updates the unlabeled shapelets using the given data, W, SS and additional variables and parameters
def update_S(Z, Y, un_X, l_X, W, G, S, un_Xkj_skp, l_Xkj_skp, Hij_skp, H, parameters):
    DS_t = S[:, 1:]                 # extracts shapelets

    # Loops through update script IMax times
    for i in range(parameters['Imax']):
        DS_t = S[:, 1:] - parameters['eta'] *  S_derivation(Z, Y, un_X, l_X, W, G, S, un_Xkj_skp, l_Xkj_skp, Hij_skp, H, parameters)    # updates shapelets
    
    S = np.concatenate((S[:, 0:1], DS_t), axis=1) # reincorporates shapelet length column
    # Returns the updated unlabeled shapelets
    return S

# Updates the labeled shapelets using the given data, W, SS and additional variables and parameters
def update_lS(labeled_Y, labeled_X, W, labeled_Shape_t, labeled_Xkj_tp1_skl, lSSij_sil, SS, parameters):
    DS_t = labeled_Shape_t[:, 1:]                   # extracts shapelets

    # Loops through update script IMax times
    for i in range(parameters['Imax']):           
        DS_t = labeled_Shape_t[:, 1:] - parameters['eta'] * lS_derivation(labeled_Y, labeled_X, W, labeled_Shape_t, labeled_Xkj_tp1_skl, lSSij_sil, SS, parameters)    # updates shapelets
    
    labeled_Shape_t = np.concatenate((labeled_Shape_t[:, 0:1], DS_t), axis=1) # reincorporates shapelet length column
    # Returns the updated labeled shapelets
    return labeled_Shape_t

# Updates W using the given data and additional parameters
def update_W(labeled_X_t, unlabeled_X_t, Z_t, labeled_Y, parameters):
    mX, _ = unlabeled_X_t.shape                     # for generating I matrix
    p1 = parameters['lambda_2'] * np.eye(mX) + parameters['lambda_3'] * np.dot(unlabeled_X_t, unlabeled_X_t.T) + parameters['lambda_4'] * np.dot(labeled_X_t, labeled_X_t.T)    # weighted addition to XX^T
    p2 = parameters['lambda_3'] * np.dot(unlabeled_X_t, Z_t.T) + parameters['lambda_4'] * np.dot(labeled_X_t, labeled_Y.T)  # weighted XY^T
    W = np.matmul(np.linalg.inv(p1), p2)            # combination of both calculations

    # Returns the updated W
    return W

# Updates Z using the given data, updated W, L_G and additional parameters
def update_Z(W_t, unlabeled_X_t, L_Gt, parameters):
    mL, _ = L_Gt.shape                              # for generating I matrix
    p1 = parameters['lambda_3'] * np.dot(W_t.T, unlabeled_X_t) + parameters['zeta_2']   # weighted Z prediction
    p2 = L_Gt + parameters['lambda_3'] * np.eye(mL) + parameters['zeta_1'] * np.eye(mL) # weighted diagonal addition to L_G
    Z = np.matmul(p1, np.linalg.inv(p2))            # combination of both calculations
    for i in range(Z.shape[1]):
        Z[:, i] = z_regularization(Z[:, i])
    
    # Returns updated Y for unlabeled cases
    return Z