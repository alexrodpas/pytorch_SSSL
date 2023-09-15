import numpy as np
from derivatives import S_derivation, lS_derivation

# Updates the unlabeled shapelets using the given data, W, SS and additional variables and parameters
def update_S(unlabeled_Y, unlabeled_X, W, ST_t, unlabeled_Shape_t, unlabeled_Xkj_skl, unSSij_sil, SS, parameters):
    DS_t = unlabeled_Shape_t[:, 1:]                 # extracts shapelets

    # Loops through update script IMax times
    for i in range(parameters['Imax']):
        DS_t = DS_t - parameters['eta'] * S_derivation(unlabeled_Y, unlabeled_X, W, ST_t, unlabeled_Shape_t, unlabeled_Xkj_skl, unSSij_sil, SS, parameters) # updates shapelets
    
    # Returns the updated unlabeled shapelets
    return np.concatenate((unlabeled_Shape_t[:, 0:1], DS_t), axis=1) # reincorporates shapelet length column

# Updates the labeled shapelets using the given data, W, SS and additional variables and parameters
def update_lS(labeled_Y, labeled_X, W, labeled_Shape_t, labeled_Xkj_tp1_skl, lSSij_sil, SS, parameters):
    DS_t = labeled_Shape_t[:, 1:]                   # extracts shapelets

    # Loops through update script IMax times
    for i in range(parameters['Imax']):               
        DS_t = DS_t - parameters['eta'] * lS_derivation(labeled_Y, labeled_X, W, labeled_Shape_t, labeled_Xkj_tp1_skl, lSSij_sil, SS, parameters)   # updates shapelets

    # Returns the updated labeled shapelets
    return np.concatenate((labeled_Shape_t[:, 0:1], DS_t), axis=1) # reincorporates shapelet length column

# Updates W using the given data and additional parameters
def update_W(labeled_X_t, unlabeled_X_t, unlabeled_Y_t, labeled_Y, parameters):
    mX, _ = unlabeled_X_t.shape                     # for generating I matrix
    p1 = parameters['lambda_2'] * np.dot(unlabeled_X_t, unlabeled_X_t.T) + parameters['lambda_4'] * np.dot(labeled_X_t, labeled_X_t.T) + parameters['lambda_3'] * np.eye(mX)    # weighted addition to XX^T
    p2 = parameters['lambda_2'] * np.dot(unlabeled_X_t, unlabeled_Y_t.T) + parameters['lambda_4'] * np.dot(labeled_X_t, labeled_Y)  # weighted XY^T
    W_tp1 = np.dot(np.linalg.inv(p1), p2)           # combination of both calculations

    # Returns the updated W
    return W_tp1

# Updates Y using the given data, updated W, L_G and additional parameters
def update_Y(W_tp1, unlabeled_X_t, L_Gt, parameters):
    p1 = parameters['lambda_2'] * np.dot(W_tp1.T, unlabeled_X_t)    # weighted Y prediction
    mL, _ = L_Gt.shape                              # for generating I matrix
    p2 = L_Gt + parameters['lambda_2'] * np.eye(mL) # weighted diagonal addition to L_G
    Y_tp1 = np.dot(p1, np.linalg.inv(p2))           # combination of both calculations
    
    # Returns updated Y for unlabeled cases
    return Y_tp1