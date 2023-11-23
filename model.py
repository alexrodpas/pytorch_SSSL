import numpy as np
from torch import nn
from functions import objective
from utils import spectral_timeseries_similarity, distance_timeseries_shapelet, shapelet_similarity, EM, s_initialization, z_regularization
from updates import update_S, update_W, update_Z

# Semi-Supervised Shapelet Learning model
class SSSL(nn.Module):
    # Initializes the model
    def __init__(self, labeled_TS, unlabeled_TS, labeled_Y, parameters):
        super(SSSL, self).__init__()
        
        self.params = parameters            # model training parameters
        self.labeled_TS = labeled_TS        # labeled time series
        self.unlabeled_TS = unlabeled_TS    # unlabeled time series
        self.labeled_Y = labeled_Y.T        # labels
        self.S = s_initialization(unlabeled_TS, self.params)  # shapelets
        unlabeled_X, _ = distance_timeseries_shapelet(unlabeled_TS, self.S, self.params['alpha'])
        centroid, self.Z = EM(unlabeled_X, self.params['C'])   # pseudo labels
        self.W = self.params['w'] * np.vstack((-centroid[0,:], centroid[1:,:])) # W
    
    # Does a forward pass of the model on TS
    def forward(self, TS):
        # Forward pass on data
        X, Xkj_skl = distance_timeseries_shapelet(TS, self.S, self.params['alpha'])
        SS, _, SSij_sil = shapelet_similarity(self.S, self.params['alpha'], self.params['sigma'])
        
        # Returns the calculated X, SS and derivatives
        return X, Xkj_skl, SS, SSij_sil
    
    # Trains the model
    def train(self, num_epochs, logger=False):
        # Trains the model num-epochs times
        for i in range(num_epochs):
            
            print(f"S: \n{self.S}")
            print(f"W: \n{self.W}")
            print(f"Z: \n{self.Z}")
            
            # Forward pass of the model, calculates trace value
            labeled_X, lXij_skp, l_H, l_Hij_skp = self(self.labeled_TS)
            unlabeled_X, unXij_skp, un_H, un_Hij_skp = self(self.unlabeled_TS)
            L_G, G = spectral_timeseries_similarity(unlabeled_X, self.params['sigma'])
            F = objective(labeled_X, unlabeled_X, self.Z, self.labeled_Y, L_G, un_H, self.W, self.params)
            if np.isnan(F):             # failsafe
                break

            if logger:
                print("---------------------------------")      # logging
                print(f"Epoch {i + 1}: Objective value = {F}")  # logging

            # Backward pass of the model, updates parameters
            self.S = update_S(self.Z, self.labeled_Y, unlabeled_X, labeled_X, self.W, G, self.S, unXij_skp, lXij_skp, un_Hij_skp, un_H, self.params)
            self.S[:, 1:] = z_regularization(self.S[:, 1:]) # applies regularization
            self.W = z_regularization(update_W(labeled_X, unlabeled_X, self.Z, self.labeled_Y, self.params))
            self.Z = update_Z(self.W, unlabeled_X, L_G, self.params)
            
            # Stops training if the objective value is low
            if F < 100:
                break
        
        if logger:
            print("---------------------------------")      # logging

    # Tests the model's output on TS against Y
    def test(self, TS, Y):
        TS[:, 1:] = z_regularization(TS[:, 1:])
        X, _, _, _ = self(TS)           # forward pass
        Z = np.matmul(self.W.T, X).T    # calculates label values
        mZ, _ = Z.shape                 # for looping through Z

        # Calculates the model accuracy
        correct = 0                     # number of correct predictions
        for i in range(mZ):
            if Y[i, np.argmax(Z, axis=1)[i]] == 1: # matches predictions to ground truth
                correct += 1            # prediction was correct
        
        print(f"Model accuracy: {correct/mZ*100}%")     # logging
