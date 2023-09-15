import numpy as np
import torch
from torch import nn
from functions import trace_value
from utils import spectral_timeseries_similarity, distance_timeseries_shapelet, shapelet_similarity, EM, s_initialization, z_regularization, reshape_y_true
from updates import *

# Semi-Supervised Shapelet Learning model
class SSSL(nn.Module):
    # Initializes the model
    def __init__(self, labeled_TS, unlabeled_TS, labeled_Y, parameters):
        super(SSSL, self).__init__()
        
        self.params = parameters            # model training parameters
        self.labeled_TS = labeled_TS        # labeled time series
        self.unlabeled_TS = unlabeled_TS    # unlabeled time series
        self.labeled_Y = labeled_Y          # labels
        self.labeled_S = s_initialization(labeled_TS, self.params)      # labeled shapelets
        self.unlabeled_S = s_initialization(unlabeled_TS, self.params)  # unlabeled shapelets
        unlabeled_X, _ = distance_timeseries_shapelet(unlabeled_TS, self.unlabeled_S, self.params['alpha'])
        centroid, self.unlabeled_Y = EM(unlabeled_X, self.params['C'])  # pseudo labels
        self.W = self.params['w'] * np.vstack((-centroid[0,:], centroid[1:,:])) # W
    
    # Does a forward pass of the model on TS
    def forward(self, TS, labeled):
        # Forward pass on labeled data
        if labeled:
            X, Xkj_skl = distance_timeseries_shapelet(TS, self.labeled_S, self.params['alpha'])
            SS, _, SSij_sil = shapelet_similarity(self.labeled_S, self.params['alpha'], self.params['sigma'])
        # Forward pass on unlabeled data
        else:
            X, Xkj_skl = distance_timeseries_shapelet(TS, self.unlabeled_S, self.params['alpha'])
            SS, _, SSij_sil = shapelet_similarity(self.unlabeled_S, self.params['alpha'], self.params['sigma'])
        
        # Returns the calculated X, SS and derivatives
        return X, Xkj_skl, SS, SSij_sil
    
    # Trains the model
    def train(self, num_epochs, logger=False):
        # Trains the model num-epochs times
        for i in range(num_epochs):
            # Forward pass of the model, calculates trace value
            labeled_X, lXkj_skl, labeled_SS, lSSij_sil = self(self.labeled_TS, True)
            unlabeled_X, unXkj_skl, unlabeled_SS, unSSij_sil = self(self.unlabeled_TS, False)
            L_G, G = spectral_timeseries_similarity(unlabeled_X, self.params['sigma'])
            F = trace_value(labeled_X, unlabeled_X, self.unlabeled_Y, self.labeled_Y, L_G, unlabeled_SS, self.W, self.params)
            if np.isnan(F):             # failsafe
                break

            if logger:
                print("---------------------------------")  # logging
                print(f"Epoch {i + 1}: Trace value = {F}")  # logging

            # Backward pass of the model, updates parameters
            self.W = z_regularization(update_W(labeled_X, unlabeled_X, self.unlabeled_Y, self.labeled_Y, self.params))
            self.unlabeled_Y = update_Y(self.W, unlabeled_X, L_G, self.params)
            self.unlabeled_S = update_S(self.unlabeled_Y, unlabeled_X, self.W, G, self.unlabeled_S, unXkj_skl, unSSij_sil, unlabeled_SS, self.params)
            self.labeled_S = update_lS(self.labeled_Y, labeled_X, self.W, self.labeled_S, lXkj_skl, lSSij_sil, labeled_SS, self.params)
            self.unlabeled_S[:, 1:] = z_regularization(self.unlabeled_S[:, 1:]) # applies regularization
            self.labeled_S[:, 1:] = z_regularization(self.labeled_S[:, 1:])     # applies regularization
            
            # Stops training if the trace value is low
            if F < 100:
                break
        
        if logger:
            print("---------------------------------")      # logging

    # Tests the model's output on TS against Y
    def test(self, TS, Y):
        X, _, _, _ = self(TS, False)    # forward pass
        Z = np.dot(self.W.T, X).T       # calculates label values
        mZ, _ = Z.shape                 # for looping through Z

        # Calculates the model accuracy
        correct = 0                     # number of correct predictions
        for i in range(mZ):
            if Y[i, np.argmax(Z, axis=1)[i]] == 1: # matches predictions to ground truth
                correct += 1            # prediction was correct
        
        print(f"Model accuracy: {correct/mZ*100}%")     # logging
