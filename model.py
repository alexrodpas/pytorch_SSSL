import numpy as np
import torch
from torch import nn
from utils import spectral_timeseries_similarity, distance_timeseries_shapelet, shapelet_similarity, s_initialization, z_regularization

# Semi-Supervised Shapelet Learning model
class SSSL(nn.Module):
    # Initializes the model
    def __init__(self, parameters):
        super(SSSL, self).__init__()
        
        self.params = parameters            # model training parameters
        self.lengths, S = s_initialization(self.params)     # shapelets
        self.S = nn.Parameter(torch.from_numpy(S))  # S
        self.W = nn.Linear(parameters['R'], parameters['C'], bias=False)
        nn.init.xavier_uniform_(self.W.weight)
    
    # Does a forward pass of the model on TS
    def forward(self, TS):
        # Forward pass on data
        X = distance_timeseries_shapelet(TS, self.lengths, self.S, self.params['alpha'])
        Z = self.W(X)               # calculates label values
        
        # Returns the calculated X, SS and derivatives
        return X, Z
    
    # Trains the model
    def train(self, num_epochs, labeled_dataloader, unlabeled_dataloader, logger=False):
        loss_fn = nn.NLLLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.params['eta'])
        
        labeled_batches = len(labeled_dataloader)
        unlabeled_batches = len(unlabeled_dataloader)
        max_batches = max(labeled_batches, unlabeled_batches)
        
        # Trains the model num-epochs times
        for i in range(num_epochs):
            batch = 0
            labeled_iter = iter(labeled_dataloader)
            unlabeled_iter = iter(unlabeled_dataloader)
            total_loss = 0
            
            # Loops through batches of labeled and unlabeled data
            while (batch < max_batches):
                batch += 1
                loss = 0
                
                # Labeled loss
                if (batch <= labeled_batches):
                    labeled_TS, y = next(labeled_iter)
                    _, labeled_Z = self(labeled_TS)
                    loss += loss_fn(labeled_Z, y)
                    
                # Unlabeled loss
                if (batch <= unlabeled_batches):
                    unlabeled_TS = next(unlabeled_iter)
                    unlabeled_X, unlabeled_Z = self(unlabeled_TS)
                    unlabeled_y = torch.argmax(unlabeled_Z, dim=1)  # pseudo-labels
                    loss += (batch - 1)/(max_batches - 1) * loss_fn(unlabeled_Z, unlabeled_y)
                    
                    # Timeseries similarity penalty
                    unlabeled_Z = nn.functional.one_hot(unlabeled_y).to(torch.float)
                    L_G = spectral_timeseries_similarity(unlabeled_X, self.params['sigma'])
                    loss += torch.trace(torch.matmul(torch.matmul(unlabeled_Z.T, L_G), unlabeled_Z))
                    
                # Regularization and similarity penalties
                SS = shapelet_similarity(self.lengths, self.S, self.params['alpha'], self.params['sigma'])
                loss += self.params['lambda_1'] * torch.linalg.norm(SS)**2
                loss += self.params['lambda_2'] * torch.linalg.norm(self.W.weight)**2
                    
                # Backward pass through the model, updates parameters
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if logger:
                print("---------------------------------")      # logging
                print(f"Epoch {i + 1}: Loss = {total_loss}")    # logging

        if logger:
            print("---------------------------------")          # logging

    # Tests the model's output on TS against Y
    def test(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                total += len(y)
                pred = torch.argmax(self(X)[1], dim=1)
                correct += torch.sum(torch.where(pred == y, 1, 0))
        
        print(f"Model accuracy: {correct/total*100}%")          # logging
