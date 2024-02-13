import torch
from torch import nn
from utils import spectral_timeseries_similarity, distance_timeseries_shapelet, shapelet_similarity, s_initialization, calculate_Z

# Semi-Supervised Shapelet Learning model
class SSSL(nn.Module):
    # Initializes the model
    def __init__(self, parameters):
        super(SSSL, self).__init__()
        
        self.params = parameters            # model training parameters
        self.lengths, S = s_initialization(self.params)     # shapelets
        self.S = nn.Parameter(torch.from_numpy(S))  # S
        self.W = nn.Linear(parameters['k'] * parameters['R'], parameters['C'], bias=False)
        nn.init.xavier_uniform_(self.W.weight)
    
    # Does a forward pass of the model on TS
    def forward(self, TS):
        # Forward pass on data
        X = distance_timeseries_shapelet(TS, self.lengths, self.S, self.params['alpha'])
        Z = self.W(X)                       # calculates label values
        
        # Returns the calculated X, SS and derivatives
        return X, Z
    
    # Trains the model
    def train(self, num_epochs, labeled_dataloader, unlabeled_dataloader, val_dataloader=None, logger=False):
        loss_fn = nn.NLLLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.params['eta'])
        
        labeled_batches = len(labeled_dataloader)
        unlabeled_batches = len(unlabeled_dataloader)
        max_batches = max(labeled_batches, unlabeled_batches)
        
        # Trains the model num-epochs times
        for epoch in range(num_epochs):
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
                    labeled_y = nn.functional.one_hot(y, num_classes=self.params['C'])
                    _, labeled_Z = self(labeled_TS)
                    loss += 0.5 * self.params['lambda_4'] * torch.sum(torch.pow(labeled_Z - labeled_y, 2))
                    
                # Unlabeled loss
                if (batch <= unlabeled_batches):
                    unlabeled_TS = next(unlabeled_iter)
                    unlabeled_X, unlabeled_pred = self(unlabeled_TS)
                    unlabeled_Z = torch.argmax(unlabeled_pred, dim=1)
                    unlabeled_y = nn.functional.one_hot(unlabeled_Z, num_classes=self.params['C']).to(torch.float)   # pseudo-labels
                    Z = calculate_Z(unlabeled_y)
                    loss += 0.5 * self.params['lambda_3'] * (batch - 1)/(max_batches - 1) * torch.sum(torch.pow(unlabeled_pred - Z.T, 2))
                    
                    # Timeseries similarity penalty
                    # print(f"Pred:\n{unlabeled_Z}\nZ:\n{Z}")
                    L_G = spectral_timeseries_similarity(unlabeled_X, self.params['sigma'])
                    penalty = 0.5 * torch.trace(torch.matmul(torch.matmul(Z, L_G), Z.T))
                    print(f"Classification boundary loss: {penalty}")
                    if not torch.isnan(penalty):
                        loss += penalty
                    
                # Regularization and similarity penalties
                SS = shapelet_similarity(self.lengths, self.S, self.params['alpha'], self.params['sigma'])
                loss += 0.5 * self.params['lambda_1'] * torch.linalg.norm(SS)**2
                loss += 0.5 * self.params['lambda_2'] * torch.linalg.norm(self.W.weight)**2
                
                # Normalization penalties
                norm_S = torch.zeros(self.S.shape)
                for i in range(len(self.lengths)):
                    min_S = self.S[i, :self.lengths[i]].min()
                    max_S = self.S[i, :self.lengths[i]].max()
                    norm_S[i, :self.lengths[i]] = (self.S[i, :self.lengths[i]] - min_S)/(max_S - min_S)
                
                loss += self.params['lambda_5'] * torch.linalg.norm(self.S - norm_S)**2
                
                min_W = self.W.weight.min()
                max_W = self.W.weight.max()
                norm_W = (self.W.weight - min_W)/max_W
                loss += self.params['lambda_6'] * torch.linalg.norm(self.W.weight - norm_W)**2
                    
                # Backward pass through the model, updates parameters
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if logger:
                print("---------------------------------")      # logging
                print(f"Epoch {epoch + 1}: Loss = {total_loss}")# logging
                
            if val_dataloader is not None:
                self.test(val_dataloader)

        if logger:
            print("---------------------------------")          # logging
            print(f"Shapelets:\n{self.S}")                    # logging
            print(f"Weights:\n{self.W.weight}")               # logging

    # Tests the model's output on TS against Y
    def test(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                total += len(y)
                pred = torch.argmax(self(X)[1], dim=1)
                correct += torch.sum(torch.where(pred == y, 1, 0))
                # print(pred)
        
        print(f"Model accuracy: {correct/total*100}%")          # logging
