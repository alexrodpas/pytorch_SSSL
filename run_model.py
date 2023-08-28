import pandas as pd
from model import SSSL
from dataset import load_dataset

def main():
    # Model and training parameters
    parameters = {}
    parameters['Lmin'] = 5                      # the minimum length of shapelets we plan to learn
    parameters['k'] = 1                         # the number of shapelets in equal length
    parameters['R'] = 3                         # the number of scales of shapelets length
    parameters['C'] = 2                         # the number of classes/clusters
    parameters['alpha'] = -1e2                  # parameter in Soft Minimum Function
    parameters['sigma'] = 1e1                   # parameter in RBF kernel
    parameters['lambda_1'] = -8                 # regularization parameter
    parameters['lambda_2'] = -8                 # regularization parameter
    parameters['lambda_3'] = -8                 # regularization parameter
    parameters['lambda_4'] = -8                 # regularization parameter
    parameters['Imax'] = 25                     # the number of internal iterations
    parameters['eta'] = 1e-1                    # learning rate
    parameters['epsilon'] = 1e-1                # internal convergence parameter
    parameters['w'] = 1e-2                      # weight initialization coefficient
    labeled_ratio = 0.2                         # percents of labled data
    batch_size = 64                             # batch size of loaded data
    num_epochs = 15                             # number of epochs to train the model

    # Loads data for training
    print("Loading training dataset...")        # logging
    data = pd.read_csv('ItalyPowerDemand_TRAIN')    # reads data
    labeled_data, labels, unlabeled_data = load_dataset(data, labeled_ratio, batch_size)    # loads data
    print("Training dataset loaded")            # logging
    print("---------------------------------")  # logging

    # Initializes Semi-Supervised Shapelets Learning model
    print("Initializing model...")              # logging
    SSSL_model = SSSL(labeled_data, unlabeled_data, labels, parameters) # initializes model
    print("Model initialized")                  # logging
    print("---------------------------------")  # logging

    # Trains Semi-Supervised Shapelets Learning model
    print("Training model...")                  # logging
    SSSL_model.train(num_epochs, logger=True)   # trains model
    print("Model trained")                      # logging
    print("---------------------------------")  # logging

    # Loads data for testing
    print("Loading testing dataset...")         # logging
    data = pd.read_csv('ItalyPowerDemand_TEST') # reads data
    labeled_data, labels, unlabeled_data = load_dataset(data, 1.0, batch_size)    # loads data
    print("Testing dataset loaded")             # logging
    print("---------------------------------")  # logging

    # Tests Semi-Supervised Shapelets Learning model
    print("Testing model...")                   # logging
    SSSL_model.test(num_epochs, logger=True)    # tests model
    print("Model tested")                       # logging
    print("---------------------------------")  # logging

    print("All done!")                          # script is complete

if __name__ == "__main__":
    main()