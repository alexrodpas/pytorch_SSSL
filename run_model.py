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
    parameters['sigma'] = 1e0                   # parameter in RBF kernel
    parameters['lambda_1'] = 1e0                # regularization parameter for shapelet similarity
    parameters['lambda_2'] = 1e0                # regularization parameter for classification boundary
    parameters['lambda_3'] = 1e0                # regularization parameter for least square minimization with respect to unlabeled time series
    parameters['lambda_4'] = 1e0                # regularization parameter for least square minimization with respect to labeled time series
    parameters['Imax'] = 50                     # the number of internal iterations
    parameters['eta'] = 1e-2                    # learning rate
    parameters['epsilon'] = 1e-1                # internal convergence parameter
    parameters['w'] = 1e-2                      # weight initialization coefficient
    parameters['zeta_1'] = 1e1                  # additional orthogonality constant
    parameters['zeta_2'] = 1e1                  # additional orthogonality constant
    labeled_ratio = 0.9                         # percents of labeled data
    batch_size = 64                             # batch size of loaded data
    num_epochs = 15                             # number of epochs to train the model
    
    train_data_file = 'datasets/ItalyPowerDemand_TRAIN.csv'
    test_data_file = 'datasets/ItalyPowerDemand_TEST.csv'

    # Loads data for training
    print("Loading training dataset...")        # logging
    labeled_data, labels, unlabeled_data = load_dataset(train_data_file, labeled_ratio, batch_size)    # loads data
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
    labeled_data, labels, unlabeled_data = load_dataset(test_data_file, 1.0, batch_size)    # loads data
    print("Testing dataset loaded")             # logging
    print("---------------------------------")  # logging

    # Tests Semi-Supervised Shapelets Learning model
    print("Testing model...")                   # logging
    SSSL_model.test(labeled_data, labels)       # tests model
    print("Model tested")                       # logging
    print("---------------------------------")  # logging

    print("All done!")                          # script is complete

if __name__ == "__main__":
    main()