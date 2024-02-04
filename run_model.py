from model import SSSL
from dataset import load_dataset

def main():
    # Model and training parameters
    parameters = {
        'Lmin': 5,                              # the minimum length of shapelets we plan to learn
        'k': 2,                                 # the number of shapelets in equal length
        'R': 3,                                 # the number of scales of shapelets length
        'C': 2,                                 # the number of classes/clusters
        'alpha': -1e2,                          # parameter in Soft Minimum Function
        'sigma': 1e0,                           # parameter in RBF kernel
        'lambda_1': 5e-1,                       # regularization parameter for shapelet similarity
        'lambda_2': 5e-1,                       # regularization parameter for classification boundary
        'lambda_3': 1e0,                        # regularization parameter for least square minimization with respect to unlabeled time series
        'lambda_4': 1e0,                        # regularization parameter for least square minimization with respect to labeled time series
        'Imax': 50,                             # the number of internal iterations
        'eta': 1e-2,                            # learning rate
        'epsilon': 1e-1,                        # internal convergence parameter
        'w': 1e-2,                              # weight initialization coefficient
        'zeta_1': 1e1,                          # additional orthogonality constant
        'zeta_2': 1e1}                          # additional orthogonality constant
    labeled_ratio = 0.1                         # percents of labeled data
    batch_size = 16                             # batch size of loaded data
    num_epochs = 20                             # number of epochs to train the model
    
    train_data_file = 'datasets/ItalyPowerDemand_TRAIN.csv'
    test_data_file = 'datasets/ItalyPowerDemand_TEST.csv'

    # Loads data for training
    print("Loading training dataset...")        # logging
    labeled_dataloader, unlabeled_dataloader = load_dataset(train_data_file, labeled_ratio, batch_size)    # loads data
    print("Training dataset loaded")            # logging
    print("---------------------------------")  # logging

    # Initializes Semi-Supervised Shapelets Learning model
    print("Initializing model...")              # logging
    SSSL_model = SSSL(parameters)               # initializes model
    print("Model initialized")                  # logging
    print("---------------------------------")  # logging

    # Trains Semi-Supervised Shapelets Learning model
    print("Training model...")                  # logging
    SSSL_model.train(num_epochs, labeled_dataloader, unlabeled_dataloader, logger=True)   # trains model
    print("Model trained")                      # logging
    print("---------------------------------")  # logging

    # Loads data for testing
    print("Loading testing dataset...")         # logging
    test_dataloader, _ = load_dataset(test_data_file, 1.0, batch_size)    # loads data
    print("Testing dataset loaded")             # logging
    print("---------------------------------")  # logging

    # Tests Semi-Supervised Shapelets Learning model
    print("Testing model...")                   # logging
    SSSL_model.test(test_dataloader)            # tests model
    print("Model tested")                       # logging
    print("---------------------------------")  # logging

    print("All done!")                          # script is complete

if __name__ == "__main__":
    main()