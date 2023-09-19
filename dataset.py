import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils import reshape_y_true, z_regularization

# Helper class for creating labeled time series dataset
class LabeledTimeseriesDataset(Dataset):
    # Initializes the dataset for labeled time series
    def __init__(self, data, transform=None, target_transform=None):
        self.timeseries = data.iloc[:, 1:]              # time series data
        self.labels = data.iloc[:, 0]                   # labels
        self.transform = transform                      # time series transform
        self.target_transform = target_transform        # label transform

    # Returns the length of the dataset
    def __len__(self):
        return len(self.labels)

    # Returns a time series-label pair with applied transforms
    def __getitem__(self, index):
        timeseries = self.timeseries.iloc[index]        # extracts time series
        label = self.labels.iloc[index]                 # extracts label

        if self.transform:
            timeseries = self.transform(timeseries)     # applies transform to time series

        if self.target_transform:
            label = self.target_transform(label)        # applies transform to label
        
        # Returns transformed time series and label
        return timeseries, label

# Helper class for creating unlabeled time series dataset
class UnlabeledTimeseriesDataset(Dataset):
    # Initializes the dataset for unlabeled time series
    def __init__(self, data, transform=None):
        self.timeseries = data                          # time series data
        self.transform = transform                      # time series transform

    # Returns the length of the dataset
    def __len__(self):
        return len(self.timeseries)

    # Returns a time series with applied transforms
    def __getitem__(self, index):
        timeseries = self.timeseries.iloc[index]        # extracts time series

        if self.transform:
            timeseries = self.transform(timeseries)     # applies transform to time series
        
        # Returns transformed time series
        return timeseries

# Loads the data from the given file path and creates labeled and unlabeled
# datasets based on the given label ratio and loaded in the given batch size
def load_dataset(file_path, label_ratio, batch_size):
    label_ratio = max(label_ratio, 0.0)                 # failsafe
    
    # Loads data from csv file
    data = pd.read_csv(file_path, header=None, index_col=None)

    # Separates labeled and unlabeled data based on the given label ratio
    if label_ratio < 1.0:
        labeled_index = data.sample(frac=label_ratio, replace=False).index  # extracts labeled indices
        labeled_data = data.iloc[labeled_index].drop(columns=0).to_numpy()       # extracts labeled data
        labels = reshape_y_true(labeled_data[:, 0].astype(int), np.max(labeled_data[:, 0].astype(int)) + 1)   # one-hot encoding
        m, n = labeled_data.shape
        labeled_data = np.hstack([(n - 1) * np.ones((m, 1)), z_regularization(labeled_data[:, 1:])])  # adds length column

        unlabeled_data = data.drop(index=labeled_index).to_numpy()  # removes labeled data
        m, n = unlabeled_data.shape
        unlabeled_data = np.hstack([(n - 1) * np.ones((m, 1)), z_regularization(unlabeled_data[:, 1:])])  # adds length column
    else:
        labeled_data = data.drop(columns=0).to_numpy()              # extracts labeled data
        labels = reshape_y_true(labeled_data[:, 0].astype(int), np.max(labeled_data[:, 0].astype(int)) + 1)   # one-hot encoding
        m, n = labeled_data.shape
        labeled_data = np.hstack([(n - 1) * np.ones((m, 1)), z_regularization(labeled_data[:, 1:])])  # adds length column
        unlabeled_data = None

    # Creates datasets based on the labeled and unlabeled data
    # labeled_dataset = LabeledTimeseriesDataset(labeled_data)
    # unlabeled_dataset = UnlabeledTimeseriesDataset(data)

    # Creates data loaders for the labeled and unlabeled datasets
    # labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    # unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    # Returns the labeled and unlabeled data
    return labeled_data, labels, unlabeled_data#, labeled_dataloader, unlabeled_dataloader