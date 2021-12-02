# v0

# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#%% DataClass definition
class News_dataset(Dataset):
    def __init__(self, data_df, args):
        # Remove empty entries
        if eval(args.remove_empty_entries) == True:
            print(f'Data shape before slicing = {data_df.shape}')
            slicer = [x != [0] * args.bow_len for x in data_df.bow]
            data_df = data_df[slicer]
            print(f'Data shape after slicing = {data_df.shape}')
            
        # Extract selected features
        self.bow = data_df['bow'].to_list()
        self.labels = data_df['label1'].to_list()

        # Generate tensors
        self.bow = torch.FloatTensor(self.bow)
        self.labels = torch.LongTensor(self.labels)
                                        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        X_bow = self.bow[idx]
        Y_labels = self.labels[idx]
        
        return X_bow, Y_labels

#%% Model definition
class News_model(nn.Module):
            
    def __init__(self, args):
        super(News_model, self).__init__()

        self.n_classes = args.num_classes
        self.bow_len = args.bow_len
        self.dropout = args.dropout
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.bow_len)
        self.bn2 = nn.BatchNorm1d(int(self.bow_len/2))
        
        # Fully connected 1
        self.fc_1 = nn.Linear(in_features = self.bow_len,
                              out_features = int(self.bow_len/2))

        # Fully connected 2
        self.fc_2 = nn.Linear(in_features = int(self.bow_len/2),
                              out_features = self.n_classes)

        # Relu
        self.relu = nn.ReLU()
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()       
        
        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
    def forward(self, X_bow):
        
        # Multi-label classifier      
        X = self.bn1(X_bow)                               # batch_size x bow_len
        X = self.fc_1(X)                                  # batch_size x (bow_len/2)
        X = self.relu(X)                                  # batch_size x (bow_len/2)
        X = self.drops(X)                                 # batch_size x (bow_len/2)
        X = self.bn2(X)                                   # batch_size x (bow_len/2)
        X = self.fc_2(X)                                  # batch_size x n_classes
        X = self.sigmoid(X)                               # batch_size x n_classes

        return X
