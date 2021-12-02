# v0

# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#%% DataClass definition
class News_dataset(Dataset):
    def __init__(self, data_df, args):
        # Remove empty entries
        if args.remove_empty_entries == True:
            slicer = [x != ['','','',''] for x in data_df.headline]
            data_df = data_df[slicer]
            
        # Extract selected features
        self.news_id_1 = [x[0] for x in data_df['ids']] 
        self.news_id_2 = [x[1] for x in data_df['ids']] 
        self.news_id_3 = [x[2] for x in data_df['ids']] 
        self.news_id_4 = [x[3] for x in data_df['ids']] 
        self.labels = data_df['label1'].to_list()

        # Generate tensors
        self.news_id_1 = torch.LongTensor(self.news_id_1)
        self.news_id_2 = torch.LongTensor(self.news_id_2)
        self.news_id_3 = torch.LongTensor(self.news_id_3)
        self.news_id_4 = torch.LongTensor(self.news_id_4)
        self.labels = torch.LongTensor(self.labels)
                                        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        X_news_id_1 = self.news_id_1[idx]
        X_news_id_2 = self.news_id_2[idx]
        X_news_id_3 = self.news_id_3[idx]
        X_news_id_4 = self.news_id_4[idx]
        Y_labels = self.labels[idx]
        
        return X_news_id_1, X_news_id_2, X_news_id_3, X_news_id_4, Y_labels

#%% Model definition
class News_model(nn.Module):
            
    def __init__(self, args, id_2_vec):
        super(News_model, self).__init__()

        self.h_dim = args.hidden_dim
        self.emb_dim = args.emb_dim
        self.n_classes = args.num_classes
        self.seq_len = args.seq_len
        self.dropout = args.dropout
        self.freeze_emb = args.freeze_emb
        print('Converting id_2_vec to tensor')
        self.emb_weights = torch.FloatTensor(list(id_2_vec.values()))
        print('Done')             
        
        # Embedding layer
        self.embeddings = nn.Embedding.from_pretrained(self.emb_weights,
                                                       freeze = self.freeze_emb,
                                                       padding_idx = args.pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size = 300,
                            hidden_size = self.h_dim,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True)
        
        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim * 2 * 4,
                                out_features = self.n_classes)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.h_dim * 2 * 4)

    def forward(self, X_news_id_1, X_news_id_2, X_news_id_3, X_news_id_4):
        
        # Embedding step
        X_1 = self.embeddings(X_news_id_1)                  # batch_size x emb_dim
        X_2 = self.embeddings(X_news_id_2)                  # batch_size x emb_dim
        X_3 = self.embeddings(X_news_id_3)                  # batch_size x emb_dim
        X_4 = self.embeddings(X_news_id_4)                  # batch_size x emb_dim
        
        # LSTM step
        X_1 = self.lstm(X_1)[0]                             # batch_size x seq_len x (2 x hidden_dim)
        X_1_fwd = X_1[:, -1, :self.h_dim]                   # batch_size x h_dim
        X_1_bwd = X_1[:, 0, self.h_dim:]                    # batch_size x h_dim
        X_1 = torch.cat([X_1_fwd, X_1_bwd], dim = 1)        # batch_size x (2 x h_dim)
        X_1 = self.drops(X_1)                               # batch_size x (2 x h_dim)

        X_2 = self.lstm(X_2)[0]                             # batch_size x seq_len x (2 x hidden_dim)
        X_2_fwd = X_2[:, -1, :self.h_dim]                   # batch_size x h_dim
        X_2_bwd = X_2[:, 0, self.h_dim:]                    # batch_size x h_dim
        X_2 = torch.cat([X_2_fwd, X_2_bwd], dim = 1)        # batch_size x (2 x h_dim)
        X_2 = self.drops(X_2)                               # batch_size x (2 x h_dim)

        X_3 = self.lstm(X_3)[0]                             # batch_size x seq_len x (2 x hidden_dim)
        X_3_fwd = X_3[:, -1, :self.h_dim]                   # batch_size x h_dim
        X_3_bwd = X_3[:, 0, self.h_dim:]                    # batch_size x h_dim
        X_3 = torch.cat([X_3_fwd, X_3_bwd], dim = 1)        # batch_size x (2 x h_dim)
        X_3 = self.drops(X_3)                               # batch_size x (2 x h_dim)

        X_4 = self.lstm(X_4)[0]                             # batch_size x seq_len x (2 x hidden_dim)
        X_4_fwd = X_4[:, -1, :self.h_dim]                   # batch_size x h_dim
        X_4_bwd = X_4[:, 0, self.h_dim:]                    # batch_size x h_dim
        X_4 = torch.cat([X_4_fwd, X_4_bwd], dim = 1)        # batch_size x (2 x h_dim)
        X_4 = self.drops(X_4)                               # batch_size x (2 x h_dim)
       
        # Aggregator
        X = torch.cat([X_1, X_2, X_3, X_4], dim = 1)        # batch_size x (2 x 4 x h_dim)
        
        # Multi-label classifier      
        X = self.bn1(X)                                     # batch_size x (2 x 4 x h_dim)
        X = self.fc_out(X)                                  # batch_size x n_classes
        out = self.sigmoid(X)                               # batch_size x n_classes

        return out
