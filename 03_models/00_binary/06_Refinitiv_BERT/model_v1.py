# v0 -> Preprocesses dataset to correct mistake with empty headlines
# v1 -> Added layers to the binary classifier


# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

#%% DataClass definition
class News_dataset(Dataset):
    def __init__(self, data_df):
        # Define empty paragraph
        self.empty_token_ids = torch.zeros(256).type(torch.LongTensor)
        self.empty_token_ids[0] = 101
        self.empty_token_ids[1] = 102
        # Extract selected features
        self.token_ids = data_df['token_ids_1'].to_list()
        self.token_ids = [x[0] for x in self.token_ids]
        self.token_types = data_df['token_types_1'].to_list()
        self.token_types = [x[0] for x in self.token_types]
        self.att_masks = data_df['att_masks_1'].to_list()
        self.att_masks = [x[0] for x in self.att_masks]
        self.labels = data_df['label1'].to_list()
        # Remove empty news
        slicer = [torch.equal(x, self.empty_token_ids) for x in self.token_ids]
        self.token_ids = [i for idx, i in enumerate(self.token_ids) if not(slicer[idx])]
        self.token_types = [i for idx, i in enumerate(self.token_types) if not(slicer[idx])]
        self.att_masks = [i for idx, i in enumerate(self.att_masks) if not(slicer[idx])]
        self.labels = [i for idx, i in enumerate(self.labels) if not(slicer[idx])]
        # Generate tensors
        self.token_ids = torch.stack(self.token_ids)
        self.token_types = torch.stack(self.token_types)
        self.att_masks = torch.stack(self.att_masks)
        self.labels = torch.LongTensor(self.labels)
                                        
    def __len__(self):
        return len(self.token_ids)
        
    def __getitem__(self, idx):
        X_token_ids = self.token_ids[idx]
        X_token_types = self.token_types[idx]
        X_att_masks = self.att_masks[idx]
        Y_labels = self.labels[idx]
        
        return X_token_ids, X_token_types, X_att_masks, Y_labels

#%% Model definition
class News_model(nn.Module):
            
    def __init__(self, args):
        super(News_model, self).__init__()

        self.h_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.n_labels = args.num_labels
        self.seq_len = args.seq_len
        self.dropout = args.dropout
                     
        # Bert layer
        self.model_name = args.model_name
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        
        # Transformer layers
        #self.transf_enc_facts = nn.TransformerEncoderLayer(d_model = self.h_dim,
        #                                                   nhead = self.n_heads)
        
        # Fully connected #1
        self.fc_1 = nn.Linear(in_features = self.h_dim, out_features = 100)
        
        # Fully connected output
        self.fc_out = nn.Linear(in_features = 100, out_features = self.n_labels)

        # Relu
        self.relu = nn.ReLU()

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalizations
        self.bn1 = nn.BatchNorm1d(self.h_dim)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, X_token_ids, X_token_types, X_att_masks):
        # BERT encoder
        X_bert = {'input_ids': X_token_ids,
                  'token_type_ids': X_token_types,
                  'attention_mask': X_att_masks}
        out = self.bert_model(**X_bert,
                              output_hidden_states = True)       # Tuple
        out = out['pooler_output']                               # batch_size x h_dim
        
        # Binary classifier      
        out = self.bn1(out)                                      # batch_size x h_dim
        out = self.fc_1(out)                                     # batch_size x 100
        out = self.relu(out)                                     # batch_size x 100
        out = self.drops(out)                                    # batch_size x 100
        #out = self.bn2(out)                                      # batch_size x 100
        out = self.fc_out(out)                                   # batch_size x 1
        out = self.sigmoid(out)                                  # batch_size x 1

        return out
