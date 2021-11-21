# Model with transformers layer on top of 4 news per day

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
        self.token_ids = data_df['token_ids'].to_list()
        self.token_ids = [torch.cat(x) for x in self.token_ids]
        self.token_types = data_df['token_types'].to_list()
        self.token_types = [torch.cat(x) for x in self.token_types]
        self.att_masks = data_df['att_masks'].to_list()
        self.att_masks = [torch.cat(x) for x in self.att_masks]
        self.labels = data_df['label1'].to_list()
        
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
        
        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim * 4,
                                out_features = self.n_labels)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.h_dim)

    def forward(self, X_token_ids, X_token_types, X_att_masks):
        # BERT encoder #1
        X_bert = {'input_ids': X_token_ids[0:256],
                  'token_type_ids': X_token_types[0:256],
                  'attention_mask': X_att_masks[0:256]}
        out = self.bert_model(**X_bert,
                               output_hidden_states = True) # Tuple
        out_bert_1 = out['pooler_output']                   # batch_size x h_dim
        
        # BERT encoder #2
        X_bert = {'input_ids': X_token_ids[256:512],
                  'token_type_ids': X_token_types[256:512],
                  'attention_mask': X_att_masks[256:512]}
        out = self.bert_model(**X_bert,
                               output_hidden_states = True) # Tuple
        out_bert_2 = out['pooler_output']                   # batch_size x h_dim
        
        # BERT encoder #3
        X_bert = {'input_ids': X_token_ids[512:768],
                  'token_type_ids': X_token_types[512:768],
                  'attention_mask': X_att_masks[512:768]}
        out = self.bert_model(**X_bert,
                               output_hidden_states = True) # Tuple
        out_bert_3 = out['pooler_output']                   # batch_size x h_dim
        
        # BERT encoder #4
        X_bert = {'input_ids': X_token_ids[768:1024],
                  'token_type_ids': X_token_types[768:1024],
                  'attention_mask': X_att_masks[768:1024]}
        out = self.bert_model(**X_bert,
                               output_hidden_states = True) # Tuple
        out_bert_4 = out['pooler_output']                   # batch_size x h_dim
        
        # Multi-label classifier      
        out = torch.cat([out_bert_1, out_bert_2,
                         out_bert_3, out_bert_4, dim = 1])  # batch_size x (h_dim x 4)
        #out = self.bn1(out)                                 # batch_size x h_dim
        out = self.fc_out(out)                              # batch_size x n_lab
        out = self.sigmoid(out)                             # batch_size x n_lab

        return out
