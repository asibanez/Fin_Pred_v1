# v1 -> Preprocesses dataset to correct mistake with empty headlines

# Imports
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModel

#%% DataClass definition
class News_dataset(Dataset):
    def __init__(self, data_df):
        
        ####### Preprocess empty paragraphs
        self.empty_token_ids = torch.zeros(64).type(torch.LongTensor)
        self.empty_token_ids[0] = 101
        self.empty_token_ids[1] = 102
        self.empty_token_types = torch.zeros(64).type(torch.LongTensor)
        self.empty_att_masks = torch.zeros(64).type(torch.LongTensor)
        self.empty_att_masks[0] = 1
        self.empty_att_masks[1] = 1
        
        self.token_ids = data_df['token_ids'].to_list()
        self.token_types = data_df['token_types'].to_list()
        self.att_masks = data_df['att_masks'].to_list()
        
        self.aux = []
        for x in tqdm(self.token_ids, desc = 'class instantiation: token_ids'):
            if type(x) == float:
                self.aux.append(self.empty_token_ids)
            else:
                self.aux.append(x[0])
        self.token_ids = self.aux
        
        self.aux = []
        for x in tqdm(self.token_types, desc = 'class instantiation: token_types'):
            if type(x) == float:
                self.aux.append(self.empty_token_types)
            else:
                self.aux.append(x[0])
        self.token_types = self.aux        
            
        self.aux = []
        for x in tqdm(self.att_masks, desc = 'class instantiation: att_masks'):
            if type(x) == float:
                self.aux.append(self.empty_att_masks)
            else:
                self.aux.append(x[0])
        self.att_masks = self.aux
        #######
        
        self.token_ids = torch.stack(self.token_ids)
        self.token_types = torch.stack(self.token_types)
        self.att_masks = torch.stack(self.att_masks)
        self.labels = torch.LongTensor(list(data_df['label1']))
                                        
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
        self.fc_out = nn.Linear(in_features = self.h_dim, out_features = self.n_labels)

        # Softmax
        #self.softmax = nn.Softmax(dim = 1)

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.h_dim)

    def forward(self, X_token_ids, X_token_types, X_att_masks):
        # BERT encoder
        X_bert = {'input_ids': X_token_ids,
                  'token_type_ids': X_token_types,
                  'attention_mask': X_att_masks}
        out = self.bert_model(**X_bert,
                              output_hidden_states = True)       # Tuple
        out = out['pooler_output']                               # batch_size x h_dim
        
        # Multi-label classifier      
        out = self.bn1(out)                                      # batch_size x h_dim
        out = self.fc_out(out)                                   # batch_size x n_lab

        return out
