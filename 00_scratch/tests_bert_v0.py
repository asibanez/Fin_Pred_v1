#%% Imports
from transformers import AutoModel
from transformers import AutoTokenizer

#%% Global initialization
model_name = 'ProsusAI/finbert'

#%% Tokenization
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
tokens = bert_tokenizer('hey! son of a bitch, how are you fucking doing?',
                        return_tensors = 'pt',
                        padding = 'max_length',
                        truncation = True,
                        max_length = 100)
string_test = bert_tokenizer.decode(tokens['input_ids'].tolist()[0])

#%% Print results
print(tokens)
print(string_test)

#%% Compute predictions
bert_model = AutoModel.from_pretrained(model_name)
pred_1 = bert_model(**tokens, output_hidden_states = True)
pred_2 = bert_model(**tokens)['pooler_output']

#%% Print results
print(pred_2)
print(f'Pred size = {pred_2.size()}')
