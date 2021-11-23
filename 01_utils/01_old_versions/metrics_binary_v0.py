# Computes metrics

#%% Imports
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

#%% Function definitions
# Sigmoid
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#%% Metrics 
def compute_metrics(Y_ground_truth, Y_pred_binary, Y_pred_score):
    tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_pred_binary).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(Y_ground_truth, Y_pred_score)
    
    return precision, recall, f1, auc

#%% Path definitions
base_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/01_ProsusAI_finbert/00_binary/02_toy_TEST'

#%% Global initialization
random.seed(1234)
threshold = 0.5

#%% Read data json
train_results_path = os.path.join(base_path, 'train_results.json')
test_results_path = os.path.join(base_path, 'test_results.json')

with open(train_results_path) as fr:
    train_results = json.load(fr)
    
with open(test_results_path) as fr:
    test_results = json.load(fr)
    
#%% Extract results
Y_pred_logits = test_results['Y_test_prediction_scores']
Y_ground_truth = test_results['Y_test_ground_truth']
Y_pred_label = test_results['Y_test_prediction_binary']

Y_pred_logits = [x[0] for x in Y_pred_logits]
Y_ground_truth = [x[0] for x in Y_ground_truth]

#%% Print ground truth class balance
ratio_0 = Y_ground_truth.count(0) / len(Y_ground_truth)
ratio_1 = Y_ground_truth.count(1) / len(Y_ground_truth)

print(f'\nRatio class 0 = {ratio_0 * 100:.2f}%')
print(f'Ratio class 1 = {ratio_1 * 100:.2f}%')

#%% Compute binary results
Y_pred_scores = [sigmoid(x) for x in Y_pred_logits]
Y_pred_binary = [round(x) for x in Y_pred_scores]
print(pd.value_counts(Y_pred_binary))

#%% Generate random results
random_pred_score = []

for i in range(0, len(Y_pred_scores)):
    random_pred_score.append(random.random())
    
random_pred_binary = [1 if x >= threshold else 0 for x in random_pred_score]

#%% Print results    
print(classification_report(Y_ground_truth, Y_pred_binary))

#%% Compute metrics
precision, recall, f1, auc = compute_metrics(Y_ground_truth,
                                             Y_pred_binary,
                                             Y_pred_scores)

print(f'\nPrecision =\t{precision:.2f}')
print(f'Recall =\t{recall:.2f}')
print(f'F1 =\t\t{f1:.2f}')
print(f'AUC =\t\t{auc:.2f}')

#%% Plot ROC curve
fpr_model, tpr_model, threshold_roc_model = roc_curve(Y_ground_truth, Y_pred_scores)
fpr_rand, tpr_rand, threshold_roc_rand = roc_curve(Y_ground_truth, random_pred_score)
plt.plot(fpr_model, tpr_model, linestyle='--', label='Model')
plt.plot(fpr_rand, tpr_rand, linestyle=':', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.ylim(-0.05, 1.05)
plt.grid()
plt.show()    

#%% Plot Precision - recall curve
precision_model_g, recall_model_g, threshold_model_g = precision_recall_curve(Y_ground_truth, Y_pred_scores)
precision_rand_g, recall_rand_g, threshold_rand_g = precision_recall_curve(Y_ground_truth, random_pred_score)
plt.plot(recall_model_g, precision_model_g, linestyle='--', label='Model')
plt.plot(recall_rand_g, precision_rand_g, linestyle=':', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'upper right')
plt.ylim(-0.05, 1.05)
plt.grid()
plt.show()

#%% Plot learning curves
plt.plot(train_results['training_loss'], label = 'train')
plt.plot(train_results['validation_loss'], label = 'val')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
#plt.ylim(0.05, 0.75)
plt.grid()
plt.show()

