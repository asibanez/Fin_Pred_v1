# Computes metrics

#%% Imports
import os
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#%% Path definitions
base_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/00_run_test'

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
Y_pred_score = test_results['Y_test_prediction_scores']
Y_ground_truth = test_results['Y_test_ground_truth']
Y_pred_label = test_results['Y_test_prediction_binary']

#%% Print ground truth class balance
ratio_0 = Y_ground_truth.count(0) / len(Y_ground_truth)
ratio_1 = Y_ground_truth.count(1) / len(Y_ground_truth)
ratio_2 = Y_ground_truth.count(2) / len(Y_ground_truth)

print(f'\nRatio class 0 = {ratio_0 * 100:.2f}%')
print(f'Ratio class 1 = {ratio_1 * 100:.2f}%')
print(f'Ratio class 2 = {ratio_2 * 100:.2f}%')

#%% Print results    
print(classification_report(Y_ground_truth, Y_pred_label))

#%% Plot learning curves
plt.plot(train_results['training_loss'], label = 'train')
plt.plot(train_results['validation_loss'], label = 'val')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
#plt.ylim(0.05, 0.75)
plt.grid()
plt.show()
