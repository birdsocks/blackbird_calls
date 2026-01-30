import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import bioacoustics_model_zoo as bmz
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

print("Loading model, embeddings, and data from previous scripts...")

# Load the trained model
birdnet = bmz.BirdNET()
birdnet.load("/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/blackbirdcallv1.model")

# Load embeddings
emb_val = np.load('emb_val.npy')
emb_test = np.load('emb_test.npy')

# Load classes
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load datasets
resampled_train_set = pd.read_csv('resampled_train_set.csv', index_col=[0,1,2])
resampled_val_set = pd.read_csv('resampled_val_set.csv', index_col=[0,1,2])
test_set = pd.read_csv('test_set.csv', index_col=[0,1,2])
train_set = pd.read_csv('train_set_onehot.csv')  # Original train set for comparison
val_set = pd.read_csv('val_set_onehot.csv')  # Original val set for comparison

print(f"Loaded model with {len(classes)} classes")
print(f"Classes: {classes}")

# Plot an histogram per calltype of the prediction scores and FP vs TP
print("\nGenerating validation predictions...")
preds = birdnet.network(torch.tensor(emb_val).cuda()).detach().cpu()

#validation predictions

for class_name in classes:
    class_idx = list(resampled_val_set.columns).index(class_name) # get index of class
    plt.figure()
    #plt.hist(preds[:, class_idx].numpy(), bins=50, alpha=0.7, label='All Predictions')
    
    # True Positives
    tp_indices = (resampled_val_set[class_name] == 1).values # get indices of true positives
    plt.hist(preds[tp_indices, class_idx].numpy(), bins=25, alpha=0.5, density=True, label='True Positives')
    
    # False Positives
    fp_indices = (resampled_val_set[class_name] == 0).values # get indices of false positives
    plt.hist(preds[fp_indices, class_idx].numpy(), bins=25, alpha=0.5, density=True, label='False Positives')
    
    plt.title(f'Prediction Score Distribution for {class_name}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

#Validation Classification report
# pred class is the class with the highest score
pred_labels = np.argmax(preds.numpy(), axis=1)
resampled_val_set_labels = np.argmax(resampled_val_set.values, axis=1)
report = classification_report(resampled_val_set_labels, pred_labels, target_names=classes, zero_division=0)
print(report)

#Test Classification report
# Plot an histogram per calltype of the prediction scores and FP vs TP
preds = birdnet.network(torch.tensor(emb_test).cuda()).detach().cpu()

for class_name in classes:
    class_idx = list(test_set.columns).index(class_name) # get index of class
    plt.figure()
    #plt.hist(preds[:, class_idx].numpy(), bins=50, alpha=0.7, label='All Predictions')
    
    # True Positives
    tp_indices = (test_set[class_name] == 1).values # get indices of true positives
    plt.hist(preds[tp_indices, class_idx].numpy(), bins=25, alpha=0.5, density=True, label='True Positives')
    
    # False Positives
    fp_indices = (test_set[class_name] == 0).values # get indices of false positives
    plt.hist(preds[fp_indices, class_idx].numpy(), bins=25, alpha=0.5, density=True, label='False Positives')
    
    plt.title(f'Prediction Score Distribution for {class_name}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# pred class is the class with the highest score
pred_labels = np.argmax(preds.numpy(), axis=1)
test_labels = np.argmax(test_set.values, axis=1)
report = classification_report(test_labels, pred_labels, target_names=classes, zero_division=0)
print(report)

# plot a table of number of samples in training set per class and the F1score, precision and recall in test set per class
test_set = test_set.astype(float)
# Convert labels to float (in case they're boolean or object type)
train_labels = resampled_train_set.astype(float).values
val_labels = resampled_val_set.astype(float).values
test_labels = test_set.astype(float).values
test_labels = np.argmax(test_set.values, axis=1)
summary_data = []
for i, class_name in enumerate(classes):
    n_samples_train = resampled_train_set[class_name].sum()
    n_samples_val = resampled_val_set[class_name].sum()
    n_original_train = train_set[class_name].sum()
    n_original_val = val_set[class_name].sum()

    precision = precision_score(test_labels, pred_labels, average=None, zero_division=0)[i]
    recall = recall_score(test_labels, pred_labels, average=None, zero_division=0)[i]
    f1 = f1_score(test_labels, pred_labels, average=None, zero_division=0)[i]

    summary_data.append({
        'Class': class_name,
        'Num Samples (Train)': n_samples_train,
        'Num Samples (Original Train)': n_original_train,
        'Num Samples (Val)': n_samples_val,
        'Num Samples (Original Val)': n_original_val,
        'Precision (Test)': precision,
        'Recall (Test)': recall,
        'F1 Score (Test)': f1
    })
summary_df = pd.DataFrame(summary_data)
summary_df

#Plot normalized confusion matrix
cm = confusion_matrix(test_labels, pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.show() 