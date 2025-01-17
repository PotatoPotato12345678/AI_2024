import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD
import pandas as pd
import pickle

# read the dataset
with open('label/data.pickle', mode='rb') as f:
  data = pickle.load(f)

# decompose it to text and label
texts = [data[d]["abstract"] for d in data]
labels = [data[d]["label"] for d in data]

# Convert data to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts).toarray()

# Encode labels into integer number
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# 5 cross-validation setup
skf = StratifiedKFold(n_splits=5)
fold = 1

epoch_num = 23              # epoch number
btch_size = 64              # batch size
nodes_first_hidden = 256    # the number of units in the first hidden layer
nodes_second_hidden = 128   # the number of units in the second hidden layer
dropout_rate = 0.2          # dropout rate

print("---------------------------------------------")
print(f"epoch_num          : {epoch_num}")
print(f"btch_size          : {btch_size}")
print(f"nodes_first_hidden : {nodes_first_hidden}")
print(f"nodes_second_hidden: {nodes_second_hidden}")
print(f"dropout_rate       : {dropout_rate}")
print("---------------------------------------------")

# since it's 5 cross-validation, lists stores values to plot or get average later.
all_fpr = [] # false positive rate
all_tpr = [] # true positive rate
roc_aucs = [] # ROC
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_conf_matrices = []

for train_index, test_index in skf.split(X_tfidf, y_encoded):
    # spliited training data and test data are assigned.
    X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]


    model = Sequential([
        Input(shape=(X_tfidf.shape[1],)), # Input layer
        Dropout(dropout_rate),            # Dropout layer
        Dense(nodes_first_hidden, activation='relu'), # First hidden layer with relu
        Dropout(dropout_rate),            # Dropout layer
        Dense(nodes_second_hidden, activation='relu'), # Second hidden layer with relu
        Dropout(dropout_rate),            # Dropout layer
        Dense(len(label_encoder.classes_), activation='softmax') # Softmax layer with 3 units since 3 categories
    ])

    sgd = SGD(learning_rate=0.01) # the optimization uses Stochastic gradient descent with learning rate = 0.01
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # model is set up. The loss function is Loss =−log(P​), where P is the true positive probability.

    history = model.fit(X_train, y_train, epochs=epoch_num, batch_size=btch_size, verbose=0) # apply the model to the dataset. verbose=0 makes no output on the terminal during the training.


    y_pred_proba = model.predict(X_test) # test with the test data, and output the probability for each class for each test abstracts.
    y_pred = np.argmax(y_pred_proba, axis=1) # take the index that has the highest probability => the predicted label (integer)

    # Metrics
    accuracy = np.mean(y_pred == y_test) # count the number of TRUE (after comparison) and devide them by the number of test data.
    all_accuracy.append(accuracy)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0) # outputs many result of metrics including precision, recall.
    recall = report['weighted avg']['recall']
    precision = report['weighted avg']['precision']
    f1 = 2 * (precision * recall) / (precision + recall) # f1 score is calculated based on the formula.

    all_recall.append(recall)
    all_precision.append(precision)
    all_f1.append(f1)

    # ROC Curve
    # fpr: False positive rate
    # tpr: True positive rate
    fpr, tpr, _ = roc_curve(to_categorical(y_test, num_classes=len(label_encoder.classes_)).ravel(), y_pred_proba.ravel())

    roc_auc = auc(fpr, tpr)  # calculate the area under the curve
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    roc_aucs.append(roc_auc)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred) # compute each component for confusion matrix.
    conf_matrix = conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True)  # A / B, A: convert it into float to output precise, B: compute the sum in each row. 
                                                                                           # Therefore, the confusion matrix focus on the probability of the prediction for each class. 
    all_conf_matrices.append(conf_matrix)



# Calculate mean ROC curve
mean_fpr = np.linspace(0, 1, 100)

interpolated_tprs = []
for fpr, tpr in zip(all_fpr, all_tpr):
    interpolated_tpr = np.interp(mean_fpr, fpr, tpr) # true positive value is interporated.
    interpolated_tprs.append(interpolated_tpr)

mean_tpr = np.mean(interpolated_tprs, axis=0) # take the mean
mean_tpr[-1] = 1.0 # ensure the end point is 1.0
mean_auc = auc(mean_fpr, mean_tpr) #calculate 

# Average confusion matrix
mean_conf_matrix = np.mean(all_conf_matrices, axis=0)


# Plot mean confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(mean_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
          xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Mean Confusion Matrix Across Folds")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(f"./result/final/img/Confusion_matrix.png")
plt.close()

# Plot mean ROC curve
plt.figure(figsize=(8, 8))
for i_r, roc_auc_value in enumerate(roc_aucs):
  plt.plot(all_fpr[i_r], all_tpr[i_r], alpha=0.3, label=f"Fold {i_r+1} ROC (AUC = {roc_auc_value:.2f})")
plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC (AUC = {mean_auc:.2f})", lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Mean ROC Curve Across Folds")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(f"./result/final/img/ROC.png")
plt.close()

# Averaged evaluation
averaged_evaluation = {
  "Accuracy": np.mean(all_accuracy),
  "Precision": np.mean(all_precision),
  "Recall": np.mean(all_recall),
  "F1-Score": np.mean(all_f1)
}

print("-----------------------------------")
print("Accuracy :\t{0}".format(np.mean(all_accuracy)))
print("Precision:\t{0}".format(np.mean(all_precision)))
print("Recall   :\t{0}".format(np.mean(all_recall)))
print("F1-Score :\t{0}".format(np.mean(all_f1)))
print("-----------------------------------")

# Save the averaged evaluation metrics
with open(f"./result/final/ave_evaluation", "wb") as f:
  pickle.dump(averaged_evaluation, f)

#save the model
with open(f"./model/model", "wb") as f:
  pickle.dump(model, f)