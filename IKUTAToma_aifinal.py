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

with open('label/data.pickle', mode='rb') as f:
  data = pickle.load(f)

# Example text data (replace with your actual data)
texts = [data[d]["abstract"] for d in data]
labels = [data[d]["label"] for d in data]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts).toarray()

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# 5-fold cross-validation setup
skf = StratifiedKFold(n_splits=5)
fold = 1

# epoch_num = 23
# btch_size = 64
# nodes_first_hidden = 256
# nodes_second_hidden = 128
# dropout_rate = 0.2
epoch_num = 1
btch_size = 2000
nodes_first_hidden = 1
nodes_second_hidden = 1
dropout_rate = 0.2

print("---------------------------------------------")
print(f"epoch_num          : {epoch_num}")
print(f"btch_size          : {btch_size}")
print(f"nodes_first_hidden : {nodes_first_hidden}")
print(f"nodes_second_hidden: {nodes_second_hidden}")
print(f"dropout_rate       : {dropout_rate}")
print("---------------------------------------------")

all_fpr = []
all_tpr = []
roc_aucs = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_conf_matrices = []

for train_index, test_index in skf.split(X_tfidf, y_encoded):
    X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Build the model
    model = Sequential([
        Input(shape=(X_tfidf.shape[1],)),
        Dropout(dropout_rate),
        Dense(nodes_first_hidden, activation='relu'),
        Dropout(dropout_rate),
        Dense(nodes_second_hidden, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    sgd = SGD(learning_rate=0.01)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epoch_num, batch_size=btch_size, verbose=0)

    # Evaluate the model
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Metrics
    accuracy = np.mean(y_pred == y_test)
    all_accuracy.append(accuracy)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    recall = report['weighted avg']['recall']
    precision = report['weighted avg']['precision']
    f1 = 2 * (precision * recall) / (precision + recall)

    all_recall.append(recall)
    all_precision.append(precision)
    all_f1.append(f1)

    # ROC Curve
    fpr, tpr, _ = roc_curve(to_categorical(y_test, num_classes=len(label_encoder.classes_)).ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    roc_aucs.append(roc_auc)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix = conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True)  # Normalize row-wise
    all_conf_matrices.append(conf_matrix)

    # Average confusion matrix
    mean_conf_matrix = np.mean(all_conf_matrices, axis=0)

    # Calculate mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

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

    with open(f"./model/model", "wb") as f:
      pickle.dump(model, f)