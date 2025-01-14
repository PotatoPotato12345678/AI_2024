import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD

import pickle

with open('pickle/data.pickle', mode='rb') as f:
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

accuracy_list = []
recall_list = []
precision_list = []

#initial situation
# epoch_num = 5
# btch_size = 128
# nodes_first_hidden = 16
# nodes_second_hidden = 8
# dropout_rate = 0.5

epoch_num = 1
btch_size = 4800
nodes_first_hidden = 1
nodes_second_hidden = 1
dropout_rate = 0.5

# test situation
v_epoch_num = np.arange(1,20)
v_btch_size = np.logspace(0, 7, num=8, base=2)
v_nodes_first_hidden = np.logspace(0, 7, num=8, base=2)
v_nodes_second_hidden = np.logspace(0, 7, num=8, base=2)
v_dropout_rate = np.linspace(0,1,11)

v_list = [v_epoch_num,v_btch_size,v_nodes_first_hidden,v_nodes_second_hidden,v_dropout_rate]

# for i, v in enumerate(v_list):
#     epoch_num           = v[i] if i == 0 else 1
#     btch_size           = v[i] if i == 1 else 4800
#     nodes_first_hidden  = v[i] if i == 2 else 1
#     nodes_second_hidden = v[i] if i == 3 else 1
#     dropout_rate        = v[i] if i == 4 else 0.5 




for train_index, test_index in skf.split(X_tfidf, y_encoded):
    X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Build the model
    model = Sequential([
        Dense(X_tfidf.shape[1], input_dim=X_tfidf.shape[1], activation='relu'),
        Dropout(dropout_rate),
        Dense(nodes_first_hidden, activation='relu'),
        Dropout(dropout_rate),
        Dense(nodes_second_hidden, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    sgd = SGD(learning_rate=0.01)
    
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epoch_num, batch_size=btch_size, verbose=0)  # Adjust epochs as needed
    
    # Evaluate the model
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metrics
    accuracy_list.append(np.mean(y_pred == y_test))
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    recall_list.append(report['weighted avg']['recall'])
    precision_list.append(report['weighted avg']['precision'])
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Normalize the confusion matrix row-wise
    conf_matrix = conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"./result/img/{fold}_confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(to_categorical(y_test, num_classes=len(label_encoder.classes_)).ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f"./result/img/{fold}_ROC.png")
    
    # Learning curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./result/img/{fold}Learning_Curve.png")
    fold += 1

# Final metrics
print(f'Average Accuracy: {np.mean(accuracy_list):.2f}')
print(f'Average Recall: {np.mean(recall_list):.2f}')
print(f'Average Precision: {np.mean(precision_list):.2f}')


