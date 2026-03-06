from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import sys
import os

# Add parent folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import build_model

import pickle
with open("./Dataset/sleep_stage_dataset.pkl","rb") as f:
    dataset = pickle.load(f)
signals = dataset['signals']
window_time = dataset['window_time']
labels = dataset['labels']
participant_ids =dataset['participants']

for i in range(len(labels)):
    if labels[i]=="Body event":
        labels[i]="Normal"
    elif labels[i]=="Mixed Apnea":
        labels[i]="Hypopnea"
    else:
        continue
# print(np.unique(labels))
label_map = {'Normal':0, 'Hypopnea':1, 'Obstructive Apnea':2}
y_labels = np.array([label_map[label] for label in labels])
# print(y_labels)
signals = signals.reshape(8800,960,-1)
y_labels = y_labels.reshape(8800,-1)


X=signals
y = y_labels
# normalize per window
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

unique_participants = np.unique(participant_ids)

fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1s = []
fold_cms = []



for test_pid in unique_participants:
    train_idx = np.where(participant_ids != test_pid)[0]
    test_idx  = np.where(participant_ids == test_pid)[0]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(X_train.shape) 
    print(X_test.shape)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.ravel())
    class_weights = dict(zip(classes, class_weights))
    model = build_model()  
    print(classes)
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=1,
        class_weight=class_weights
    )
    
    # Evaluate metrics
    y_pred_probs = model.predict(X_test)
    
    # Storing accuracy, precision, recall, confusion matrix per fold
    # y_pred = np.argmax(y_pred_probs, axis=-1)
    y_pred = y_pred_probs.argmax(axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred,labels=classes)
    
    # Storing metrics for this fold
    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1s.append(f1)
    fold_cms.append(cm)
    
    print(f"Fold Test Participant {test_pid}  Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

print("\nLOPO CV Results")
print("Average Accuracy:", np.mean(fold_accuracies))
print("Average Precision:", np.mean(fold_precisions))
print("Average Recall:", np.mean(fold_recalls))
print("Average F1-Score:", np.mean(fold_f1s))