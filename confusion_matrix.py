import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)

train_dataset = load_dataset(
    "ourafla/Mental-Health_Text-Classification_Dataset", 
    data_files="mental_heath_unbanlanced.csv"
)

val_dataset = load_dataset(
    "ourafla/Mental-Health_Text-Classification_Dataset", 
    data_files="mental_health_combined_test.csv"
)


train_df = pd.DataFrame(train_dataset['train'])
val_df = pd.DataFrame(val_dataset['train'])


text_col = 'text'   
label_col = 'status' 

X_train_raw = train_df[text_col]
y_train = train_df[label_col]
X_val_raw = val_df[text_col]
y_val = val_df[label_col]

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train_raw)
X_val_vec   = vectorizer.transform(X_val_raw)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_val_vec)


acc  = model.score(X_val_vec, y_val)
prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_val, y_pred, average='weighted', zero_division=0)

print(f"Overall Accuracy:  {acc:.3f}  — Overall prediction accuracy across all classes")
print(f"Weighted Precision: {prec:.3f}  — Average success rate when model flags any category")
print(f"Weighted Recall:    {rec:.3f}  — Average proportion of actual cases caught")
print(f"Weighted F1 Score:  {f1:.3f}  — Harmonic balance of overall Precision and Recall")

print("\n📋 Detailed Category Breakdown:")
print(classification_report(y_val, y_pred, zero_division=0))

print("🎨 Plotting Confusion Matrix...")
cm = confusion_matrix(y_val, y_pred)

unique_classes = np.unique(y_val)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=unique_classes,
            yticklabels=unique_classes)
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Multiclass Mental Health Confusion Matrix')
plt.tight_layout()
plt.show()