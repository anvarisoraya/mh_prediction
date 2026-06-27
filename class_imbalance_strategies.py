import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

strategies = {
    "Naive": LogisticRegression(),
    "Class Weights": LogisticRegression(class_weight='balanced'),
    "SMOTE": LogisticRegression(),
    "Undersampling": LogisticRegression()
}

for name, model in strategies.items():
    print(f"\n--- Strategy: {name} ---")
    
    if name == "SMOTE":
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        model.fit(X_res, y_res)
    elif name == "Undersampling":
        X_res, y_res = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
        model.fit(X_res, y_res)
    elif name == "SMOTETomek":
        X_res, y_res = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
        model.fit(X_res, y_res)
    else:
        model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))