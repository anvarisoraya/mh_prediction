import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('Student Mental health.csv')
df.columns = [
    'Timestamp', 'Gender', 'Age', 'Course', 'Year_of_Study', 
    'CGPA', 'Marital_Status', 'Depression', 'Anxiety', 
    'Panic_Attack', 'Treatment'
]
df = df.dropna()

le = LabelEncoder()
le_course = LabelEncoder()
le_cgpa = LabelEncoder()

df['Course'] = le_course.fit_transform(df['Course'])
df['CGPA'] = le_cgpa.fit_transform(df['CGPA']) # Encodes string ranges to numbers

cat_cols = ['Gender', 'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df[['Marital_Status', 'Course', 'CGPA', 'Anxiety', 'Panic_Attack']]
y = df['Depression']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      
    random_state=42,    
    stratify=y          
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   

print('Before scaling - CGPA column (first 5):\n', X_train.iloc[:5, 2].values)
print('After scaling  - CGPA column (first 5):\n', X_train_scaled[:5, 2].round(2))


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}


for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    train_preds = model.predict(X_train_scaled)
    test_preds  = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test, test_preds)

    
    results[name] = {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
    print(f'{name:<25} Train: {train_acc:.2%}  |  Test: {test_acc:.2%}')