import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data():
    df = sns.load_dataset('titanic')
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    
    data = df[features + ['survived']].copy()
    data['age'].fillna(data['age'].median(), inplace=True)
    data['sex'] = data['sex'].map({'female': 0, 'male': 1})
    data.dropna(inplace=True)
    
    X = data[features].values.astype(np.float32)
    y = data['survived'].values.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # Convert arrays to PyTorch Tensors
    X_tr = torch.tensor(X_train)
    y_tr = torch.tensor(y_train).unsqueeze(1)
    X_te = torch.tensor(X_test)
    y_te = torch.tensor(y_test).unsqueeze(1)
    
    return X_tr, X_te, y_tr, y_te, y_test

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),       
            nn.Linear(16, 8),
            nn.ReLU(),       
            nn.Linear(8, 1), 
            nn.Sigmoid()      
        )
    
    def forward(self, x):
        return self.net(x)

def train_model(model, X_tr, y_tr, epochs=100):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        
        y_pred = model(X_tr)
        
        loss = criterion(y_pred, y_tr)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, y_test = load_and_preprocess_data()
    
    model = SimpleNet()
    train_model(model, X_tr, y_tr, epochs=100)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        probs = model(X_te).numpy().flatten()
        preds = (probs >= 0.5).astype(int)
    
    print(classification_report(y_test, preds, target_names=['Did not survive', 'Survived']))