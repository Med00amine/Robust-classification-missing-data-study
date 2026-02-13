import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class CNN(nn.Module):
    def __init__(self, epochs=35, batch_size=64,device=None,lr=0.001):
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        
        self.to(self.device)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.gap(x).view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def fit(self, X, y, verbose=True):
        self.train()

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(loader):.4f}")
        return self

    def predict(self, X):
        self.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
    
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)
                
                logits = self(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                predictions.extend(preds.cpu().numpy())

        return np.vstack(predictions).flatten().astype(int)
    
    
import numpy as np
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    X = np.random.rand(10, 128).astype(np.float32)
    y = np.random.randint(0, 2, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    model = CNN()

    print("Training...")
    model.fit(X_train, y_train)

    print("\nPredicting...")
    y_pred = model.predict(X_test)

    print(f"Predictions: {y_pred[:10]}")
