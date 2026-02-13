import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Encoder(nn.Module):
    def __init__(self, 
                 input_len, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2, 
                 dim_feedforward=128, 
                 kernel_size=5, 
                 dropout=0.3,
                 lr=3e-4,
                 batch_size=64,
                 epochs=35,
                 device=None):
        super(Encoder, self).__init__()
        self.epochs=epochs
        self.batch_size=batch_size
        self.lr = lr
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_model, 
                      kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, input_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.to(self.device)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv1d(x)
        
        x = x.permute(0, 2, 1)
        
        x = x + self.pos_embedding
        
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)
        
        logits = self.classifier(x)
        return logits

    def fit(self, X_train, y_train,verbose=True):
        self.train()
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training on {self.device} for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 5 == 0 :
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {total_loss / len(dataloader):.4f}")
        
        return self

    def predict(self, X_test, batch_size=32):
        self.eval()
        
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(self.device)
                
                logits = self.forward(batch_X)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                predictions.extend(preds.cpu().numpy())
                
        return np.vstack(predictions).flatten().astype(int)

import numpy as np
from sklearn.model_selection import train_test_split

if __name__=="__main__":

    X = np.random.rand(8600, 128).astype(np.float32)
    y = np.random.randint(0, 2, 8600)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Encoder(input_len=128,epochs=5)


    print("Training...")
    model.fit(X_train, y_train)


    print("\nPredicting...")
    y_pred = model.predict(X_test)

    print(f"Predictions: {y_pred[:10]}")

