import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class AutoencoderImputer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        latent_dim=128,
        corruption_rate=0.3,
        epochs=80,
        lr=1e-3,
        device=None,
    ):
        super().__init__()

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.epochs = epochs
        self.lr = lr
        self.corruption_rate = corruption_rate

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def corrupt_input(self, x):
        mask = torch.rand_like(x) > self.corruption_rate
        return x * mask

    def fit(self, X_train):

        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.train()

        for epoch in range(self.epochs):

            optimizer.zero_grad()

            corrupted = self.corrupt_input(X_tensor)

            outputs = self.forward(corrupted)

            loss = criterion(outputs, X_tensor)

            loss.backward()
            optimizer.step()

        return self

    def transform(self, X_corrupted):

        self.eval()

        X_filled = np.nan_to_num(X_corrupted, nan=0.0)
        X_tensor = torch.tensor(X_filled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            reconstructed = self.forward(X_tensor)

        return reconstructed.cpu().numpy()
