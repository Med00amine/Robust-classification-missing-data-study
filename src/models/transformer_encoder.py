from .Encoder import Encoder
from sklearn.preprocessing import StandardScaler

class TransformerModel:
    def __init__(self, input_len):
        self.scaler = StandardScaler()
        self.model = Encoder(input_len=input_len)

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
