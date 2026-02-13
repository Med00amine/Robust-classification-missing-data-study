from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class MLPModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42
        )

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
