from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class KNNModel:
    def __init__(self, n_neighbors=6):
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
