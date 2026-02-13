from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"   # Handle class imbalance
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
