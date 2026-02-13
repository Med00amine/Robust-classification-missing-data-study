from .CNNarchitecture import CNN   

class CNNModel:
    def __init__(self):
        self.model = CNN()

    def train(self, X_train, y_train):
        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(y_train, "values"):
            y_train = y_train.values

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if hasattr(X_test, "values"):
            X_test = X_test.values

        return self.model.predict(X_test)


