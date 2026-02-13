from sklearn.impute import KNNImputer

def knn_impute(X_train, X_test, n_neighbors=6):

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(X_train)

    return imputer.transform(X_test)
