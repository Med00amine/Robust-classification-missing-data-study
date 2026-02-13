from sklearn.impute import SimpleImputer

def mean_impute(X_train, X_test):

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)

    return imputer.transform(X_test)
