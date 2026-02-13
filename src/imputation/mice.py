from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def mice_impute(X_train, X_test):

    imputer = IterativeImputer(random_state=42)
    imputer.fit(X_train)

    return imputer.transform(X_test)
