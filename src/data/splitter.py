from sklearn.model_selection import train_test_split

def split_data(df, target_col, drop_cols=None, test_size=0.2, random_state=42):


    X = df.drop(columns=[
    "Concentration_Class",
    "Concentration",
    "Brand",
    "Class"
    ])

    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
