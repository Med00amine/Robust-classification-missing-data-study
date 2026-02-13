import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Convert target to binary
    df["Concentration_Class"] = (df["Concentration"] > 0).astype(int)

    return df
