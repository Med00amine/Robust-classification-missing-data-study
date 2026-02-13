import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.data.splitter import split_data
from src.models.model_factory import get_model
from src.evaluation.metrics import evaluate
from src.evaluation.plots import (plot_confusion_matrix_detailed, plot_robustness_curves, save_results_table, plot_metrics_per_model)

from src.missing import inject_mcar, inject_mar, inject_mnar
from src.imputation import mean_impute, knn_impute, mice_impute
from src.imputation.autoencoder import AutoencoderImputer


DATA_PATH = "data/adulteration_dataset_26_08_2021.csv"
TARGET_COL = "Concentration_Class"

FIGURE_PATH = "results/figures"
CSV_PATH = "results/csv"


def main():
    import random
    import torch

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    
    # 1️ Load and Split
    
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df,
        TARGET_COL,
        drop_cols=["Concentration", "Brand", "Class"]
    )

    input_len = X_train.shape[1]

    # Convert once
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()


    # 2️ Train classifiers once

    model_names = ["RandomForest", "KNN", "MLP", "CNN", "Transformer"]
    trained_models = {}

    for name in model_names:
        print(f"Training clean model: {name}")
        model = get_model(name, input_len)
        model.train(X_train, y_train)
        trained_models[name] = model


    # 3️ Train Autoencoder once

    print("Training Autoencoder on clean train...")
    ae = AutoencoderImputer(input_dim=input_len)
    ae.fit(X_train_np)


    # 4️ Operational Robustness Loop

    mechanisms = {
        "MCAR": inject_mcar,
        "MAR": inject_mar,
        "MNAR": inject_mnar
    }

    rates = [0.1, 0.2, 0.3, 0.5, 0.7]

    all_results = []

    for mech_name, mech_func in mechanisms.items():

        for rate in rates:

            print(f"\nMechanism: {mech_name} | Rate: {rate}")

            # Corrupt test only
            X_test_miss = mech_func(X_test_np, rate)


            # Imputations
            imputations = {}

            # Zero-fill baseline
            imputations["Zero"] = np.nan_to_num(X_test_miss, nan=0.0)
            # Mean
            imputations["Mean"] = mean_impute(X_train_np, X_test_miss)
            # KNN
            imputations["KNN"] = knn_impute(X_train_np, X_test_miss)
            # MICE
            imputations["MICE"] = mice_impute(X_train_np, X_test_miss)
            # Autoencoder
            imputations["AE"] = ae.transform(X_test_miss)

            # Evaluate all models
            for imp_name, X_test_imp in imputations.items():

                for model_name, model in trained_models.items():

                    y_pred = model.predict(X_test_imp)

                    metrics = evaluate(y_test, y_pred)
                    
                    # Save detailed confusion matrix for each model-imputation combination
                    plot_confusion_matrix_detailed(
                        y_test,
                        y_pred,
                        model_name,
                        mech_name,
                        rate,
                        imp_name,
                        FIGURE_PATH
                    )
                    
                    metrics.update({
                        "model": model_name,
                        "mechanism": mech_name,
                        "rate": rate,
                        "imputation": imp_name
                    })

                    all_results.append(metrics)


    # 5️ Save results
    results_df = pd.DataFrame(all_results)
    plot_robustness_curves(results_df, FIGURE_PATH)
    save_results_table(results_df, CSV_PATH)
    plot_metrics_per_model(results_df, FIGURE_PATH)
    print("\nOperational robustness experiment completed.")


if __name__ == "__main__":
    main()
