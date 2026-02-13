import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# -------------------------------------------------------
# Utility
# -------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------------
# 1️⃣ Bar plots for ALL metrics
# -------------------------------------------------------
def plot_metrics_per_model(results_df, save_path):
    ensure_dir(save_path)

    metrics = ["accuracy", "precision", "recall", "f1"]

    for _, row in results_df.iterrows():
        model_name = row["model"]
        values = [row[m] for m in metrics]

        plt.figure()
        bars = plt.bar(metrics, values)

        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"Metrics - {model_name}")

        # 🔥 Add numbers on top of bars
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"metrics_{model_name}.png"))
        plt.close()

# -------------------------------------------------------
# 2️⃣ Confusion Matrix per Model
# -------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix_detailed(
    y_true,
    y_pred,
    model_name,
    mechanism,
    rate,
    imputation,
    base_path
):
    """
    Save a detailed confusion matrix plot.
    """

    # Directory structure
    save_dir = os.path.join(
        base_path,
        model_name,
        mechanism,
        f"rate_{rate}"
    )

    ensure_dir(save_dir)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )

    plt.title(
        f"{model_name} | {mechanism} | Rate={rate} | {imputation}"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = f"cm_{imputation}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# -------------------------------------------------------
# 3️⃣ Save metrics table as CSV
# -------------------------------------------------------

def save_results_table(results_df, save_path):
    ensure_dir(save_path)

    results_df.to_csv(
        os.path.join(save_path, "baseline_results.csv"),
        index=False
    )


# 4️⃣ Robustness Curves (All Imputations - Model Comparison)

def plot_robustness_curves(results_df, save_path):
    

    os.makedirs(save_path, exist_ok=True)

    imputations = results_df["imputation"].unique()
    models = results_df["model"].unique()
    mechanisms = results_df["mechanism"].unique()

    metrics = ["f1", "accuracy", "precision", "recall"]

    for imp in imputations:

        # Ignore clean baseline for curve plotting
        subset_imp = results_df[
            (results_df["imputation"] == imp) &
            (results_df["mechanism"] != "CLEAN")
        ]

        if subset_imp.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):

            ax = axes[i]

            for model in models:

                subset_model = subset_imp[
                    subset_imp["model"] == model
                ].sort_values("rate")

                if subset_model.empty:
                    continue

                ax.plot(
                    subset_model["rate"],
                    subset_model[metric],
                    marker="o",
                    linewidth=2.5,
                    label=model
                )

            ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
            ax.set_xlabel("Missing Rate")
            ax.set_ylabel(metric.upper())
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(models))

        fig.suptitle(
            f"Robustness Curves - Imputation: {imp}",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(
            os.path.join(save_path, f"robustness_{imp}.png"),
            dpi=300
        )

        plt.close()
