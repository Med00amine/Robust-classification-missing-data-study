# Operational Robustness Study under Missing Data

##  Objective

This project evaluates the operational robustness of multiple classification models under different missing data mechanisms.

The classifier is trained once on clean data and then evaluated on corrupted test data after applying different imputation strategies.

The goal is to measure how well the same trained model survives input degradation.

---
Run the Experiment
```bash
python -m src.experiments.run_experiments
```
## Experimental Design

### 1️ Data Split
- Train/Test split performed once.
- Training data remains clean.
- Test data is corrupted during evaluation.

### 2️ Missing Mechanisms
- MCAR (Missing Completely At Random)
- MAR (Missing At Random)
- MNAR (Missing Not At Random)

Missing rates tested:
- 10%
- 20%
- 30%
- 50%
- 70%

### 3️ Imputation Methods
- Zero-Fill (baseline degradation)
- Mean Imputation
- KNN Imputation
- MICE (Iterative Imputer)
- Autoencoder-based Imputation

### 4️ Models Evaluated
- Random Forest
- KNN
- MLP
- 1D CNN
- Transformer Encoder

---

##  Operational Robustness Protocol(loop)

1. Train classifier on clean training data.
2. Train Autoencoder on clean training data.
3. For each missing mechanism and rate:
   - Corrupt test data only.
   - Apply imputation.
   - Evaluate using the same trained classifier.
4. Log metrics and generate robustness curves.

---

## 📊 Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---


### Run the Experiment
```bash
python -m src.experiments.run_experiments
```

This will train all models, corrupt test data with different missing mechanisms and rates, apply imputation strategies, and generate comprehensive robustness curves.

---

## 📈 Results & Outputs

### Raw Results: CSV Format
**Location:** `results/csv/baseline_results.csv`

Contains detailed performance metrics for every combination:

| model | mechanism | rate | imputation | accuracy | precision | recall | f1 |
|-------|-----------|------|-----------|----------|-----------|--------|-----|
| MLP | MCAR | 0.1 | mean | 0.95 | 0.94 | 0.96 | 0.95 |
| ... | ... | ... | ... | ... | ... | ... | ... |

Each row represents one experiment configuration, letting you dive deep into specific comparisons.

---

### Visualizations: Robustness Curves

Four key visual summaries are generated showing how different imputation strategies affect model robustness:

#### **1. Accuracy.png**
Shows the overall correctness across all predictions. 
- **What to look for:** Models that maintain high accuracy even at high missing rates are more robust
- **Interpretation:** As missing data rate increases (left to right), accuracy typically drops—curves that stay flat are winners
- **Use case:** Pick this if overall correctness is your priority

#### **2. Precision.png**
Measures false positive rate—when the model predicts positive, how often is it right?
- **What to look for:** Imputation methods that prevent false alarms despite data corruption
- **Interpretation:** Critical for applications where wrong "yes" predictions are costly (medical diagnosis, fraud detection, etc.)
- **Use case:** Choose high-precision methods when false positives are expensive

#### **3. Recall.png**
Measures false negative rate—of all actual positives, how many did the model catch?
- **What to look for:** Which imputation keeps the model from missing important cases
- **Interpretation:** As missing data increases, recall often suffers most—good methods minimize this drop
- **Use case:** Choose high-recall methods when missing a positive case is critical (disease screening, anomaly detection)

#### **4. F1.png**
The balanced metric combining precision and recall into one score.
- **What to look for:** Smooth curves with minimal degradation across missing rates
- **Interpretation:** F1 gives a single number per configuration—useful for overall comparison without choosing precision vs. recall trade-offs
- **Use case:** Default choice when you need balanced performance

---

## 💡 How to Interpret the Results

### Reading the Curves
Each curve in the plots represents one imputation method or model combination:
- **X-axis:** Missing data rate (10% → 70%)
- **Y-axis:** Performance metric (0.0 to 1.0)
- **Shape matters:** Flat lines = robust; steep drops = fragile

### Key Insights to Look For

1. **Best overall:** Which imputation method keeps curves highest across all rates?
2. **Graceful degradation:** Which method loses the least performance at high missing rates?
3. **Missing mechanism sensitivity:** Does one mechanism hurt performance more than others?
4. **Model differences:** Do some models stay robust while others crumble?

### Example Interpretation
If your Accuracy curve shows:
- MICE imputation stays near 0.92 even at 70% missing ✅ **Robust**
- Zero-Fill drops to 0.65 at 70% missing ❌ **Fragile**

→ MICE is the better choice for real-world scenarios with unpredictable data quality issues.

---

## 📁 Project Structure

```
.
├── data/
│   └── adulteration_dataset_26_08_2021.csv
├── src/
│   ├── data/           # Data loading & train/test split
│   ├── missing/        # Missing data mechanisms (MCAR, MAR, MNAR)
│   ├── imputation/     # Various imputation strategies
│   ├── models/         # Classification models (RF, KNN, MLP, CNN, Transformer)
│   └── evaluation/     # Metrics & visualization
├── results/
│   ├── csv/            # Detailed results table
│   └── figures/        # Robustness curves for all model/mechanism/rate combos
└── readme             # This file
```

---

## 📊 Understanding the Experimental Design

**Why this approach?**

Real-world models don't get retrained when data quality degrades—a model deployed in production must handle input corruption gracefully. This study simulates that scenario:

1. **Train once** on clean data
2. **Corrupt only test data** (simulating real degradation)
3. **Test imputation robustness** by seeing how well the same trained model handles different fixes

This reveals which imputation strategies actually help your models survive in the wild, not just which improve training metrics.

---

##  What Each Missing Mechanism Means

- **MCAR (Missing Completely At Random):** Data is lost randomly, no pattern. Easiest to handle.
- **MAR (Missing At Random):** Missingness depends on observed variables. Medium difficulty.
- **MNAR (Missing Not At Random):** Missingness depends on unobserved data. Hardest case—real-world often looks like this.

The experiment tests all three so you know your imputation choice works even in worst-case scenarios.

---

##  License and Citation

This project is part of a robustness study under missing data. Results are in `results/csv/baseline_results.csv`
