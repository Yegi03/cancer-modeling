"""
src/models/gbm.py

This script demonstrates how to train and apply a Gradient Boosting Regressor
to make predictions on the "remaining 25%" of a dataset. The original PDE-based
model or assimilation code is omitted here for clarity, focusing only on the GBM
portion. Plots are generated comparing the 75% (training) data, the remaining 25%,
and the GBM predictions.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


def run_gbm_example():
    """
    Demonstrates training a GBM model on 75% of data
    and predicting on the remaining 25%.
    """
    # -------------------------------------------------------------------------
    # 1. Load Datasets
    # -------------------------------------------------------------------------
    # Adjust paths according to your folder structure
    full_data_path = "../../datasets/Qi100/Qi100.xlsx"  # 100% data
    sampled_data_path = "../../datasets/Qi100/75_qi100.xlsx"  # 75% data
    results_path_plot = "../../result/gbm_qi100_plot.png"  # Where to save plot

    print("[run_gbm_example] Loading datasets...")
    full_data = pd.read_excel(full_data_path)
    sampled_data = pd.read_excel(sampled_data_path)

    # -------------------------------------------------------------------------
    # 2. Identify the 25% Remaining Data
    # -------------------------------------------------------------------------
    # We'll merge on (var1, var2) so we can see which rows exist only in full_data
    print("[run_gbm_example] Identifying remaining 25% data...")
    merged = full_data.merge(sampled_data, on=['var1', 'var2'], how='left', indicator=True)
    remaining_25_data = merged[merged['_merge'] == 'left_only']
    remaining_25_time = remaining_25_data['var1'].values
    remaining_25_volume = remaining_25_data['var2'].values

    # -------------------------------------------------------------------------
    # 3. Prepare Data for GBM
    # -------------------------------------------------------------------------
    # X_train: times from the 75% dataset (training)
    # y_train: volumes from the 75% dataset
    # X_test: times for the 25% leftover
    print("[run_gbm_example] Preparing data for GBM training and inference...")
    X_train = sampled_data['var1'].values.reshape(-1, 1)
    y_train = sampled_data['var2'].values
    X_test = remaining_25_time.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # 4. Train Gradient Boosting Model
    # -------------------------------------------------------------------------
    print("[run_gbm_example] Training Gradient Boosting Regressor...")
    gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbm_model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 5. Predict using GBM on the remaining 25%
    # -------------------------------------------------------------------------
    print("[run_gbm_example] Predicting on the remaining 25% data...")
    gbm_predictions = gbm_model.predict(X_test)

    # -------------------------------------------------------------------------
    # 6. Plot Results
    # -------------------------------------------------------------------------
    print("[run_gbm_example] Generating plot...")
    plt.figure(figsize=(10, 6))

    # 75% (Training) data
    plt.scatter(sampled_data['var1'], sampled_data['var2'],
                color='orange', label='75% Training Data (Orange Circles)', alpha=0.7)

    # 25% (Remaining) data
    plt.scatter(remaining_25_time, remaining_25_volume,
                color='green', marker='^', label='Remaining 25% Data (Green Triangles)', alpha=0.7)

    # GBM Predictions on 25%
    plt.scatter(remaining_25_time, gbm_predictions,
                color='red', marker='x', label='GBM Predictions (Red Crosses)', alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Gradient Boosting Regressor - Prediction on Remaining 25%')
    plt.legend()
    plt.grid(True)

    # Save and show plot
    plt.savefig(results_path_plot, dpi=300)
    print(f"[run_gbm_example] Plot saved to {results_path_plot}")
    plt.show()

    print("[run_gbm_example] Done. GBM example completed.")


if __name__ == "__main__":
    run_gbm_example()