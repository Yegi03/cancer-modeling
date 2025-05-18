"""
src/models/rf.py

This script demonstrates training a RandomForestRegressor on the 75% (training) data
and predicting on the remaining 25%. It plots the training data, leftover data,
and Random Forest predictions for comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def run_rf_example():
    """
    Loads a 100% dataset (Qi100) and a 75% subset for training.
    Identifies the leftover 25%, trains a RandomForestRegressor,
    then plots predictions on the remaining data.
    """

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    # Adjust file paths according to your project layout
    full_data_path = "../../datasets/Qi100/Qi100.xlsx"       # Full (100%) data
    sampled_data_path = "../../datasets/Qi100/75_qi100.xlsx" # 75% training data
    results_plot_path = "../../result/rf_qi100_plot.png"     # Output plot

    print("[run_rf_example] Loading datasets...")
    full_data = pd.read_excel(full_data_path)
    sampled_data = pd.read_excel(sampled_data_path)

    # -------------------------------------------------------------------------
    # 2. Identify Remaining 25%
    # -------------------------------------------------------------------------
    # Merge on (var1, var2) to see which rows exist only in the full data
    # (i.e., the leftover 25%)
    print("[run_rf_example] Determining leftover 25% data...")
    merged = full_data.merge(sampled_data, on=['var1', 'var2'], how='left', indicator=True)
    remaining_25_data = merged[merged['_merge'] == 'left_only']
    remaining_25_time = remaining_25_data['var1'].values
    remaining_25_volume = remaining_25_data['var2'].values

    # -------------------------------------------------------------------------
    # 3. Prepare Data for Random Forest
    # -------------------------------------------------------------------------
    print("[run_rf_example] Preparing training and test sets for RF...")
    # 75% data for training
    X_train = sampled_data['var1'].values.reshape(-1, 1)
    y_train = sampled_data['var2'].values
    # 25% data for inference
    X_test = remaining_25_time.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # 4. Train Random Forest Regressor
    # -------------------------------------------------------------------------
    print("[run_rf_example] Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 5. Predict on the remaining 25%
    # -------------------------------------------------------------------------
    print("[run_rf_example] Predicting on leftover 25% data using RF...")
    rf_predictions = rf_model.predict(X_test)

    # -------------------------------------------------------------------------
    # 6. Plot the Results
    # -------------------------------------------------------------------------
    print("[run_rf_example] Generating plot...")
    plt.figure(figsize=(10, 6))

    # Plot 75% (training) data
    plt.scatter(sampled_data['var1'], sampled_data['var2'], color='orange',
                label='75% Training Data (Orange Circles)', alpha=0.7)

    # Plot leftover 25% data
    plt.scatter(remaining_25_time, remaining_25_volume, color='green', marker='^',
                label='Remaining 25% Data (Green Triangles)', alpha=0.7)

    # Plot RF predictions
    plt.scatter(remaining_25_time, rf_predictions, color='red', marker='x',
                label='RF Predictions (Red Crosses)', alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Random Forest Predictions on Remaining 25% (Qi100)')
    plt.legend()
    plt.grid(True)

    plt.savefig(results_plot_path, dpi=300)
    print(f"[run_rf_example] Plot saved to {results_plot_path}")
    plt.show()

    print("[run_rf_example] Done. RF example completed.")

if __name__ == "__main__":
    run_rf_example()