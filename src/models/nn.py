"""
src/models/nn.py

This script trains a simple feedforward Neural Network (NN) to predict
the remaining 25% of data, using 75% of the dataset for training.
It scales the inputs and outputs, trains a Keras/TensorFlow model,
and evaluates predictions on the leftover data portion.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn utilities for scaling and metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# TensorFlow / Keras for building the Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def run_nn_example():
    """
    Demonstrates a feedforward Neural Network approach to predict the
    leftover 25% of data given 75% training data. It scales inputs/outputs,
    trains a simple NN, then evaluates and plots predictions.
    """
    # -------------------------------------------------------------------------
    # 1. Load Datasets
    # -------------------------------------------------------------------------
    # Adjust paths according to your folder structure
    full_data_path = "../../datasets/Qi100/Qi100.xlsx"      # 100% data
    sampled_data_path = "../../datasets/Qi100/75_qi100.xlsx" # 75% data
    results_plot = "../../result/nn_qi100_plot.png"          # Where to save plot

    print("[run_nn_example] Loading full and sampled datasets...")
    full_data = pd.read_excel(full_data_path)
    sampled_data = pd.read_excel(sampled_data_path)

    # -------------------------------------------------------------------------
    # 2. Identify the 25% Remaining Data
    # -------------------------------------------------------------------------
    print("[run_nn_example] Identifying remaining 25% data...")
    merged = full_data.merge(sampled_data, on=['var1', 'var2'], how='left', indicator=True)
    remaining_25_data = merged[merged['_merge'] == 'left_only']
    remaining_25_time = remaining_25_data['var1'].values
    remaining_25_volume = remaining_25_data['var2'].values

    # -------------------------------------------------------------------------
    # 3. Prepare Data for NN
    # -------------------------------------------------------------------------
    print("[run_nn_example] Preparing data for NN training (75%) and inference (25%)...")
    # X_train: times from the 75% dataset
    # y_train: volumes from the 75% dataset
    # X_test: times for the 25% leftover
    X_train = sampled_data['var1'].values.reshape(-1, 1)
    y_train = sampled_data['var2'].values
    X_test = remaining_25_time.reshape(-1, 1)
    y_test = remaining_25_volume  # for reporting MSE on leftover data

    # -------------------------------------------------------------------------
    # 4. Scale the data
    # -------------------------------------------------------------------------
    print("[run_nn_example] Scaling input (time) and output (volume) features...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # y must be reshaped for the scaler
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # -------------------------------------------------------------------------
    # 5. Define and Create the Neural Network
    # -------------------------------------------------------------------------
    print("[run_nn_example] Building Neural Network model...")
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(1,)),
        Dense(64, activation='relu'),
        Dense(1)  # single continuous output
    ])

    nn_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

    # -------------------------------------------------------------------------
    # 6. Train the NN
    # -------------------------------------------------------------------------
    print("[run_nn_example] Training NN (scaled data)...")
    nn_model.fit(X_train_scaled, y_train_scaled, epochs=100, verbose=1)

    # -------------------------------------------------------------------------
    # 7. Predict on the remaining 25%
    # -------------------------------------------------------------------------
    print("[run_nn_example] Predicting leftover 25% data with NN...")
    nn_predictions_scaled = nn_model.predict(X_test_scaled).flatten()
    # Inverse scale predictions back to original "volume" domain
    nn_predictions = scaler_y.inverse_transform(nn_predictions_scaled.reshape(-1, 1)).flatten()

    # -------------------------------------------------------------------------
    # 8. Evaluate and Plot
    # -------------------------------------------------------------------------
    print("[run_nn_example] Calculating MSE on leftover data and generating plot...")
    mse = mean_squared_error(y_test, nn_predictions)
    rmse = np.sqrt(mse)
    print(f"NN Mean Squared Error (MSE): {mse:.4f}")
    print(f"NN Root Mean Squared Error (RMSE): {rmse:.4f}")

    plt.figure(figsize=(10, 6))
    # 75% (Training) data
    plt.scatter(X_train, y_train, color='orange', label='75% Data (Orange Circles)', alpha=0.7)

    # 25% (Remaining) data
    plt.scatter(remaining_25_time, remaining_25_volume, color='green', marker='^',
                label='Remaining 25% Data (Green Triangles)', alpha=0.7)

    # NN Predictions
    plt.scatter(remaining_25_time, nn_predictions, color='red', marker='x',
                label='NN Predictions (Red Crosses)', alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Neural Network Predictions on Remaining 25% (Qi100)')
    plt.legend()
    plt.grid(True)

    plt.savefig(results_plot, dpi=300)
    print(f"[run_nn_example] Plot saved to {results_plot}")
    plt.show()

    print("[run_nn_example] Done. NN example completed.")

if __name__ == "__main__":
    run_nn_example()