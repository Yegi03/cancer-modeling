"""
src/integration/integration.py

This script demonstrates a PDE-ODE integration for a model, using a set of parameters.
It loads two datasets: a full one (e.g., Qi100) and a 75% subset. It then simulates the
model, updates predictions via simple data assimilation, and compares them to the 25% leftover data.

Note:
- By default, this file uses Qi100 data and parameters. If you want to switch to Qi70,
  you can uncomment the Qi70 section and comment out the Qi100 references.

"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def main():
    """
    Main workflow for:
      1. Loading data (full + 75%).
      2. Defining PDE-ODE system parameters.
      3. Solving the system with solve_ivp (method='BDF').
      4. Doing a simple 'data assimilation' approach on every 5th data point.
      5. Plotting results and highlighting the 25% leftover data.
    """

    # -------------------------------------------------------------------------
    # 1. Load the Data
    # -------------------------------------------------------------------------
    # Paths for Qi100 (default)
    full_data_path = "../../datasets/Qi100/Qi100.xlsx"
    sampled_data_path = "../../datasets/Qi100/75_qi100.xlsx"

    # -------------------------------------------------------------------------
    # If you want Qi70, uncomment these lines and comment out the Qi100 lines:
    # full_data_path = "../../datasets/Qi70/Qi70.xlsx"
    # sampled_data_path = "../../datasets/Qi70/75_qi70.xlsx"
    # -------------------------------------------------------------------------

    print("[main] Loading sampled (75%) data and full data...")

    # Load the 75% (sampled) dataset
    sampled_data = pd.read_excel(sampled_data_path)
    sampled_time = sampled_data['var1'].values
    sampled_volume = sampled_data['var2'].values

    # Load the full 100% dataset
    full_data = pd.read_excel(full_data_path)
    time_points = full_data['var1'].values
    actual_volumes = full_data['var2'].values

    # -------------------------------------------------------------------------
    # 2. Define the PDE-ODE Model
    # -------------------------------------------------------------------------
    def model(t, y, params):
        """
        The PDE-ODE system in an ODE-friendly format.
        y: [C, Cb, Cbs, Ci, T]
        params: (D, epsilon, k_a, k_d, k_i, a, K, alpha_1, alpha_2,
                 beta_1, beta_2, k_1, k_2, c, K_T).
        Returns the time derivatives for each state variable.
        """
        C, Cb, Cbs, Ci, T = y
        (D, epsilon, k_a, k_d, k_i, a, K, alpha_1, alpha_2,
         beta_1, beta_2, k_1, k_2, c, K_T) = params

        # dC/dt, dCb/dt, dCbs/dt, dCi/dt, dT/dt
        dCdt = D * (C / epsilon) - (k_a * Cbs * C / epsilon) + (k_d * Cb)
        dCbdt = (k_a * Cbs * C / epsilon) - (k_d * Cb) - (k_i * Cb)
        dCbsdt = (-k_a * Cbs * C / epsilon
                  + k_d * Cb
                  + k_i * Cb
                  + a * Cbs * (1 - Cbs / K)
                  - alpha_1 * Cbs * T)
        dCidt = (k_i * Cb
                 + Cbs * T
                 - beta_1 * Ci * T
                 - k_2 * C * Ci)
        dTdt = (c * T * (1 - T / K_T)
                - alpha_2 * Cbs * T
                - beta_2 * Ci * T
                - k_1 * C * T)
        return [dCdt, dCbdt, dCbsdt, dCidt, dTdt]

    # -------------------------------------------------------------------------
    # 3. Define Model Parameters (Example Qi100 values)
    # -------------------------------------------------------------------------
    # If you switch to Qi70, replace these with Qi70 param values and comment these out.
    params = [
        0.02003298, 0.11736967, 0.02423186, 0.02888676,
        0.03001929, 0.02581725, 0.12067404, 0.1239408,
        0.11591689, 0.11797582, 0.12631316, 0.11355364,
        0.11801597, 0.11291372, 0.11694338
    ]

    print("[main] Model parameters (for Qi100):")
    print(params)

    # Extract them as needed:
    # D, epsilon, k_a, k_d, k_i, a, K, alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T = params

    # -------------------------------------------------------------------------
    # 4. Solve the PDE-ODE system
    # -------------------------------------------------------------------------
    # Use the first volume as an initial condition for C, and some placeholders for others
    initial_conditions = [actual_volumes[0], 4, -2.5, 0, 40]

    print("[main] Solving PDE/ODE system with solve_ivp (method='BDF')...")
    solution_bdf = solve_ivp(
        fun=lambda t, y: model(t, y, params),
        t_span=(time_points[0], time_points[-1]),
        y0=initial_conditions,
        t_eval=time_points,
        method='BDF',
        atol=1e-6,
        rtol=1e-6
    )

    # Extract the model's predicted volume from the first state variable
    model_predictions = solution_bdf.y[0]  # C = y[0,:]

    # -------------------------------------------------------------------------
    # 5. Simple Data Assimilation
    # -------------------------------------------------------------------------
    # Assume new measurements are available every 5th data point
    print("[main] Performing a basic assimilation on every 5th data point...")
    measurements = actual_volumes[::5]
    measurement_times = time_points[::5]

    # Interpolate model predictions to the same measurement times
    predicted_volumes = np.interp(measurement_times, solution_bdf.t, model_predictions)
    # Combine model predictions and actual measurements by averaging
    updated_predictions = (predicted_volumes + measurements) / 2

    # -------------------------------------------------------------------------
    # 6. Identify the Remaining 25% Data
    # -------------------------------------------------------------------------
    # Merge the full dataset with the 75% subset to find leftover 25%
    leftover_merge = full_data.merge(sampled_data, on=['var1', 'var2'], how='left', indicator=True)
    leftover_data = leftover_merge[leftover_merge['_merge'] == 'left_only']
    leftover_time = leftover_data['var1'].values
    leftover_volume = leftover_data['var2'].values

    # -------------------------------------------------------------------------
    # 7. Plot Results
    # -------------------------------------------------------------------------
    print("[main] Plotting results...")
    plt.figure(figsize=(12, 6))

    # 7a. Original model predictions (blue line)
    plt.plot(time_points, model_predictions, 'b-', label='Simulated Original Data (Blue Line)')

    # 7b. Data assimilation updates (plot as X markers)
    plt.plot(measurement_times, updated_predictions, 'x', color='green',
             label='Assimilated Data (Green X)')

    # 7c. 75% sampled data (dark orange circles)
    plt.scatter(sampled_time, sampled_volume, color='#FF8C00',
                label='75% Sampled Data (Dark Orange Circles)', alpha=0.7)

    # 7d. Leftover 25% (yellow triangles)
    plt.scatter(leftover_time, leftover_volume, color='yellow', marker='^',
                label='Remaining 25% Data (Yellow Triangles)', alpha=0.7)

    # Plot labeling
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Integration and Simple Data Assimilation')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("[main] Done. Integration example completed.")

if __name__ == "__main__":
    main()