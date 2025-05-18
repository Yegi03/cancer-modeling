import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
from scipy.integrate import solve_ivp

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['figure.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.dpi'] = 300

def load_growth_data():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cancer-modeling--main', 'datasets')
    qi70 = pd.read_excel(os.path.join(base, 'Qi70', 'Qi70.xlsx'))
    qi100 = pd.read_excel(os.path.join(base, 'Qi100', 'Qi100.xlsx'))
    return qi70, qi100

def load_mcmc_params():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cancer-modeling--main', 'datasets')
    params70 = pd.read_excel(os.path.join(base, 'Qi70', 'estimated_params_qi70.xlsx'))
    params100 = pd.read_excel(os.path.join(base, 'Qi100', 'estimated_params_qi100.xlsx'))
    return params70, params100

def plot_math_bayes_figure():
    # Load data
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cancer-modeling--main', 'datasets')
    qi70 = pd.read_excel(os.path.join(base, 'Qi70', 'Qi70.xlsx'))
    qi100 = pd.read_excel(os.path.join(base, 'Qi100', 'Qi100.xlsx'))
    train70 = pd.read_excel(os.path.join(base, 'Qi70', '75_qi70.xlsx'))
    train100 = pd.read_excel(os.path.join(base, 'Qi100', '75_qi100.xlsx'))
    params70 = pd.read_excel(os.path.join(base, 'Qi70', 'estimated_params_qi70.xlsx'))
    params100 = pd.read_excel(os.path.join(base, 'Qi100', 'estimated_params_qi100.xlsx'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Panel A: Tumor growth curves (100% data)
    axes[0,0].plot(qi70['var1'], qi70['var2'], 'o-', label='Qi70 (100%)')
    axes[0,0].plot(qi100['var1'], qi100['var2'], 's-', label='Qi100 (100%)')
    axes[0,0].set_xlabel('Time (days)')
    axes[0,0].set_ylabel('Tumor Volume (mm³)')
    axes[0,0].set_title('A: Tumor Growth Curves (100% Data)')
    axes[0,0].legend()

    # Panel B: Bar plot of all best-fit parameter values
    param_names = params70.columns.tolist()
    x = np.arange(len(param_names))
    width = 0.35
    axes[0,1].bar(x - width/2, params70.iloc[0].values, width, label='Qi70')
    axes[0,1].bar(x + width/2, params100.iloc[0].values, width, label='Qi100')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(param_names, rotation=90, fontsize=7)
    axes[0,1].set_title('B: Best-fit Parameter Comparison')
    axes[0,1].legend()

    # Panel C: Model fit comparison (100% data, 75% train, 25% test, model prediction)
    # For now, just overlay 100% data, 75% train, and 25% test
    # (If you have model predictions, you can add them here)
    # For Qi70
    all_times = qi70['var1'].values
    all_vols = qi70['var2'].values
    train_times = train70['var1'].values
    train_vols = train70['var2'].values
    test_mask = ~np.isin(all_times, train_times)
    test_times = all_times[test_mask]
    test_vols = all_vols[test_mask]
    axes[1,0].plot(all_times, all_vols, 'o-', color='gray', label='Qi70 All Data (100%)', alpha=0.5)
    axes[1,0].scatter(train_times, train_vols, color='tab:blue', label='Qi70 Train (75%)')
    axes[1,0].scatter(test_times, test_vols, color='tab:orange', label='Qi70 Test (25%)')
    # For Qi100
    all_times2 = qi100['var1'].values
    all_vols2 = qi100['var2'].values
    train_times2 = train100['var1'].values
    train_vols2 = train100['var2'].values
    test_mask2 = ~np.isin(all_times2, train_times2)
    test_times2 = all_times2[test_mask2]
    test_vols2 = all_vols2[test_mask2]
    axes[1,0].plot(all_times2, all_vols2, 's-', color='lightgray', label='Qi100 All Data (100%)', alpha=0.5)
    axes[1,0].scatter(train_times2, train_vols2, color='tab:green', label='Qi100 Train (75%)')
    axes[1,0].scatter(test_times2, test_vols2, color='tab:red', label='Qi100 Test (25%)')
    axes[1,0].set_xlabel('Time (days)')
    axes[1,0].set_ylabel('Tumor Volume (mm³)')
    axes[1,0].set_title('C: Data Split and Comparison')
    axes[1,0].legend(fontsize=7)

    # Panel D: Table of parameter values (as a plot)
    cell_text = []
    for i in range(len(param_names)):
        cell_text.append([f'{params70.iloc[0,i]:.3g}', f'{params100.iloc[0,i]:.3g}'])
    axes[1,1].axis('off')
    table = axes[1,1].table(cellText=cell_text,
                            rowLabels=param_names,
                            colLabels=['Qi70', 'Qi100'],
                            loc='center',
                            cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    axes[1,1].set_title('D: Parameter Table')

    plt.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), '../figures')
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, 'math_bayes_multi_panel.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_split_comparison(dataset_name, all_data, train_data, outdir):
    all_times = all_data['var1'].values
    all_vols = all_data['var2'].values
    train_times = train_data['var1'].values
    train_vols = train_data['var2'].values
    test_mask = ~np.isin(all_times, train_times)
    test_times = all_times[test_mask]
    test_vols = all_vols[test_mask]

    plt.figure(figsize=(7,5))
    # 100% data: dark, filled circles (no line)
    plt.scatter(all_times, all_vols, marker='o', color='black', alpha=0.85, label=f'{dataset_name} All Data (100%)', s=45)
    # 75% training: empty squares
    plt.scatter(train_times, train_vols, marker='s', facecolors='none', edgecolors='tab:blue', s=70, label=f'{dataset_name} Train (75%)', linewidths=1.5)
    # 25% test: crosses
    plt.scatter(test_times, test_vols, marker='x', color='tab:red', s=70, label=f'{dataset_name} Test (25%)', linewidths=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Tumor Volume (mm³)')
    plt.title(f'{dataset_name} Data Split Visualization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{dataset_name.lower()}_split_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_split_comparisons():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cancer-modeling--main', 'datasets')
    outdir = os.path.join(os.path.dirname(__file__), '../figures')
    os.makedirs(outdir, exist_ok=True)
    # Qi70
    qi70 = pd.read_excel(os.path.join(base, 'Qi70', 'Qi70.xlsx'))
    train70 = pd.read_excel(os.path.join(base, 'Qi70', '75_qi70.xlsx'))
    plot_split_comparison('Qi70', qi70, train70, outdir)
    # Qi100
    qi100 = pd.read_excel(os.path.join(base, 'Qi100', 'Qi100.xlsx'))
    train100 = pd.read_excel(os.path.join(base, 'Qi100', '75_qi100.xlsx'))
    plot_split_comparison('Qi100', qi100, train100, outdir)

def simulate_model(times, params):
    # This is a simplified version using your ODE/PDE structure for demonstration
    # You can replace this with your full system if needed
    # For now, let's use a simple logistic growth as a placeholder
    # params: [D, epsilon, k_a, C_bs_init, k_d, k_i, a, K, alpha_1, alpha_2, beta_1, beta_2, k_1, k_2, c, K_T]
    c, K_T = params[-2], params[-1]
    def logistic(t, y):
        return c * y * (1 - y / K_T)
    y0 = [40]  # initial tumor volume (can be replaced with your initial condition)
    sol = solve_ivp(logistic, [times[0], times[-1]], y0, t_eval=times, method='RK45')
    return sol.y[0]

def plot_model_fit(dataset_name, all_data, train_data, param_file, outdir):
    all_times = all_data['var1'].values
    all_vols = all_data['var2'].values
    train_times = train_data['var1'].values
    train_vols = train_data['var2'].values
    test_mask = ~np.isin(all_times, train_times)
    test_times = all_times[test_mask]
    test_vols = all_vols[test_mask]
    # Load estimated parameters
    params = pd.read_excel(param_file).iloc[0].values
    # Simulate model prediction
    model_pred = simulate_model(all_times, params)
    # Model prediction at test points
    model_pred_test = model_pred[test_mask]
    plt.figure(figsize=(7,5))
    # Actual data (100%): circles
    plt.scatter(all_times, all_vols, marker='o', color='black', alpha=0.7, label=f'{dataset_name} Data (100%)', s=45)
    # Model prediction: line
    plt.plot(all_times, model_pred, color='tab:blue', label='Model Prediction', linewidth=2)
    # 25% test: actual data as red crosses
    plt.scatter(test_times, test_vols, marker='x', color='tab:red', s=70, label=f'{dataset_name} Test Data (25%)', linewidths=2)
    # 25% test: model prediction as blue crosses
    plt.scatter(test_times, model_pred_test, marker='x', color='tab:blue', s=70, label=f'{dataset_name} Model Prediction (Test Points)', linewidths=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Tumor Volume (mm³)')
    plt.title(f'{dataset_name} Model Fit and Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{dataset_name.lower()}_model_fit.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_param_bar(params70, params100, outdir):
    param_names = params70.columns.tolist()
    x = np.arange(len(param_names))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, params70.iloc[0].values, width, label='Qi70')
    plt.bar(x + width/2, params100.iloc[0].values, width, label='Qi100')
    plt.xticks(x, param_names, rotation=90, fontsize=7)
    plt.title('Best-fit Parameter Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'param_bar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_model_fits_and_params():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cancer-modeling--main', 'datasets')
    outdir = os.path.join(os.path.dirname(__file__), '../figures')
    os.makedirs(outdir, exist_ok=True)
    # Qi70
    qi70 = pd.read_excel(os.path.join(base, 'Qi70', 'Qi70.xlsx'))
    train70 = pd.read_excel(os.path.join(base, 'Qi70', '75_qi70.xlsx'))
    param70_file = os.path.join(base, 'Qi70', 'estimated_params_qi70.xlsx')
    plot_model_fit('Qi70', qi70, train70, param70_file, outdir)
    # Qi100
    qi100 = pd.read_excel(os.path.join(base, 'Qi100', 'Qi100.xlsx'))
    train100 = pd.read_excel(os.path.join(base, 'Qi100', '75_qi100.xlsx'))
    param100_file = os.path.join(base, 'Qi100', 'estimated_params_qi100.xlsx')
    plot_model_fit('Qi100', qi100, train100, param100_file, outdir)
    # Bar plot of parameters
    params70 = pd.read_excel(param70_file)
    params100 = pd.read_excel(param100_file)
    plot_param_bar(params70, params100, outdir)

if __name__ == '__main__':
    plot_math_bayes_figure()
    plot_all_split_comparisons()
    plot_all_model_fits_and_params() 