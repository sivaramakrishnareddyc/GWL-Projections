import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Define the classes
classes = ['annual', 'mixed', 'inertial']

# Define the metrics with full names for better readability
metric_names = {
    'NSE': 'Nash-Sutcliffe Efficiency',
    'R2': 'R-squared',
    'MAE': 'Mean Absolute Error',
    'KGE': 'Kling-Gupta Efficiency',
    'NRMSE': 'Normalized Root Mean Square Error',
    'RMSE': 'Root Mean Square Error'
}

# Define the metrics
metrics = list(metric_names.keys())

for metric in metrics:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)

    min_value = float('inf')
    max_value = float('-inf')

    for j, Class in enumerate(classes):
        # Read the data for the specific class, model (GRU), and metric
        filename = f"/media/chidesiv/DATA2/Final_phase/Final_plots/Metrics_train_val/Cdf_plots_metrics/scores_era5GRUla8{Class}.csv"
        df = pd.read_csv(filename)

        # Filter the data for the specific class and metric
        metric_values_tr = df[df['Class'] == Class][f'{metric}_tr']
        metric_values_te = df[df['Class'] == Class][f'{metric}_te']

        # Update min and max values
        min_value = min(min_value, metric_values_tr.min(), metric_values_te.min())
        max_value = max(max_value, metric_values_tr.max(), metric_values_te.max())

        # Calculate CDF for training
        cdf_tr = stats.cumfreq(metric_values_tr, numbins=100)
        x_tr = cdf_tr.lowerlimit + np.linspace(0, cdf_tr.binsize * cdf_tr.cumcount.size, cdf_tr.cumcount.size)
        y_tr = cdf_tr.cumcount / len(metric_values_tr)

        # Calculate CDF for testing
        cdf_te = stats.cumfreq(metric_values_te, numbins=100)
        x_te = cdf_te.lowerlimit + np.linspace(0, cdf_te.binsize * cdf_te.cumcount.size, cdf_te.cumcount.size)
        y_te = cdf_te.cumcount / len(metric_values_te)

        # Plot training with solid line
        axs[j].plot(x_tr, y_tr, label='Training', color='blue', linestyle='-')
        # Plot testing with dashed line
        axs[j].plot(x_te, y_te, label='Testing', color='red', linestyle='--')

        # Set title for the subplot with increased font size and full metric name
        axs[j].set_title(f'ERA5 - {Class} ({metric_names[metric]})', fontsize=16)

        # Set y-axis label for all subplots with increased font size
        axs[j].set_ylabel('CDF', fontsize=14)

        # Display legend for all subplots with increased font size
        axs[j].legend(fontsize=12)

        # Increase the font size of ticks in both axes
        axs[j].tick_params(axis='both', which='major', labelsize=14)

        axs[j].grid(True)

    # Set x-axis limits based on min and max values
    padding = (max_value - min_value) * 0.1  # 10% padding
    for ax in axs:
        ax.set_xlim(min_value - padding, max_value + padding)

    # Set x-axis label for the last subplot with increased font size
    axs[-1].set_xlabel(f'{metric_names[metric]} Value', fontsize=14)

    # Set an overall title for the entire figure
    fig.suptitle(f'GRU Model - {metric_names[metric]} Comparison', fontsize=18, y=1.02)

    plt.tight_layout()

    # Save the plot with increased font size
    plt.savefig(f'Comp_{metric.upper()}_GRU_ERA5.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    plt.close()