import numpy as np
import matplotlib.pyplot as plt


def plot_metrics_comparison(dataset, metrics, model_names):
    """
    Create comparison bar plots for model metrics.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # Define model colors and markers
    colors = {
        'L1-LR': '#1f77b4',
        'L1-LR+': '#1f77b4',
        'L2-LR': '#ff7f0e',
        'L2-LR+': '#ff7f0e',
        'DT': '#2ca02c',
        'DT+': '#2ca02c',
        'RF': '#8c564b',
        'RF+': '#8c564b',
        'LGBM': '#9467bd',
        'LGBM+': '#9467bd',
        'LGBM+C': '#9467bd',
        'Ours': '#d62728',
        'Old': '#7f7f7f'
    }
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        y_pos = np.arange(len(model_names))
        values = metrics[metric]
        
        # Plot bar plots
        for i, (name, value) in enumerate(zip(model_names, values)):
            color = colors.get(name)
            mean_val = np.mean(value)
            std_val = np.std(value)
            
            # Plot bar
            ax.bar(i, mean_val, color=color, alpha=0.7, width=0.8)
            
            # Plot error bars
            ax.errorbar(i, mean_val, yerr=std_val, fmt='none', capsize=3, color='black', elinewidth=1, capthick=1)
        
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
        ax.set_xticks(y_pos)
        ax.set_xticklabels(model_names, rotation=90)
        ax.set_title(f'({chr(97+idx)}) {metric} ({dataset})')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        min_val = min([np.mean(v) for v in values]) - 0.05
        max_val = max([np.mean(v) for v in values]) + 0.05
        ax.set_ylim(min_val, max_val)
    
    plt.tight_layout()
    return fig