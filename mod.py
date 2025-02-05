import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from mpi4py import MPI

from Datasets.data import Dataset
from Models.models import train_old_model, train_new_models
from Models.lct import LocalCorrectionTree
from utils import plot_metrics_comparison

# Load and prepare data
def load_prep_data(dataset_name):
    if dataset_name in ['Adult', 'Bank']:
        dataset = Dataset(dataset_name)
        X, y, feature_names, x = dataset.load_data()
        X_old, y_old, X_new, y_new = dataset.split_old_new_data(X, y)
        return X_old, y_old, X_new, y_new, feature_names, x
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
def process_model(dataset, model):
    print(f"\nProcessing {dataset} dataset for {model} model")

    # 1. Load data
    if dataset == '7-point':
        X_old, y_old, X_new, y_new, X_clinical = load_prep_data(dataset)
    else:
        X_old, y_old, X_new, y_new, feature_names, x = load_prep_data(dataset)

    X_new, X_old = np.array(X_new), np.array(X_old)
    y_new, y_old = np.array(y_new), np.array(y_old)

    # 2. Train base models on old data
    base_models_dict = train_old_model(dataset, X_new, X_old, y_new, y_old)

    # 3. Initialize metrics dictionaries for both base and corrected models
    metrics = {
        'Accuracy': {model: [], f'{model}_LCT': []},
        'Precision': {model: [], f'{model}_LCT': []},
        'Recall': {model: [], f'{model}_LCT': []},
        'F1': {model: [], f'{model}_LCT': []}
    }

    # 4. Perform 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_new, y_new)):
        print(f"Processing fold {fold + 1}/5")

        X_train, X_test = X_new[train_idx], X_new[test_idx]
        y_train, y_test = y_new[train_idx], y_new[test_idx]

        # Train LCT-corrected models on fold training data
        fold_corrected = train_new_models(dataset, X_train, y_train, base_models_dict, LocalCorrectionTree)

        # Evaluate both base and corrected models
        # Evaluate base model
        if model == 'DNN':
            base_preds = base_models_dict[model].predict(X_test)
            final_preds_base = base_preds
        else:
            base_probs = base_models_dict[model].predict_proba(X_test)
            final_preds_base = np.argmax(base_probs, axis=1)

        # Calculate base model metrics
        metrics['Accuracy'][model].append(accuracy_score(y_test, final_preds_base))
        metrics['Precision'][model].append(precision_score(y_test, final_preds_base, average='macro'))
        metrics['Recall'][model].append(recall_score(y_test, final_preds_base, average='macro'))
        metrics['F1'][model].append(f1_score(y_test, final_preds_base, average='macro'))

        # Evaluate LCT-corrected model
        lct_model = fold_corrected[f'{model}_LCT']
        corrections = lct_model.predict(X_test)
        if model != 'DNN':
            base_probs = base_models_dict[model].predict_proba(X_test)
            corrected_probs = base_probs + corrections
            final_preds_lct = np.argmax(corrected_probs, axis=1)
        else:
            # Handle DNN case for LCT if necessary
            pass

        # Calculate LCT model metrics
        metrics['Accuracy'][f'{model}_LCT'].append(accuracy_score(y_test, final_preds_lct))
        metrics['Precision'][f'{model}_LCT'].append(precision_score(y_test, final_preds_lct, average='macro'))
        metrics['Recall'][f'{model}_LCT'].append(recall_score(y_test, final_preds_lct, average='macro'))
        metrics['F1'][f'{model}_LCT'].append(f1_score(y_test, final_preds_lct, average='macro'))

    return metrics

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    datasets = ['Adult', 'Bank']
    base_models = ['DT', 'RF', 'LGBM']

    # Distribute models across processes
    num_models = len(base_models) * len(datasets)
    start_idx = rank * num_models // size
    end_idx = (rank + 1) * num_models // size

    model_results = {}
    for i in range(start_idx, end_idx):
        dataset_idx = i // len(base_models)
        model_idx = i % len(base_models)
        dataset = datasets[dataset_idx]
        model = base_models[model_idx]

        metrics = process_model(dataset, model)
        if dataset not in model_results:
            model_results[dataset] = {}
        model_results[dataset][model] = metrics

    # Gather results from all processes
    gathered_results = comm.gather(model_results, root=0)

    if rank == 0:
        combined_results = {}
        for results in gathered_results:
            for dataset, models in results.items():
                if dataset not in combined_results:
                    combined_results[dataset] = {}
                for model, metrics in models.items():
                    if model not in combined_results[dataset]:
                        combined_results[dataset][model] = metrics
                    else:
                        for metric, values in metrics.items():
                            for key, val_list in values.items():
                                combined_results[dataset][model][metric][key].extend(val_list)

        # Print and plot results
        for dataset, models in combined_results.items():
            print(f"\nCombined Model Results for {dataset}:")
            for model, metrics in models.items():
                for metric, values in metrics.items():
                    print(f"\n{metric}:")
                    for key, val_list in values.items():
                        mean_val = np.mean(val_list)
                        std_val = np.std(val_list)
                        print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

            # Prepare metrics for comparison plot
            plot_metrics = {
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1': []
            }

            all_models = base_models + [f'{m}_LCT' for m in base_models]
            for metric in plot_metrics:
                for m in all_models:
                    if m in [f'{model}_LCT' for model in base_models]:
                        model_name = m[:-4]
                    else:
                        model_name = m
                    if model_name in models:
                        if m in models[model_name][metric]:
                            plot_metrics[metric].append(models[model_name][metric][m])

            # Create and save comparison plot
            fig = plot_metrics_comparison(dataset, plot_metrics, all_models)
            plt.savefig(f'{dataset}_combined_comparison.png', bbox_inches='tight', dpi=400)
            plt.close()

        print("\nDone evaluating models.")

if __name__ == "__main__":
    main()
