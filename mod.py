import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Models.model import train_lct, train_old_model, train_base_competitor_models,  train_competitor_standard_corrected
from Models.lct import LocalCorrectionTree

from utils import analyze_and_save_corrections

from Datasets.data import Dataset

def evaluate_all_methods(
    dataset_name, 
    X_new,
    X_old,
    y_new,
    y_old,
    feature_names,
    localcorrectiontree_cls,
    random_state=42
):
    """
    Splits X,y into 5 folds. In each fold:
      1) Create a “train” (fold_train_idx) and “test” (fold_test_idx).
      2) Train or fine-tune models on the train set (with any hyperparameter
         tuning using that train set only).
      3) Evaluate on test.
    Accumulate test metrics for each method across folds. 
    Finally, produce boxplots for accuracy, precision, recall, F1. 
    """
    
    print("=== Training old model on X_old, y_old ===")
    old_model = train_old_model(dataset_name, X_old, y_old, model_type='LGBM')
    old_models_dict = {"LGBM": old_model}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # We’ll store results as: results[model_name]['accuracy'] = [fold1, fold2, ...]
    results = {}
    all_models = {}
    
    # (a) old models
    for old_m_name, old_m in old_models_dict.items():
        all_models[f"Old_{old_m_name}"] = old_m
        
    fold_idx = 1
    for train_index, test_index in skf.split(X_new, y_new):
        print(f"\n========= Fold {fold_idx} =========")
        fold_idx += 1
        
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = y_new[train_index], y_new[test_index]
        
        # 2) Train the naive/new-only baselines
        base_models = train_base_competitor_models(dataset_name, X_train, y_train)
        
        # 3) Train standard correction methods
        competitor_corrected = train_competitor_standard_corrected(
            dataset_name, 
            X_train, 
            y_train,
            old_model=old_model,
            base_competitor_models=base_models
        )
        
        # 4) Train LCT to correct each old model
        lct_corrected = train_lct(
            dataset_name, 
            X_train, 
            y_train, 
            old_models_dict, 
            localcorrectiontree_cls
        )
        
        # 5) Evaluate each method on the test set. 
        #    This includes:
        #      - The old model(s)
        #      - The naive baselines (L1-LR, L2-LR, DT, RF, LGBM)
        #      - The standard corrections (L1-LR+, ..., LGBM+C, etc.)
        #      - Our LCT corrections
        
        # (b) naive/new-only
        for m_name, m in base_models.items():
            all_models[m_name] = m
        
        # (c) standard corrections
        for m_name, m in competitor_corrected.items():
            all_models[m_name] = m
        
        # (d) LCT
        for m_name, m in lct_corrected.items():
            all_models[m_name] = m
        
        # Now actually do test predictions
        for model_key, model_obj in all_models.items():
            
            # If model_obj is an LCT, we do old_scores + correction, then argmax
            #if isinstance(model_obj, LocalCorrectionTree) and model_key == "Ours":
            if model_key == "Ours":
                
                model_obj = all_models[model_key]
                #try:
                old_probs = all_models[f"Old_LGBM"].predict_proba(X_test)
                #except:
                #    preds_ = old_models_dict[model_key].predict(X_test)
                #    unique_labels = np.unique(preds_)
                #    n_cls = len(unique_labels)
                #    old_probs = np.zeros((len(X_test), n_cls))
                #    for i, lab in enumerate(unique_labels):
                #        old_probs[preds_ == lab, i] = 1.0
                        
                corrections = model_obj.predict(X_test)  
                corrected_scores = old_probs + corrections
                y_pred = np.argmax(corrected_scores, axis=1)
                
                analyze_and_save_corrections(dataset_name, old_scores=old_probs, new_scores=corrected_scores, 
                                             corrections=corrections, X_test = X_test, y_test = y_test,
                                             feature_names = feature_names, lct = model_obj)
            
            elif model_key in competitor_corrected:
                
                if model_key == "LGBM+C":
                    y_pred = model_obj.predict(X_test)
                else:
                    # Generate features using the original old_model (not corrected model)
                    old_scores_test = old_model.predict_proba(X_test)
                    X_test_concat = np.hstack([X_test, old_scores_test])
                    y_pred = model_obj.predict(X_test_concat)

            else:
                # Normal scikit-learn model
                y_pred = model_obj.predict(X_test)
            
            # Evaluate metrics
            acc = accuracy_score(y_test, y_pred)
            # For binary or multi-class, set average appropriately
            if dataset_name in ['CTG', '7-point']:
                # multi-class => macro avg
                prec = precision_score(y_test, y_pred, average='macro')
                rec = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
            else:
                # binary => standard
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            
            # Store
            if model_key not in results:
                results[model_key] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                }
            results[model_key]['accuracy'].append(acc)
            results[model_key]['precision'].append(prec)
            results[model_key]['recall'].append(rec)
            results[model_key]['f1'].append(f1)
    
    # Once done with all folds, produce boxplots for each metric
    _plot_results_boxplots(results, dataset_name)
    
    return results

def _plot_results_boxplots(results_dict, dataset_name):
    """
    Create side-by-side horizontal box plots (Accuracy, Precision, Recall, F1).
    Each plot shows distribution across all folds for the given model.
    Models are arranged with base versions next to their enhanced counterparts.
    """

    # Convert results to DataFrame
    rows = []
    for model_name, metrics in results_dict.items():
        n_folds = len(metrics['accuracy'])
        for i in range(n_folds):
            rows.append({
                'models':     model_name,
                'accuracy':  metrics['accuracy'][i],
                'precision': metrics['precision'][i],
                'recall':    metrics['recall'][i],
                'f1':        metrics['f1'][i]
            })
            
    df = pd.DataFrame(rows)

    desired_order = [
        "L1-LR", "L1-LR+",
        "L2-LR", "L2-LR+",
        "DT", "DT+",
        "RF", "RF+",
        "LGBM", "LGBM+", "LGBM+C",
        "Ours", "Old_LGBM"
    ]
    
    # Filter and order models
    model_order = [m for m in desired_order if m in df['models'].unique()]

    # Style configuration
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=0.8)
    fig, axes = plt.subplots(1, 4, figsize=(10, 4), sharey=True)

    # Color palette (same as before for consistency)
    palette = {
        "L1-LR": "blue", "L1-LR+": "blue",
        "L2-LR": "orange", "L2-LR+": "orange",
        "DT": "green", "DT+": "green",
        "RF": "yellow", "RF+": "yellow",
        "LGBM": "purple", "LGBM+": "purple",
        "LGBM+C": "brown",
        "Ours": "pink",
        "Old_LGBM": "gray"
    }

    def plot_metric(ax, metric_col, title_str):
        sns.boxplot(
            data=df,
            x=metric_col,
            y='models',
            order=model_order,
            ax=ax,
            orient='h',
            palette=palette,
            width=0.3,
            whis=1.5  # IQR multiplier for whisker length
        )
        ax.set_title(f"{title_str} ({dataset_name})")
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.5)

    # Create plots for each metric
    plot_metric(axes[0], 'accuracy',  'Accuracy')
    plot_metric(axes[1], 'precision', 'Precision')
    plot_metric(axes[2], 'recall',    'Recall')
    plot_metric(axes[3], 'f1',        'F1')
    
    plt.tight_layout(pad=1.0)
    plt.savefig(f"Comparison_Boxplots_{dataset_name}.png")

###############################################################################
#   USAGE EXAMPLE (MPI version)
###############################################################################
if __name__ == "__main__":
    for dataset_name in ['Adult', 'Bank']:
        dataset = Dataset(dataset_name=dataset_name)
        
        # Load your data
        X, y, feature_names, x = dataset.load_data()
        X_old, y_old, X_new, y_new = dataset.split_old_new_data(X, y)
        
        X_old, y_old = np.array(X_old), np.array(y_old)
        X_new, y_new = np.array(X_new), np.array(y_new)
        
        # Call the MPI-based evaluation
        results = evaluate_all_methods(
            dataset_name=dataset_name,
            X_new=X_new, 
            X_old=X_old,
            y_new=y_new,
            y_old=y_old,
            feature_names=feature_names,
            localcorrectiontree_cls=LocalCorrectionTree
        )
