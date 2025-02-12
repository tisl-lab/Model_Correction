import numpy as np
import pandas as pd
import optuna

import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

# Scikit-learn and LightGBM
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score, 
    roc_auc_score
)
from lightgbm import LGBMClassifier
from mpi4py import MPI

# Plotting
import seaborn as sns
sns.set(style="whitegrid")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

seed = 0
###############################################################################
#   A. Parallel Optuna Optimization with MPI
###############################################################################
def parallel_optuna_optimization(objective, n_trials=100, direction='maximize'):
    """
    MPI-parallelized Optuna optimization: splits the total number of trials
    among the available MPI ranks. Each rank runs local_trials. The best
    params from each rank are gathered at the root (rank=0), which picks
    the global best.
    """
    trials_per_proc = n_trials // size
    remaining_trials = n_trials % size

    study = optuna.create_study(direction=direction)
    local_trials = trials_per_proc + (1 if rank < remaining_trials else 0)
    
    if local_trials > 0:
        study.optimize(objective, n_trials=local_trials)

    best_params = comm.gather(study.best_params, root=0)
    best_scores = comm.gather(study.best_value, root=0)

    if rank == 0:
        best_idx = np.argmax(best_scores) if direction == 'maximize' else np.argmin(best_scores)
        return best_params[best_idx]
    return None

###############################################################################
#   B. Old Model Training (LightGBM) on "old data"
###############################################################################
def train_old_model(dataset_name, X_old, y_old, model_type='LGBM', init_model=None):
    if model_type.lower() != 'lgbm':
        raise ValueError("Only 'LGBM' is supported in this demonstration.")
    
    # Decide scoring/cv based on dataset
    if dataset_name in ['Adult', 'Bank']:
        scoring = 'f1'
        cv = 3
    elif dataset_name == 'CTG':
        scoring = 'f1_macro'
        cv = 5
    elif dataset_name == '7-point':
        def macro_recall_scorer(estimator, X, y):
            return recall_score(y, estimator.predict(X), average='macro')
        scoring = macro_recall_scorer
        cv = 5
    else:
        raise ValueError("Unsupported dataset for LightGBM training")
    
    # Parallel objective
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
            'class_weight': 'balanced', 
        }
        model = LGBMClassifier(**params)
        cv_scores = cross_val_score(model, X_old, y_old, cv=cv, scoring=scoring)
        return cv_scores.mean()
    
    best_params = parallel_optuna_optimization(objective, n_trials=100, direction='maximize')

    # Only rank=0 actually finalizes the model
    if rank == 0 and best_params is not None:
        # Final training
        if init_model:
            model = LGBMClassifier(**best_params, init_model=init_model)
        else:
            model = LGBMClassifier(**best_params)
        model.fit(X_old, y_old)
        return model
    return None

###############################################################################
#   C. Local Correction Tree (LCT) Training
###############################################################################
def train_lct(dataset_name, X_new, y_new, old_models, localcorrectiontree_cls):
    """
    For each old_model, we build a Local Correction Tree (LCT).
    We'll store them in a dictionary. 
    """
    corrected_models = {}
    
    # Collect old-model predictions
    old_scores_map = {}
    
    for model_name, model in old_models.items():
        if model is None:
            continue
        # If it has predict_proba, use that
        #try:
        old_scores_map[model_name] = model.predict_proba(X_new)
        #except AttributeError:
            # Or convert from discrete predictions to a 1-hot style
        #    preds = model.predict(X_new)
        #    unique_labels = np.unique(preds)
        #    n_classes = len(unique_labels)
        #    tmp_scores = np.zeros((len(X_new), n_classes))
        #    for i, label in enumerate(unique_labels):
        #        tmp_scores[preds == label, i] = 1.0
        #    old_scores_map[model_name] = tmp_scores
    
    # The objective for a single old_model
    def lct_objective(trial, model_name):
        if dataset_name in ['Adult', 'Bank', 'CTG']:
            lambda_reg = trial.suggest_float("lambda_reg", 0.05, 0.5)
        else:  # e.g. '7-point'
            lambda_reg = trial.suggest_float("lambda_reg", 0.01, 0.1)
        
        # Hard-coded or dataset-based parameters
        max_depth = 5
        nprune = 1
        epsilon_min = 0.01
        epsilon_fail = 0.499
        
        if dataset_name in ['Adult', 'Bank']:
            min_samples_leaf = 64
            epsilon_max = 0.5
        elif dataset_name == 'CTG':
            min_samples_leaf = 8
            epsilon_max = 1.0
        else:
            min_samples_leaf = 16
            epsilon_max = 0.5
        
        lct = localcorrectiontree_cls(
            lambda_reg=lambda_reg,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            nprune=nprune,
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
            epsilon_fail=epsilon_fail
        )
        # Fit LCT
        lct.fit(X_new, y_new, old_scores_map[model_name])
        
        # Evaluate negative accuracy
        corrected_scores = old_scores_map[model_name] + lct.predict(X_new)
        preds = np.argmax(corrected_scores, axis=1)
        return -accuracy_score(y_new, preds)
    
    # For each old model, do an Optuna search in parallel
    for model_name, _ in old_models.items():
        if model_name not in old_scores_map:
            continue
        
        best_params = parallel_optuna_optimization(
            lambda trial: lct_objective(trial, model_name),
            n_trials=100,
            direction='minimize'
        )
        if rank == 0 and best_params is not None:
            # Final re-training with best lambda
            best_lambda = best_params['lambda_reg']
            if dataset_name in ['Adult', 'Bank']:
                min_samples_leaf = 64
                epsilon_max = 0.5
            elif dataset_name == 'CTG':
                min_samples_leaf = 8
                epsilon_max = 1.0
            else:
                min_samples_leaf = 16
                epsilon_max = 0.5
            
            lct_final = localcorrectiontree_cls(
                lambda_reg=best_lambda,
                max_depth=5,
                min_samples_leaf=min_samples_leaf,
                nprune=1,
                epsilon_min=0.01,
                epsilon_max=epsilon_max,
                epsilon_fail=0.499
            )
            lct_final.fit(X_new, y_new, old_scores_map[model_name])
            corrected_models["Ours"] = lct_final
    
    return corrected_models if rank == 0 else None
        
###############################################################################
#   D. Train basic competitor models on new data (L1-LR, L2-LR, DT, RF, LGBM)
###############################################################################
def train_base_competitor_models(dataset_name, X_new, y_new):
    """
    Returns a dict of {model_name: model_object}
    """
    competitor_models = {} if rank == 0 else None

    # L1-LR and L2-LR (no hyperparam search here, for brevity)
    if rank == 0:
        competitor_models['L1-LR'] = LogisticRegressionCV(
            penalty='l1', solver='liblinear'
        ).fit(X_new, y_new)
        competitor_models['L2-LR'] = LogisticRegressionCV(
            penalty='l2'
        ).fit(X_new, y_new)

    # Decision Tree with parallel Optuna
    def dt_objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.0001, 0.1),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1.0),
            'class_weight': 'balanced'
        }
        model = DecisionTreeClassifier(**params)
        if dataset_name in ['Adult', 'Bank']:
            scoring = 'f1'
            cv = 3
        elif dataset_name == 'CTG':
            scoring = 'f1_macro'
            cv = 5
        elif dataset_name == '7-point':
            def macro_recall_scorer(est, X_, y_):
                return recall_score(y_, est.predict(X_), average='macro')
            scoring = macro_recall_scorer
        else:
            scoring = 'accuracy'
        
        return cross_val_score(model, X_new, y_new, cv=cv, scoring=scoring).mean()
    
    best_dt_params = parallel_optuna_optimization(dt_objective, 100, 'maximize')
    if rank == 0 and best_dt_params:
        competitor_models['DT'] = DecisionTreeClassifier(**best_dt_params).fit(X_new, y_new)

    # Random Forest with parallel Optuna
    def rf_objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.0001, 0.1),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1.0),
            'class_weight': 'balanced_subsample'
        }
        model = RandomForestClassifier(**params)
        if dataset_name in ['Adult', 'Bank']:
            scoring = 'f1'
            cv = 3
        elif dataset_name == 'CTG':
            scoring = 'f1_macro'
            cv = 5
        elif dataset_name == '7-point':
            def macro_recall_scorer(est, X_, y_):
                return recall_score(y_, est.predict(X_), average='macro')
            scoring = macro_recall_scorer
        else:
            scoring = 'accuracy'
        
        return cross_val_score(model, X_new, y_new, cv=cv, scoring=scoring).mean()
    
    best_rf_params = parallel_optuna_optimization(rf_objective, 100, 'maximize')
    if rank == 0 and best_rf_params:
        competitor_models['RF'] = RandomForestClassifier(**best_rf_params).fit(X_new, y_new)

     # LGBM
    def lgbm_objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
            'class_weight': 'balanced',
        }
        model = LGBMClassifier(**params)
        if dataset_name in ['Adult', 'Bank']:
            scoring = 'f1'
            cv = 3
        elif dataset_name == 'CTG':
            scoring = 'f1_macro'
            cv = 3
        elif dataset_name == '7-point':
            def macro_recall_scorer(est, X_, y_):
                return recall_score(y_, est.predict(X_), average='macro')
            scoring = macro_recall_scorer
        else:
            scoring = 'accuracy'
        
        return cross_val_score(model, X_new, y_new, cv=cv, scoring=scoring).mean()
    
    best_lgbm_params = parallel_optuna_optimization(lgbm_objective, 100, 'maximize')
    if rank == 0 and best_lgbm_params:
        competitor_models['LGBM'] = LGBMClassifier(**best_lgbm_params).fit(X_new, y_new)

    return competitor_models

##############################################################################################################
#   E. Train "Standard Corrected" Models (L1-LR+, L2-LR+, DT+, RF+, LGBM+, LGBM+C), (and LGBM+D for 7-point).
##############################################################################################################

def train_competitor_standard_corrected(dataset_name, X_new, y_new, old_model, 
                                      base_competitor_models, existing_models=None):
    
    """
    Train competitor models with correction using old model predictions.
    
    Parameters:
    - dataset_name: Name of the dataset.
    - X_new: New feature data.
    - y_new: New target data.
    - old_model: Old model for incremental training.
    - base_competitor_models: Dictionary of base competitor models.
    - existing_models: Optional dictionary of existing models.
    
    Returns:
    - A dictionary of trained competitor models.
    """
    
    competitor_models = existing_models if existing_models is not None else {}
    
    # Define scoring and CV folds based on dataset
    if dataset_name in ['Adult', 'Bank']:
        scoring = 'f1'
        cv = 3
    elif dataset_name == 'CTG':
        scoring = 'f1_macro'
        cv = 5
    elif dataset_name == '7-point':
        scoring = 'macro_recall'
        cv = 5
    else:
        raise ValueError("Unsupported dataset")
    
    for old_model_name, old_model_obj in base_competitor_models.items():
        old_scores_train = old_model_obj.predict_proba(X_new)
        X_train_concat = np.hstack([X_new, old_scores_train])
        
        if old_model_name == 'L1-LR':
            competitor_models[f"{old_model_name}+"] = LogisticRegressionCV(
                penalty='l1', solver='liblinear').fit(X_train_concat, y_new)
        elif old_model_name == 'L2-LR':
            competitor_models[f"{old_model_name}+"] = LogisticRegressionCV(
                penalty='l2').fit(X_train_concat, y_new)
        
        elif old_model_name == 'DT':
            competitor_models[f"{old_model_name}+"] = _optuna_decision_tree_corrected(
                X_train_concat, y_new, scoring, cv)
        elif old_model_name == 'RF':
            competitor_models[f"{old_model_name}+"] = _optuna_random_forest_corrected(
                X_train_concat, y_new, scoring, cv)
        
        elif old_model_name == 'LGBM':
            competitor_models[f"{old_model_name}+"] = _optuna_lgbm_corrected(
                X_train_concat, y_new, scoring, cv)
    
    # LGBM+C
    competitor_models['LGBM+C'] = _optuna_incremental(X_new, y_new, cv, old_model)
    
    return competitor_models


def _optuna_decision_tree_corrected(X_concat, y, scoring, cv):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.0001, 0.1),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1.0),
            'class_weight': 'balanced'
        }
        model = DecisionTreeClassifier(**params)
        return cross_val_score(model, X_concat, y, cv=cv, scoring=scoring).mean()
    
    best_params = parallel_optuna_optimization(objective, n_trials=100, direction='maximize')
    best_params = comm.bcast(best_params, root=0)  # Broadcast best params to all ranks
    
    return DecisionTreeClassifier(**best_params).fit(X_concat, y)

def _optuna_random_forest_corrected(X_concat, y_concat, scoring, cv):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.0001, 0.1),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1.0),
            'class_weight': 'balanced_subsample'
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X_concat, y_concat, cv=cv, scoring=scoring).mean()
    
    best_params = parallel_optuna_optimization(objective, n_trials=100, direction='maximize')
    best_params = comm.bcast(best_params, root=0)  # Broadcast best params to all ranks
    
    return RandomForestClassifier(**best_params).fit(X_concat, y_concat)

def _optuna_lgbm_corrected(X_concat, y_concat, scoring, cv):
    """
    MPI-parallel Optuna for LGBMClassifier, returning a fitted model on rank=0.
    """
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
            'class_weight': 'balanced'
            }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_concat, y_concat, cv=cv, scoring=scoring)
        return scores.mean()
    
    best_params = parallel_optuna_optimization(objective, n_trials=100, direction='maximize')
    if rank == 0 and best_params is not None:
        return LGBMClassifier(**best_params).fit(X_concat, y_concat)
    return None

def _optuna_incremental(X, y, n_splits, old_booster):
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'class_weight': 'balanced'}
            
        # Manual cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []

        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # For each fold, create a new LGBMClassifier and fit with old booster
            model = LGBMClassifier(**params)
            model.fit(
                X_train, 
                y_train, 
                init_model=old_booster.booster_
            )

            # Evaluate on the validation fold
            y_pred = model.predict_proba(X_valid)[:, 1]

            fold_score = roc_auc_score(y_valid, y_pred)
            scores.append(fold_score)
        return np.mean(scores)
        
    best_params = parallel_optuna_optimization(objective, n_trials=100, direction='maximize')
    best_params = comm.bcast(best_params, root=0)  # Broadcast best params to all ranks
        
    model = LGBMClassifier(**best_params).fit(X, y, init_model=old_booster.booster_)
    return model
###############################################################################
