import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import optuna

def objective(trial, X, y, model_type, cv, scoring):
    if model_type == 'lgbm':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'class_weight': 'balanced',
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100)
        }
        model = LGBMClassifier(**params)
    elif model_type == 'dt':
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.0001, 0.1, log=True),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0, log=False),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1.0, log=True),
            'class_weight': 'balanced'
        }
        model = DecisionTreeClassifier(**params)
    elif model_type == 'rf':
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.0001, 0.1, log=True),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0, log=False),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1.0, log=True),
            'class_weight': 'balanced_subsample',
            #'n_estimators': trial.suggest_int('n_estimators', 50, 200)
        }
        model = RandomForestClassifier(**params)

    score = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return score.mean()

def train_old_model(dataset_name, X_new, X_old, y_new, y_old):
    """Trains base models: L1-LR, L2-LR, DT, RF, and LGBM"""
    models = {}
    
    # Train L1-LR and L2-LR base models
    models['L1-LR'] = LogisticRegressionCV(
        penalty='l1', 
        solver='liblinear',
        cv=3, 
        scoring='f1', 
        max_iter=1000
    )
    models['L1-LR'].fit(X_new, y_new)
    
    models['L2-LR'] = LogisticRegressionCV(
        penalty='l2',
        solver='lbfgs',
        cv=3,
        scoring='f1',
        max_iter=1000
    )
    models['L2-LR'].fit(X_new, y_new)
    
    # Train DT, RF, LGBM with hyperparameter optimization
    for model_type in ['dt', 'rf', 'lgbm']:
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(
                trial, 
                X_old, 
                y_old, 
                model_type,
                cv=3 if dataset_name in ['Adult', 'Bank'] else 5,
                scoring='f1' if dataset_name in ['Adult', 'Bank'] else 'f1_macro',
            ),
            n_trials=100
        )
        best_params = study.best_params
        
        if model_type == 'dt':
            models['DT'] = DecisionTreeClassifier(**best_params)
        elif model_type == 'rf':
            models['RF'] = RandomForestClassifier(**best_params)
        elif model_type == 'lgbm':
            models['LGBM'] = LGBMClassifier(**best_params)
        
        models[model_type.upper()].fit(X_new, y_new)
    
    return models


def train_new_models(dataset, X_new, y_new, old_models, localcorrectiontree):
    """Trains enhanced models: L1-LR+, L2-LR+, DT+, RF+, LGBM+, LGBM+C, and Ours"""
    models = {}
    
    # Get predictions from all base models
    old_scores = {}
    for model_name, model in old_models.items():
        old_scores[model_name] = model.predict_proba(X_new)
    
    # LCT (Ours) optimization
    def lct_objective(trial):
        if dataset in ['Adult', 'Bank', 'CTG']:
            lambda_reg = trial.suggest_float("lambda_reg", 0.05, 0.5, log=True)
        else:  # 7-point dataset
            lambda_reg = trial.suggest_float("lambda_reg", 0.01, 0.1, log=True)
        
        max_depth = 5
        min_samples_leaf = 64 if dataset in ['Adult', 'Bank'] else 8 if dataset == 'CTG' else 16
        
        lct = localcorrectiontree(
            lambda_reg=lambda_reg,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            #nprune=1,
            #min=0.01,
            #max=0.5 if dataset in ['Adult', 'Bank', '7-point'] else 1.0,
            #fail=0.499
        )
        lct.fit(X_new, y_new, old_scores['LGBM'])
        corrections = lct.predict(X_new)
        corrected_scores = old_scores['LGBM'] + corrections
        return -np.mean(np.argmax(corrected_scores, axis=1) == y_new)

    study = optuna.create_study(direction="minimize")
    study.optimize(lct_objective, n_trials=100)
    best_lambda = study.best_trial.params["lambda_reg"]
    
    # Train LCT (Ours model)
    models['Ours'] = localcorrectiontree(
        lambda_reg=best_lambda,
        max_depth=5,
        min_samples_leaf=64 if dataset in ['Adult', 'Bank'] else 8 if dataset == 'CTG' else 16,
        nprune=1,
        min=0.01,
        max=0.5 if dataset in ['Adult', 'Bank', '7-point'] else 1.0,
        fail=0.499
    )
    models['Ours'].fit(X_new, y_new, old_scores['LGBM'])
    
    # Train enhanced versions of base models
    for penalty in ['l1', 'l2']:
        base_scores = old_scores[f'{penalty.upper()}-LR']
        X_new_with_scores = np.hstack([X_new, base_scores])
        models[f'{penalty.upper()}-LR+'] = LogisticRegressionCV(
            penalty=penalty, cv=3, scoring='f1', max_iter=1000)
        models[f'{penalty.upper()}-LR+'].fit(X_new_with_scores, y_new)
    
    # Train enhanced tree-based models
    for model_type in ['dt', 'rf', 'lgbm']:
        base_scores = old_scores[model_type.upper()]
        X_new_with_scores = np.hstack([X_new, base_scores])
        
        if model_type == 'dt':
            models['DT+'] = DecisionTreeClassifier(**old_models['DT'].get_params())
        elif model_type == 'rf':
            models['RF+'] = RandomForestClassifier(**old_models['RF'].get_params())
        elif model_type == 'lgbm':
            models['LGBM+'] = LGBMClassifier(**old_models['LGBM'].get_params())
            if dataset in ['Adult', 'Bank', 'CTG']:
                models['LGBM+C'] = LGBMClassifier(**old_models['LGBM'].get_params())
                models['LGBM+C'].fit(X_new, y_new, init_model=old_models['LGBM'])
        
        models[f'{model_type.upper()}+'].fit(X_new_with_scores, y_new)
    
    return models