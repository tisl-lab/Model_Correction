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
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.01, 0.1, log=True),
            'max_features': trial.suggest_float('max_features', 0.3, 0.8, log=False),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1e-3, log=True),
            'class_weight': 'balanced'
        }
        model = DecisionTreeClassifier(**params)
    elif model_type == 'rf':
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.01, 0.1, log=True),
            'max_features': trial.suggest_float('max_features', 0.3, 0.8, log=False),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1e-3, log=True),
            'class_weight': 'balanced_subsample',
            'n_estimators': trial.suggest_int('n_estimators', 50, 200)
        }
        model = RandomForestClassifier(**params)

    score = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return score.mean()

def train_old_model(dataset_name, X, y):
    if dataset_name in ['Adult', 'Bank', 'CTG']:
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(
                trial, 
                X, 
                y, 
                'lgbm',
                cv=3 if dataset_name in ['Adult', 'Bank'] else 5,
                scoring='f1' if dataset_name in ['Adult', 'Bank'] else 'f1_macro',
            ),
            n_trials=50
        )
        best_params = study.best_params
        best_params['class_weight'] = 'balanced'
        model = LGBMClassifier(**best_params)
        model.fit(X, y)
        return model
    else:
        return None

def train_new_models(dataset, X_new, y_new, old_model, localcorrectiontree):
    models = {}
    
    # LCT optimization
    def lct_objective(trial):
        lambda_reg = trial.suggest_float("lambda_reg", 0.05, 0.5, log=True)
        lct = localcorrectiontree(lambda_reg=lambda_reg, max_depth=5, min_samples_leaf=64)
        old_scores_new = old_model.predict_proba(X_new)
        lct.fit(X_new, y_new, old_scores_new)
        corrections = lct.predict(X_new)
        corrected_scores = old_scores_new + corrections
        return -np.mean(np.argmax(corrected_scores, axis=1) == y_new)

    study = optuna.create_study(direction="minimize")
    study.optimize(lct_objective, n_trials=50)
    
    best_lambda = study.best_trial.params["lambda_reg"]
    
    # Create LCT (Ours model)
    old_scores_new = old_model.predict_proba(X_new)
    models['Ours'] = localcorrectiontree(
        lambda_reg=best_lambda,
        max_depth=5,
        min_samples_leaf=64 if dataset in ['Adult', 'Bank'] else 8 if dataset == 'CTG' else 16
    )
    models['Ours'].fit(X_new, y_new, old_scores_new)
    
    # Train L1-LR and L2-LR
    for penalty in ['l1', 'l2']:
        models[f'{penalty.upper()}-LR'] = LogisticRegressionCV(
            penalty=penalty, cv=3, scoring='f1', max_iter=1000)
        models[f'{penalty.upper()}-LR'].fit(X_new, y_new)
        
        X_new_with_scores = np.hstack([X_new, old_scores_new])
        models[f'{penalty.upper()}-LR+'] = LogisticRegressionCV(
            penalty=penalty, cv=3, scoring='f1', max_iter=1000)
        models[f'{penalty.upper()}-LR+'].fit(X_new_with_scores, y_new)
    
    # Train DT, RF, LGBM
    for model_type in ['dt', 'rf', 'lgbm']:
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda local_trial: objective(
                local_trial,
                X_new,
                y_new,
                model_type,
                cv=3 if dataset in ['Adult', 'Bank'] else 5,
                scoring='f1' if dataset in ['Adult', 'Bank'] else 'f1_macro'
            ),
            n_trials=50
        )
        best_params = study.best_params

        if model_type == 'dt':
            models['DT'] = DecisionTreeClassifier(**best_params)
            models['DT+'] = DecisionTreeClassifier(**best_params)
        elif model_type == 'rf':
            models['RF'] = RandomForestClassifier(**best_params)
            models['RF+'] = RandomForestClassifier(**best_params)
        elif model_type == 'lgbm':
            models['LGBM'] = LGBMClassifier(**best_params)
            models['LGBM+'] = LGBMClassifier(**best_params)

        models[model_type.upper()].fit(X_new, y_new)
        
        X_new_with_scores = np.hstack([X_new, old_scores_new])
        models[f'{model_type.upper()}+'].fit(X_new_with_scores, y_new)
    
    if dataset in ['Adult', 'Bank', 'CTG']:
        models['LGBM+C'] = LGBMClassifier(**best_params)
        models['LGBM+C'].fit(X_new, y_new, init_model=old_model)
    
    return models