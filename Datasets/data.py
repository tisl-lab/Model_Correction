import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class AdultDataset:
    def __init__(self):
        self.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                        'hours-per-week', 'native-country', 'income']
        self.cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        self.num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    def load_data(self, filepath='/Users/davembiazi/Desktop/Projects/Model Correction/datasets/adult/adult.data'):
        data = pd.read_csv(filepath, names=self.columns, skipinitialspace=True)
        data = data.dropna()
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
        encoded_features = encoder.fit_transform(data[self.cat_features])
        feature_names = encoder.get_feature_names_out(self.cat_features)
        
        X = np.hstack((data[self.num_features].values, encoded_features))
        y = (data['income'] == '>50K').astype(int)
        
        return X, y, list(self.num_features) + list(feature_names), data

    def split_old_new_data(self, X, y):
        total_samples = len(X)
        old_data_size = total_samples // 2
        
        # Calculate probabilities proportional to 1/age
        probabilities = 1 / X[:, 0]  # age is the first column
        probabilities /= probabilities.sum()
        
        old_indices = np.random.choice(total_samples, size=old_data_size, replace=False, p=probabilities)
        new_indices = np.setdiff1d(np.arange(total_samples), old_indices)
        
        X_old, y_old = X[old_indices], y[old_indices]
        X_new, y_new = X[new_indices], y[new_indices]
        
        return X_old, y_old, X_new, y_new