import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo

class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == "Adult":
            self.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                            'hours-per-week', 'native-country', 'income']
            self.cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 
                                'race', 'sex', 'native-country']
            self.num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            self.target = 'income'
            self.positive_class = '>50K'
            
        elif dataset_name == 'Bank':
            self.cat_features = ["job", "marital", "education", "default", "housing", "loan",
                                 "contact", "month", "day_of_week", "poutcome"]
            self.num_features = ["age", "campaign", "pdays", 'balance', "previous"]
            self.target = 'y'
            self.positive_class = 'yes'
        else:
            raise ValueError("Unsupported dataset. Choose 'Adult' or 'Bank'.")

    def load_data(self):
        if self.dataset_name == "Adult":
            filepath = '/Users/davembiazi/Desktop/Projects/Model Correction/datasets/adult/adult.data'
            data = pd.read_csv(filepath, names=self.columns, skipinitialspace=True)
            data = data.dropna()
        elif self.dataset_name == "Bank":
            bank_marketing = fetch_ucirepo(id=222)
            data = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)
            data = data.drop('duration', axis=1)
        else:
            raise ValueError("Unsupported dataset. Choose 'Adult' or 'Bank'.")
        
        empty_features = [col for col in self.cat_features if data[col].empty]
        if empty_features:
            raise ValueError(f"The following categorical features are empty: {empty_features}")
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        encoded_features = encoder.fit_transform(data[self.cat_features])
        encoded_feature_names = encoder.get_feature_names_out(self.cat_features)
        
        X = np.hstack((data[self.num_features].values, encoded_features))
        y = (data[self.target] == self.positive_class).astype(int)
        
        all_feature_names = self.num_features + list(encoded_feature_names)
        
        return X, y, all_feature_names, data

    def split_old_new_data(self, X, y):
        total_samples = len(X)
        old_data_size = total_samples // 2
        
        probabilities = 1 / X[:, 0]
        probabilities /= probabilities.sum()
        
        old_indices = np.random.choice(total_samples, size=old_data_size, replace=False, p=probabilities)
        new_indices = np.setdiff1d(np.arange(total_samples), old_indices)
        
        X_old, y_old = X[old_indices], y[old_indices]
        X_new, y_new = X[new_indices], y[new_indices]
        
        return X_old, y_old, X_new, y_new

# Usage example
'''datasets = ['Bank', 'Adult']
for dataset_name in datasets:
    print(f"\nProcessing {dataset_name} dataset:")
    dataset = Dataset(dataset_name)
    X, y, feature_names, raw_data = dataset.load_data()
    print(f"Feature: {X[:,0]}")
    X_old, y_old, X_new, y_new = dataset.split_old_new_data(X, y)

    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"Old data samples: {len(X_old)}, Positive samples: {y_old.sum()}")
    print(f"New data samples: {len(X_new)}, Positive samples: {y_new.sum()}")
'''