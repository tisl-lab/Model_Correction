import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
import heapq

###############################################################################
# LocalCorrectionTree Implementation
###############################################################################
class LocalCorrectionTree:
    def __init__(self, lambda_reg=0.1, max_depth=5, min_samples_leaf=64):
        self.lambda_reg = lambda_reg
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = None
        self.nodes = []
        self.children = []
        self.splits = []

    def fit(self, X, y, old_scores):
        n_samples, _ = X.shape
        self.n_classes = old_scores.shape[1]

        root_indices = np.arange(n_samples)
        w_root = self._find_w(root_indices, old_scores, y)

        root_obj = self._objective_function(w_root, root_indices, old_scores, y)
        heap = [(-root_obj, 0, root_indices, w_root)]
        
        self.nodes = [(-1, None, w_root)]
        self.children = [(-1, -1)]
        self.splits = [root_indices]

        node_count = 1

        while heap and node_count < 2 ** self.max_depth:
            neg_obj, node_id, indices, w_node = heapq.heappop(heap)

            if len(indices) < 2 * self.min_samples_leaf:
                continue

            best_split = self._find_best_split(X, old_scores, y, indices)
            if best_split is None:
                continue

            j_star, t_star, w_left, w_right, left_idx, right_idx, _ = best_split

            left_id = node_count
            node_count += 1
            self.nodes.append((-1, None, w_left))
            self.children.append((-1, -1))
            self.splits.append(left_idx)

            right_id = node_count
            node_count += 1
            self.nodes.append((-1, None, w_right))
            self.children.append((-1, -1))
            self.splits.append(right_idx)

            self.nodes[node_id] = (j_star, t_star, w_node)
            self.children[node_id] = (left_id, right_id)

            if len(left_idx) >= self.min_samples_leaf:
                left_obj = self._objective_function(w_left, left_idx, old_scores, y)
                heapq.heappush(heap, (-left_obj, left_id, left_idx, w_left))
            if len(right_idx) >= self.min_samples_leaf:
                right_obj = self._objective_function(w_right, right_idx, old_scores, y)
                heapq.heappush(heap, (-right_obj, right_id, right_idx, w_right))

    def predict(self, X):
        n_samples = len(X)
        corrections = np.zeros((n_samples, self.n_classes))

        for i in range(n_samples):
            node_id = 0
            while True:
                feature_idx, threshold, w_node = self.nodes[node_id]
                left_id, right_id = self.children[node_id]

                if feature_idx == -1:
                    corrections[i] = w_node
                    break

                if X[i, feature_idx] < threshold:
                    node_id = left_id
                else:
                    node_id = right_id
        return corrections

    def _objective_function(self, w, indices, old_scores, y):
        w = w.reshape(-1, self.n_classes)
        sub_scores = old_scores[indices] + w
        sub_labels = y[indices]
        probs = softmax(sub_scores, axis=1)
        log_likelihood = -np.sum(np.log(probs[np.arange(len(indices)), sub_labels]))
        reg_term = self.lambda_reg * len(indices) * np.sum(w**2)
        return log_likelihood + reg_term

    def _find_w(self, indices, old_scores, y):
        init_w = np.zeros((1, self.n_classes))

        def obj_func(w_flat):
            w_reshaped = w_flat.reshape(1, self.n_classes)
            return self._objective_function(w_reshaped, indices, old_scores, y)

        res = minimize(obj_func, init_w.flatten(), method='L-BFGS-B')
        return res.x.reshape(1, self.n_classes)

    def _find_best_split(self, X, old_scores, y, indices):
        best_obj = np.inf
        best_split = None

        for j in range(X.shape[1]):
            x_vals = X[indices, j]
            unique_vals = np.unique(x_vals)
            if len(unique_vals) <= 1:
                continue

            cut_points = (unique_vals[:-1] + unique_vals[1:]) / 2
            for t in cut_points:
                left_indices = indices[x_vals < t]
                right_indices = indices[x_vals >= t]

                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue

                w_left = self._find_w(left_indices, old_scores, y)
                w_right = self._find_w(right_indices, old_scores, y)

                obj_left = self._objective_function(w_left, left_indices, old_scores, y)
                obj_right = self._objective_function(w_right, right_indices, old_scores, y)
                total_obj = obj_left + obj_right

                if total_obj < best_obj:
                    best_obj = total_obj
                    best_split = (j, t, w_left, w_right, left_indices, right_indices, best_obj)

        return best_split

    def prune(self, X, y, old_scores, nprune=1, epsilon_min=0.01, epsilon_max=0.5, epsilon_fail=0.499):
        leaf_indices = [[] for _ in range(len(self.nodes))]
        for i in range(len(X)):
            node_id = 0
            while True:
                feature_idx, threshold, w_node = self.nodes[node_id]
                left_id, right_id = self.children[node_id]
                if feature_idx == -1:
                    leaf_indices[node_id].append(i)
                    break
                if X[i, feature_idx] < threshold:
                    node_id = left_id
                else:
                    node_id = right_id

        for node_id, idx_list in enumerate(leaf_indices):
            if len(idx_list) < nprune:
                continue

            idx_array = np.array(idx_list)
            old_preds = np.argmax(old_scores[idx_array], axis=1)
            new_scores = old_scores[idx_array] + self.predict(X[idx_array])
            new_preds = np.argmax(new_scores, axis=1)

            changed = np.sum(old_preds != new_preds)
            incorrect = np.sum((new_preds != y[idx_array]) & (old_preds != new_preds))
            ratio_changed = changed / len(idx_list) if len(idx_list) > 0 else 0

            if ratio_changed < epsilon_min or ratio_changed > epsilon_max:
                self._zero_leaf(node_id)
            elif changed > 0 and (incorrect / changed) > epsilon_fail:
                self._zero_leaf(node_id)

    def _zero_leaf(self, node_id):
        self.nodes[node_id] = (-1, None, np.zeros_like(self.nodes[node_id][2]))

    def simplify(self):
        def simplify_node(node_id):
            feature_idx, threshold, w_node = self.nodes[node_id]
            left_id, right_id = self.children[node_id]
            if feature_idx == -1:
                return

            simplify_node(left_id)
            simplify_node(right_id)

            left_f, _, left_w = self.nodes[left_id]
            right_f, _, right_w = self.nodes[right_id]
            
            if (left_f == -1 and np.allclose(left_w, 0) and
                right_f == -1 and np.allclose(right_w, 0)):
                self.nodes[node_id] = (-1, None, np.zeros_like(w_node))

        simplify_node(0)

###############################################################################
# Test Function
###############################################################################
def test_local_correction_tree():
    """
    Tests LocalCorrectionTree using random data.
    Includes assertions for each main step: fit, predict, prune, simplify.
    """
    from numpy.random import rand, randint, randn, seed

    seed(42)
    n_samples = 60
    n_features = 5
    n_classes = 3
    X = rand(n_samples, n_features)
    y = randint(0, n_classes, size=n_samples)
    old_scores = randn(n_samples, n_classes)

    # Instantiate LCT
    lct = LocalCorrectionTree(lambda_reg=0.1, max_depth=3, min_samples_leaf=5)

    lct.fit(X, y, old_scores)
    corrections = lct.predict(X)
    assert corrections.shape == (n_samples, n_classes), "Predict output shape is incorrect"

    pre_prune_nodes = len(lct.nodes)
    lct.prune(X, y, old_scores, nprune=2)
    post_prune_nodes = len(lct.nodes)
    assert post_prune_nodes <= pre_prune_nodes, "Prune should not increase node count"

    pre_simplify_nodes = len(lct.nodes)
    lct.simplify()
    post_simplify_nodes = len(lct.nodes)
    assert post_simplify_nodes <= pre_simplify_nodes, "Simplify should not increase node count"

    print("All tests for LocalCorrectionTree passed successfully!")

###############################################################################
# Example: Randomly Sample "lambda_reg" with Optuna
###############################################################################
def objective(trial):
    """
    Optuna objective function that samples lambda_reg in [0.05, 0.5],
    then trains an LCT on random data and returns a dummy 'loss'.
    Replace this dummy training code with real data if you want a meaningful search.
    """
    # Sample random lambda
    lambda_reg = trial.suggest_float("lambda_reg", 0.05, 0.5, log=False)

    # Generate random data just for demonstration
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 3, size=50)
    old_scores = np.random.randn(50, 3)

    # Instantiate an LCT with the sampled lambda_reg
    lct_temp = LocalCorrectionTree(
        lambda_reg=lambda_reg,
        max_depth=3,
        min_samples_leaf=5,
    )
    # "Train" on the random data
    lct_temp.fit(X, y, old_scores)

    # Evaluate some dummy metric (for demonstration).
    # Typically, you'd do cross-validation or measure real performance here.
    preds = lct_temp.predict(X)
    # We'll create a dummy "loss" = sum of squares of corrections, smaller is "better"
    dummy_loss = float(np.sum(preds**2))
    return dummy_loss

#if __name__ == "__main__":
    # 1. Run your LCT tests
    #test_local_correction_tree()

    # 2. Optimizing with real random sampling for lambda_reg
    #import optuna
    #study = optuna.create_study(direction="minimize")
    #study.optimize(objective, n_trials=10)  # e.g. 10 random trials

    # 3. Print best trial and use it to instantiate LCT
    #best_trial = study.best_trial
    #best_lambda = best_trial.params["lambda_reg"]
    #print(f"\nOptuna found best lambda_reg={best_lambda:.3f} with objective={best_trial.value:.3f}.")

    ## 4. Example: build a final LCT with the best lambda and possibly real data
    #final_lct = LocalCorrectionTree(lambda_reg=best_lambda, max_depth=3, min_samples_leaf=5)
    #print(f"Final LCT instantiated with lambda_reg={best_lambda:.3f}.")
    
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Assuming the LocalCorrectionTree class is already defined

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple logistic regression model as the "old model"
old_model = LogisticRegression(random_state=42)
old_model.fit(X_train, y_train)

# Get old model scores on the test set
old_scores = old_model.predict_proba(X_test)

# Instantiate and fit the Local Correction Tree
lct = LocalCorrectionTree(lambda_reg=0.1, max_depth=3, min_samples_leaf=10)
lct.fit(X_test, y_test, old_scores)

# Get corrections from LCT
corrections = lct.predict(X_test)

# Apply corrections to old model scores
corrected_scores = old_scores + corrections

# Evaluate the performance
old_accuracy = np.mean(np.argmax(old_scores, axis=1) == y_test)
new_accuracy = np.mean(np.argmax(corrected_scores, axis=1) == y_test)

print(f"Old model accuracy: {old_accuracy:.4f}")
print(f"Corrected model accuracy: {new_accuracy:.4f}")

# Analyze the corrections
non_zero_corrections = np.sum(np.any(corrections != 0, axis=1))
print(f"Number of samples with non-zero corrections: {non_zero_corrections}")
print(f"Percentage of samples corrected: {non_zero_corrections / len(X_test) * 100:.2f}%")

# Print a few example corrections
print("\nExample corrections:")
for i in range(5):
    print(f"Sample {i}:")
    print(f"  Old scores: {old_scores[i]}")
    print(f"  Correction: {corrections[i]}")
    print(f"  New scores: {corrected_scores[i]}")
    print(f"  True label: {y_test[i]}")
    print()

# Visualize the tree structure
lct.simplify()
print("\nSimplified tree structure:")
def print_tree(node_id, depth=0):
    feature_idx, threshold, w_node = lct.nodes[node_id]
    left_id, right_id = lct.children[node_id]
    
    if feature_idx == -1:
        print("  " * depth + f"Leaf: correction = {w_node}")
    else:
        print("  " * depth + f"Node: X[{feature_idx}] <= {threshold:.4f}")
        print_tree(left_id, depth + 1)
        print_tree(right_id, depth + 1)

print_tree(0)