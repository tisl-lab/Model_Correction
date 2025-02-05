import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
import heapq

###############################################################################
# LocalCorrectionTree Implementation
###############################################################################
class LocalCorrectionTree:
    def __init__(self, lambda_reg=0.1, max_depth=5, min_samples_leaf=64, nprune=1, epsilon_min=0.01, epsilon_max=0.5, epsilon_fail=0.499):
        self.lambda_reg = lambda_reg
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.nprune = nprune
        self.n_min = epsilon_min
        self.n_max = epsilon_max
        self.n_fail = epsilon_fail
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

            #cut_points = (unique_vals[:-1] + unique_vals[1:]) / 2
            cut_points = unique_vals
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

    def prune(self, X, y, old_scores):
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
            if len(idx_list) < self.nprune:
                continue

            idx_array = np.array(idx_list)
            old_preds = np.argmax(old_scores[idx_array], axis=1)
            new_scores = old_scores[idx_array] + self.predict(X[idx_array])
            new_preds = np.argmax(new_scores, axis=1)

            changed = np.sum(old_preds != new_preds)
            incorrect = np.sum((new_preds != y[idx_array]) & (old_preds != new_preds))
            ratio_changed = changed / len(idx_list) if len(idx_list) > 0 else 0

            if ratio_changed < self.n_min or ratio_changed > self.n_max:
                self._zero_leaf(node_id)
            elif changed > 0 and (incorrect / changed) > self.n_fail:
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
