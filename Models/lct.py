import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
import heapq

###############################################################################
# LocalCorrectionTree Implementation with Pruning Strategies
###############################################################################
class LocalCorrectionTree:
    def __init__(self, lambda_reg=0.1, max_depth=5, min_samples_leaf=64, nprune=1,
                 epsilon_min=0.01, epsilon_max=0.5, epsilon_fail=0.499):
        self.lambda_reg = lambda_reg
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.nprune = nprune
        self.epsilon_min = epsilon_min    # Lower bound on fraction of changed predictions in a leaf 
        self.epsilon_max = epsilon_max    # Upper bound on fraction of changed predictions in a leaf 
        self.epsilon_fail = epsilon_fail  # Maximum allowed fraction of changed predictions that go to the wrong class
        self.n_classes = None
        self.nodes = []      # Each element: (feature_index, threshold, correction weight vector)
        self.children = []   # Each element: (left_child, right_child)
        self.splits = []     # List of indices corresponding to node splits

    def fit(self, X, y, old_scores):
        n_samples, _ = X.shape
        self.n_classes = old_scores.shape[1]

        # Root node: get indices covering all samples and compute its correction weight.
        root_indices = np.arange(n_samples)
        w_root = self._find_w(root_indices, old_scores, y)
        root_obj = self._objective_function(w_root, root_indices, old_scores, y)
        heap = [(-root_obj, 0, root_indices, w_root)]
        
        # Initialize tree: start with the root as a (temporary) leaf.
        self.nodes = [(-1, None, w_root)]
        self.children = [(-1, -1)]
        self.splits = [root_indices]
        node_count = 1

        # Recursively split nodes (using a heap to select the “worst” node first)
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

            # Change current node from leaf to internal node
            self.nodes[node_id] = (j_star, t_star, w_node)
            self.children[node_id] = (left_id, right_id)

            if len(left_idx) >= self.min_samples_leaf:
                left_obj = self._objective_function(w_left, left_idx, old_scores, y)
                heapq.heappush(heap, (-left_obj, left_id, left_idx, w_left))
            if len(right_idx) >= self.min_samples_leaf:
                right_obj = self._objective_function(w_right, right_idx, old_scores, y)
                heapq.heappush(heap, (-right_obj, right_id, right_idx, w_right))
                
        self.prune(X, y, old_scores)
        self.simplify()
        
    def predict(self, X):
        n_samples = len(X)
        corrections = np.zeros((n_samples, self.n_classes))
        # For each sample traverse the tree until a leaf is reached.
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
        # Adding a small constant to avoid log(0)
        log_likelihood = -np.sum(np.log(probs[np.arange(len(indices)), sub_labels] + 1e-10))
        reg_term = self.lambda_reg * len(indices) * np.linalg.norm(w)
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
            # Use each unique value as a candidate threshold
            for t in unique_vals:
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
        # For each leaf node, obtain the indices of samples that fall into it.
        leaf_indices = [[] for _ in range(len(self.nodes))]
        for i in range(len(X)):
            node_id = 0
            while True:
                feature_idx, threshold, _ = self.nodes[node_id]
                left_id, right_id = self.children[node_id]
                if feature_idx == -1:
                    leaf_indices[node_id].append(i)
                    break
                if X[i, feature_idx] < threshold:
                    node_id = left_id
                else:
                    node_id = right_id
        # For each leaf, apply the pruning rules
        for node_id, idx_list in enumerate(leaf_indices):
            if len(idx_list) < self.nprune:
                continue  # Skip if not enough samples fall in the leaf as per nprune.
            idx_array = np.array(idx_list)
            old_preds = np.argmax(old_scores[idx_array], axis=1)
            # Each sample in a leaf gets the same correction weight as stored in the leaf node.
            w_leaf = self.nodes[node_id][2]
            corrected_scores = old_scores[idx_array] + w_leaf
            new_preds = np.argmax(corrected_scores, axis=1)
            # I(v): all samples reaching the leaf.
            num_samples = len(idx_array)
            # ^I(v): samples whose predictions changed because of the correction.
            changed = np.sum(old_preds != new_preds)
            # ^Ici(v): among the changed predictions count those that lead to an incorrect label.
            incorrect = np.sum((new_preds != y[idx_array]) & (old_preds != new_preds))
            ratio_changed = changed / num_samples

            # Prune if the fraction of changed predictions is too small or too high,
            # or if too many of the changed samples have an incorrect prediction.
            if ratio_changed < self.epsilon_min or ratio_changed > self.epsilon_max:
                self._zero_leaf(node_id)
            elif changed > 0 and (incorrect / changed) > self.epsilon_fail:
                self._zero_leaf(node_id)

    def _zero_leaf(self, node_id):
        # Zero-out the weight vector and mark this node as a leaf.
        feature_idx, _, w_node = self.nodes[node_id]
        self.nodes[node_id] = (-1, None, np.zeros_like(w_node))

    def simplify(self):
        # Second pruning strategy: remove redundant nodes whose children are both pruned.
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
                # Remove redundancy: set parent's correction to zero and mark as leaf.
                self.nodes[node_id] = (-1, None, np.zeros_like(w_node))
                self.children[node_id] = (-1, -1)
        simplify_node(0)

    def print_tree(self, node_id=0, indent="", feature_names=None):
        feature_idx, threshold, w_node = self.nodes[node_id]
        
        if feature_idx == -1:
            # Leaf node: print its correction vector.
            print(indent + "Leaf: correction =", w_node)
        else:
            # Use the provided feature_names if available; otherwise, use the raw index.
            if feature_names is not None:
                feature = feature_names[feature_idx]
            else:
                feature = "feature_" + str(feature_idx)
            print(indent + "Node: Feature " + feature + " <= " + "{:.4f}".format(threshold))
            
            left_id, right_id = self.children[node_id]
            if left_id != -1:
                self.print_tree(left_id, indent + "  ", feature_names)
            if right_id != -1:
                self.print_tree(right_id, indent + "  ", feature_names)
