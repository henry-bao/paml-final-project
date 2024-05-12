import numpy as np


class DecisionTree:
    def __init__(
        self, max_depth=None, feature_sampling_factor=1.0, threshold_sampling_factor=1.0
    ):
        self.max_depth = max_depth
        self.feature_sampling_factor = feature_sampling_factor
        self.threshold_sampling_factor = threshold_sampling_factor

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes_ = len(self.classes_)
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self.classes_[self._traverse_tree(x, self.tree)] for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse_tree_proba(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        num_labels = len(np.unique(y))

        # Base case: Leaf node (return class distribution as probabilities)
        if depth == self.max_depth or num_labels == 1 or len(y) == 0:
            leaf_distribution = np.zeros(self.num_classes_)
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                leaf_distribution[np.searchsorted(self.classes_, unique_classes)] = (
                    counts / len(y)
                )
            return {"leaf": True, "class_distribution": leaf_distribution}

        best_feature, best_threshold = self._find_best_split(X, y)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
            "leaf": False,
        }

    def _find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        num_samples, num_features = X.shape
        sampled_features = np.random.choice(
            num_features,
            int(self.feature_sampling_factor * num_features),
            replace=False,
        )

        for feature in sampled_features:
            thresholds = np.unique(X[:, feature])
            sampled_thresholds = np.random.choice(
                thresholds,
                int(self.threshold_sampling_factor * len(thresholds)),
                replace=False,
            )

            for threshold in sampled_thresholds:
                gain = self._calculate_gain(X, y, feature, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gain(self, X, y, feature, threshold):
        parent_entropy = self._calculate_entropy(y)

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        left_entropy = self._calculate_entropy(y[left_indices])
        right_entropy = self._calculate_entropy(y[right_indices])

        num_left = len(y[left_indices])
        num_right = len(y[right_indices])
        total_samples = num_left + num_right

        gain = (
            parent_entropy
            - ((num_left / total_samples) * left_entropy)
            - ((num_right / total_samples) * right_entropy)
        )

        return gain

    def _calculate_entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        return entropy

    def _traverse_tree(self, x, tree):
        if tree.get("leaf"):
            return np.argmax(tree["class_distribution"])
        else:
            feature = tree["feature"]
            threshold = tree["threshold"]

            if x[feature] <= threshold:
                return self._traverse_tree(x, tree["left"])
            else:
                return self._traverse_tree(x, tree["right"])

    def _traverse_tree_proba(self, x, tree):
        if tree.get("leaf"):
            return tree["class_distribution"]
        else:
            feature = tree["feature"]
            threshold = tree["threshold"]

            if x[feature] <= threshold:
                return self._traverse_tree_proba(x, tree["left"])
            else:
                return self._traverse_tree_proba(x, tree["right"])
