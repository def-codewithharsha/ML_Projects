import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)
    def _split(self, X, y, feature, threshold):
        left = X[:, feature] <= threshold
        right = X[:, feature] > threshold
        return X[left], X[right], y[left], y[right]
    def _best_split(self, X, y):
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        _, n_features = X.shape

        features = np.random.choice(
            n_features,
            self.max_features,
            replace=False
        )

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_l, X_r, y_l, y_r = self._split(X, y, feature, threshold)

                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                gini = (
                    len(y_l) / len(y) * self._gini(y_l) +
                    len(y_r) / len(y) * self._gini(y_r)
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    def _build_tree(self, X, y, depth):
        if (
            depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1
        ):
            return Node(value=self._most_common_label(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=self._most_common_label(y))

        X_l, X_r, y_l, y_r = self._split(X, y, feature, threshold)

        left = self._build_tree(X_l, y_l, depth + 1)
        right = self._build_tree(X_r, y_r, depth + 1)

        return Node(feature, threshold, left, right)
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    def fit(self, X, y):
        n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )

            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        predictions = []
        for sample_preds in tree_preds:
            predictions.append(Counter(sample_preds).most_common(1)[0][0])

        return np.array(predictions)
