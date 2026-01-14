import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def _split(self, X, y, feature, threshold):
        mask = X[:, feature] <= threshold
        return X[mask], X[~mask], y[mask], y[~mask]

    def _best_split(self, X, y):
        best_gini = float("inf") 
        best_feature, best_threshold = None, None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            if len(values) == 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                _, _, y_l, y_r = self._split(X, y, feature, threshold)
                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                gini = (
                    len(y_l) / n_samples * self._gini(y_l) +
                    len(y_r) / n_samples * self._gini(y_r)
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

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

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _predict_sample(self, x, node):
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])
