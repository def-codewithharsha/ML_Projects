import numpy as np
from collections import Counter

#  NODE 
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # not None only for leaf nodes


#  DECISION TREE 
class DecisionTree:
    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts[counts > 0] / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        feature_indices = np.random.choice(
            n_features, self.max_features, replace=False
        )

        for feature in feature_indices:
            values = np.unique(X[:, feature])
            if len(values) == 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf or
                    right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                gini = (
                    left_mask.sum() / n_samples * self._gini(y[left_mask]) +
                    right_mask.sum() / n_samples * self._gini(y[right_mask])
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
            return Node(value=Counter(y).most_common(1)[0][0])

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


#  RANDOM FOREST 
class RandomForest:
    def __init__(
        self,
        n_trees=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_features = X.shape[1]
        self.max_features = self.max_features or int(np.sqrt(n_features))
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )

            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = predictions.T

        return np.array([
            Counter(row).most_common(1)[0][0] for row in predictions
        ])
