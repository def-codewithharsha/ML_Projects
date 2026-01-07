import numpy as np

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

class DecisionTree:
    def __init__(self,max_depth=5,min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.root=None

    def _gini(self,y):
        classes, counts = np.unique(y,return_counts=True)
        probabilities = counts/counts.sum()
        return 1 - np.sum(probabilities**2)
    
    def _split(self,X,y,feature,threshold):
        left_mask=X[:,feature] <= threshold
        right_mask = X[:,feature] > threshold
        return X[left_mask],X[right_mask],y[left_mask],y[right_mask]
    
    def _best_split(self,X,y):
        best_gini = float("inf")
        best_feature,best_threshold=None,None

        n_samples,n_feature = X.shape

        for feature in range(n_feature):
            thresholds=np.unique(X[:,feature])
            for threshold in thresholds:
                X_l, X_r, y_l, y_r = self._split(X, y, feature, threshold)

                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                gini =(
                    len(y_l) / len(y) * self._gini(y_l) +
                    len(y_r) / len(y) * self._gini(y_r)
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        num_classes = len(np.unique(y))

        # Stopping conditions
        if (
            depth >= self.max_depth or
            num_classes == 1 or
            num_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=self._most_common_label(y))

        X_l, X_r, y_l, y_r = self._split(X, y, feature, threshold)

        left = self._build_tree(X_l, y_l, depth + 1)
        right = self._build_tree(X_r, y_r, depth + 1)

        return Node(feature, threshold, left, right)
    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]
    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])
