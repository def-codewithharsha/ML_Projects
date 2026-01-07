import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def _predict_single(self,x):
        distances = []

        for x_train in self.X_train:
            distance = self._euclidean_distance(x,x_train)
            distances.append(distance)

        k_indices = np.argsort(distances)[:self.k]

        k_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        
        return np.array(predictions)