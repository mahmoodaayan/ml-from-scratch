import numpy as np

def euclidean_distance(point1, point2):

    point1 = np.array(point1)
    point2 = np.array(point2)

    return np.sqrt(np.sum((point1 - point2)**2))

def get_all_distances(test_points, training_data):
    distances = []
    for row in training_data:
        distances.append(euclidean_distance(test_points, row))
    return distances

def get_k_nearest(distance, labels, k):
    sorted_indicies = np.argsort(distance)
    k_nearest_labels = labels[sorted_indicies[:k]]
    return k_nearest_labels

def predict (k_nearest_labels):
    return np.argmax(np.bincount(k_nearest_labels))

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_single(self, test_point):
        distances = get_all_distances(test_point, self.X_train)
        k_nearest_labels = get_k_nearest(distances, self.y_train, self.k)
        return predict(k_nearest_labels)

    def predict (self, X_test):
        predictions = []
        for test_point in X_test:
            predictions.append(self.predict_single(test_point))

        return predictions

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test our KNN
knn = KNN(k=10)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
