import numpy as np

class KMeansClassifier:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # Inicializar centroides aleatorios
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iters):
            # Asignar cada punto al cluster más cercano
            labels = self._assign_clusters(X)

            # Actualizar los centroides
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Verificar la convergencia
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def predict(self, X):
        # Asignar cada punto al cluster más cercano y devolver las etiquetas
        return self._assign_clusters(X)

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento
    X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 2], [8, 8], [9, 10], [10, 8]])

    # Crear y entrenar el clasificador KMeans
    kmeans_classifier = KMeansClassifier(k=2)
    kmeans_classifier.fit(X_train)

    # Realizar predicciones
    predictions = kmeans_classifier.predict(X_train)

    print("Predicciones:", predictions)
    print("Centroides finales:", kmeans_classifier.centroids)
