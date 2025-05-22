import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    rand_score,
    fowlkes_mallows_score,
    mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from dataset_manager import DatasetManager


class CustomKMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        """
        Собственная реализация алгоритма K-Means.

        Параметры:
            n_clusters (int): Количество кластеров.
            max_iter (int): Максимальное число итераций.
            tol (float): Порог сходимости.
            random_state (int): Фиксация генератора случайных чисел.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels = None

    def fit(self, X: pd.DataFrame) -> None:
        """
        Обучает модель на данных X.

        Параметры:
            X (pd.DataFrame): Массив признаков.
        """
        np.random.seed(self.random_state)
        self.X = X.to_numpy()
        n_samples, n_features = self.X.shape

        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = self.X[random_idx]

        for iteration in range(self.max_iter):
            distances = self._euclidean_distance(self.X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [
                    self.X[self.labels == i].mean(axis=0)
                    if np.any(self.labels == i)
                    else self.centroids[i]
                    for i in range(self.n_clusters)
                ]
            )

            shift = np.linalg.norm(self.centroids - new_centroids)
            if shift < self.tol:
                break
            self.centroids = new_centroids

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Присваивает метки кластерам новым объектам.

        Параметры:
            X (pd.DataFrame): Признаки новых объектов.

        Возвращает:
            np.ndarray: Массив меток.
        """
        distances = self._euclidean_distance(X.to_numpy(), self.centroids)
        return np.argmin(distances, axis=1)

    def _euclidean_distance(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Вычисляет матрицу евклидовых расстояний между двумя матрицами точек.
        """
        return np.linalg.norm(A[:, np.newaxis] - B, axis=2)

    def get_labels(self) -> np.ndarray:
        """
        Возвращает метки кластеров после обучения.
        """
        return self.labels

    def get_centroids(self) -> np.ndarray:
        """
        Возвращает центры кластеров.
        """
        return self.centroids

    def compute_all_metrics(self, y_true: np.ndarray) -> dict:
        """
        Вычисляет внутренние и внешние метрики кластеризации.

        Параметры:
            y_true (np.ndarray): Истинные метки классов.

        Возвращает:
            dict: Метрики в виде словаря.
        """
        return {
            "Silhouette Score": silhouette_score(self.X, self.labels),
            "Calinski-Harabasz Index": calinski_harabasz_score(self.X, self.labels),
            "Davies-Bouldin Index": davies_bouldin_score(self.X, self.labels),
            "Adjusted Rand Index": adjusted_rand_score(y_true, self.labels),
            "Rand Index": rand_score(y_true, self.labels),
            "Fowlkes-Mallows Index": fowlkes_mallows_score(y_true, self.labels),
            "Mutual Information": mutual_info_score(y_true, self.labels),
            "Homogeneity": homogeneity_score(y_true, self.labels),
            "Completeness": completeness_score(y_true, self.labels),
            "V-measure": v_measure_score(y_true, self.labels),
        }
    
    def visualize_clusters(self) -> None:
        """
        Визуализирует результат кластеризации в пространстве двух главных компонент (PCA).
        """
        if self.labels is None or self.centroids is None:
            raise RuntimeError("Сначала вызовите fit(), чтобы обучить модель.")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        centroids_pca = pca.transform(self.centroids)

        df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_plot["cluster"] = self.labels

        plt.figure(figsize=(8, 6))
        for cluster_id in np.unique(self.labels):
            cluster_points = df_plot[df_plot["cluster"] == cluster_id]
            plt.scatter(cluster_points["PC1"], cluster_points["PC2"], label=f"Кластер {cluster_id}", alpha=0.6)

        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c="red",
            s=25,
            label="Центры кластеров"
        )

        plt.title(f"Кластеризация (k={self.n_clusters}) в пространстве PCA")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


manager = DatasetManager(source="sklearn")
manager.preprocess()
manager.remove_feature("total_phenols")
X_scaled, y_true = manager.get_preprocessed_data()

model = CustomKMeans(n_clusters=3)
model.fit(X_scaled)
model.visualize_clusters()

labels = model.get_labels()
centroids = model.get_centroids()

metrics = model.compute_all_metrics(y_true)
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")
