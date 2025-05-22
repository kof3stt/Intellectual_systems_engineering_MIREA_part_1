import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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


class DBSCANClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Кластеризация методом DBSCAN.

        Параметры:
            eps (float): Радиус ε-окрестности.
            min_samples (int): Минимальное количество точек в ε-окрестности.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels = None
        self.X = None

    def fit(self, X: pd.DataFrame) -> None:
        """
        Обучает DBSCAN и сохраняет метки кластеров.
        """
        self.X = X
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(X)

    def visualize_clusters(self) -> None:
        """
        Визуализирует кластеры в 2D-пространстве (PCA).
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df["cluster"] = self.labels

        plt.figure(figsize=(8, 6))
        for label in np.unique(self.labels):
            subset = df[df["cluster"] == label]
            plt.scatter(
                subset["PC1"],
                subset["PC2"],
                label=f"Кластер {label}" if label != -1 else "Шум",
                alpha=0.6,
            )
        plt.title("Кластеры, найденные DBSCAN")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compute_metrics(self, y_true: np.ndarray) -> dict:
        """
        Вычисляет внутренние и внешние метрики (если есть метки).

        Параметры:
            y_true (np.ndarray): Истинные метки классов.

        Возвращает:
            dict: Метрики кластеризации.
        """
        labels = self.labels
        valid_mask = labels != -1
        X_valid = self.X[valid_mask]
        labels_valid = labels[valid_mask]

        metrics = {
            "Silhouette Score": silhouette_score(X_valid, labels_valid)
            if len(set(labels_valid)) > 1
            else -1,
            "Calinski-Harabasz Index": calinski_harabasz_score(X_valid, labels_valid)
            if len(set(labels_valid)) > 1
            else -1,
            "Davies-Bouldin Index": davies_bouldin_score(X_valid, labels_valid)
            if len(set(labels_valid)) > 1
            else -1,
        }

        if y_true is not None:
            metrics.update(
                {
                    "Adjusted Rand Index": adjusted_rand_score(y_true, labels),
                    "Rand Index": rand_score(y_true, labels),
                    "Fowlkes-Mallows Index": fowlkes_mallows_score(y_true, labels),
                    "Mutual Information": mutual_info_score(y_true, labels),
                    "Homogeneity": homogeneity_score(y_true, labels),
                    "Completeness": completeness_score(y_true, labels),
                    "V-measure": v_measure_score(y_true, labels),
                }
            )

        return metrics


manager = DatasetManager(source="sklearn")
manager.preprocess()
manager.remove_feature('total_phenoles')
X_scaled, y_true = manager.get_preprocessed_data()

dbscan = DBSCANClustering(eps=2.46, min_samples=12)
dbscan.fit(X_scaled)
dbscan.visualize_clusters()

metrics = dbscan.compute_metrics(y_true)

for name, score in metrics.items():
    print(f"{name}: {score:.3f}" if score != -1 else f"{name}: недостаточно кластеров")
