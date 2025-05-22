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
from typing import Optional, Union
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from dataset_manager import DatasetManager


class KMeansClustering:
    def __init__(self, X_scaled: pd.DataFrame, n_clusters: int = 3) -> None:
        """
        Инициализирует кластеризатор KMeans.

        Параметры:
            X_scaled (pd.DataFrame): Масштабированные данные.
            n_clusters (int): Число кластеров.
        """
        self.X = X_scaled
        self.n_clusters = n_clusters
        self.model: Optional[KMeans] = None
        self.labels: Optional[np.ndarray] = None
        self.pca_components: Optional[pd.DataFrame] = None

    def fit(self) -> None:
        """
        Обучает модель KMeans и сохраняет метки кластеров.
        """
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        self.labels = self.model.fit_predict(self.X)

    def visualize_clusters(self) -> None:
        """
        Визуализирует результат кластеризации в 2D (через PCA).
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")

        pca = PCA(n_components=2)
        components = pca.fit_transform(self.X)
        df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
        df_pca["cluster"] = self.labels
        self.pca_components = df_pca

        plt.figure(figsize=(8, 6))
        for label in np.unique(self.labels):
            subset = df_pca[df_pca["cluster"] == label]
            plt.scatter(subset["PC1"], subset["PC2"], label=f"Кластер {label}", alpha=0.6)

        centers_2d = pca.transform(self.model.cluster_centers_)
        plt.scatter(
            centers_2d[:, 0], centers_2d[:, 1], c="red", s=25, label="Центры"
        )

        plt.title(f"KMeans: визуализация кластеров (k={self.n_clusters})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compute_silhouette(self) -> float:
        """
        Вычисляет среднее значение силуэт-метрики.

        Возвращает:
            float: Silhouette Score.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return silhouette_score(self.X, self.labels)

    def compute_calinski_harabasz(self) -> float:
        """
        Вычисляет индекс Калински-Харабаза.

        Возвращает:
            float: Calinski-Harabasz Index.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return calinski_harabasz_score(self.X, self.labels)

    def compute_davies_bouldin(self) -> float:
        """
        Вычисляет индекс Дэвиса-Болдина.

        Возвращает:
            float: Davies-Bouldin Index.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return davies_bouldin_score(self.X, self.labels)
    
    def compute_adjusted_rand_index(self, y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет Adjusted Rand Index (ARI).

        Параметры:
            y_true (Series | ndarray): Истинные метки.

        Возвращает:
            float: Adjusted Rand Index.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return adjusted_rand_score(y_true, self.labels)


    def compute_rand_index(self, y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет Rand Index.

        Параметры:
            y_true (Series | ndarray): Истинные метки.

        Возвращает:
            float: Rand Index.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return rand_score(y_true, self.labels)


    def compute_fowlkes_mallows(self, y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет индекс Фаулкса-Мэллоуза.

        Параметры:
            y_true (Series | ndarray): Истинные метки.

        Возвращает:
            float: Fowlkes-Mallows Index.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return fowlkes_mallows_score(y_true, self.labels)


    def compute_mutual_info(self, y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет взаимную информацию (MI).

        Параметры:
            y_true (Series | ndarray): Истинные метки.

        Возвращает:
            float: Mutual Information.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return mutual_info_score(y_true, self.labels)


    def compute_homogeneity(self, y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет метрику однородности (Homogeneity).

        Параметры:
            y_true (Series | ndarray): Истинные метки.

        Возвращает:
            float: Homogeneity Score.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return homogeneity_score(y_true, self.labels)


    def compute_completeness(self, y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет метрику полноты (Completeness).

        Параметры:
            y_true (Series | ndarray): Истинные метки.

        Возвращает:
            float: Completeness Score.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return completeness_score(y_true, self.labels)


    def compute_v_measure(self, y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет V-меру (среднее между Homogeneity и Completeness).

        Параметры:
            y_true (Series | ndarray): Истинные метки.

        Возвращает:
            float: V-Measure Score.
        """
        if self.labels is None:
            raise RuntimeError("Сначала вызовите fit().")
        return v_measure_score(y_true, self.labels)


manager = DatasetManager(source="sklearn")
manager.preprocess()
manager.remove_feature("total_phenols")
X_scaled, y_true = manager.get_preprocessed_data()

clusterer = KMeansClustering(X_scaled, n_clusters=3)
clusterer.fit()
clusterer.visualize_clusters()

# ВНУТРЕННИЕ МЕТРИКИ
silhouette = clusterer.compute_silhouette()
print(f"Silhouette Score: {silhouette:.3f}")

calinski = clusterer.compute_calinski_harabasz()
print(f"Calinski-Harabasz Index: {calinski:.2f}")

davies = clusterer.compute_davies_bouldin()
print(f"Davies-Bouldin Index: {davies:.3f}")

# ВНЕШНИЕ МЕТРИКИ
ari = clusterer.compute_adjusted_rand_index(y_true)
print(f"Adjusted Rand Index (ARI): {ari:.3f}")

ri = clusterer.compute_rand_index(y_true)
print(f"Rand Index (RI): {ri:.3f}")

fmi = clusterer.compute_fowlkes_mallows(y_true)
print(f"Fowlkes-Mallows Index: {fmi:.3f}")

mi = clusterer.compute_mutual_info(y_true)
print(f"Mutual Information: {mi:.3f}")

homogeneity = clusterer.compute_homogeneity(y_true)
print(f"Homogeneity: {homogeneity:.3f}")

completeness = clusterer.compute_completeness(y_true)
print(f"Completeness: {completeness:.3f}")

v_measure = clusterer.compute_v_measure(y_true)
print(f"V-measure: {v_measure:.3f}")
