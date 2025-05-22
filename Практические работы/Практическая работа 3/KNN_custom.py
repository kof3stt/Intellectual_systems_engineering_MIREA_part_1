import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
from collections import Counter
from dataset_manager import DatasetManager


class KNNCustom:
    def __init__(
        self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean"
    ) -> None:
        """
        Кастомная реализация классификатора k-ближайших соседей.

        Параметры:
            n_neighbors (int): Количество соседей (по умолчанию 5).
            weights (str): Стратегия взвешивания:
                - 'uniform': равные веса
                - 'distance': обратно пропорционально расстоянию
            metric (str): Метрика расстояния ('euclidean', 'manhattan', 'cosine')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Сохраняет обучающую выборку в памяти модели.

        Параметры:
            X_train (DataFrame или ndarray): Матрица признаков обучающих объектов.
            y_train (Series или ndarray): Вектор меток классов.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.classes_ = np.unique(y_train)

    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Вычисляет расстояние между двумя точками в заданной метрике.

        Параметры:
            a (ndarray): Первая точка.
            b (ndarray): Вторая точка.

        Возвращает:
            float: Расстояние между a и b в соответствии с выбранной метрикой ('euclidean', 'manhattan', 'cosine').

        Исключения:
            ValueError: Если указана неподдерживаемая метрика.
        """
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b))
        elif self.metric == "cosine":
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Предсказывает метки классов для объектов тестовой выборки.

        Параметры:
            X_test (DataFrame или ndarray): Матрица признаков тестовых объектов.

        Возвращает:
            ndarray: Вектор предсказанных меток.

        Исключения:
            RuntimeError: Если модель не была обучена методом fit().
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        X_test = np.array(X_test)
        predictions = []

        for x in X_test:
            distances = [
                self._calculate_distance(x, x_train) for x_train in self.X_train
            ]

            k_indices = np.argsort(distances)[: self.n_neighbors]
            k_labels = self.y_train[k_indices]
            k_distances = np.array(distances)[k_indices]

            if self.weights == "distance":
                weights = 1 / (k_distances + 1e-8)
            else:
                weights = np.ones_like(k_distances)

            counter = Counter()
            for label, weight in zip(k_labels, weights):
                counter[label] += weight

            predictions.append(max(counter, key=counter.get))

        return np.array(predictions)

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисляет точность (accuracy) классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            float: Доля правильных ответов от общего числа наблюдений.
        """
        return np.sum(y_true == y_pred) / len(y_true)

    def calculate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Построение матрицы ошибок (confusion matrix).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            ndarray: Матрица размера [n_classes x n_classes], где строки — истинные классы,
                    столбцы — предсказанные классы.
        """
        n_classes = len(self.classes_)
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for true, pred in zip(y_true, y_pred):
            matrix[true, pred] += 1

        return matrix

    def _calculate_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Вычисляет метрики precision, recall и F1 для каждого класса отдельно.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            Dict[int, Dict[str, float]]: Словарь, в котором для каждого класса содержатся:
                - precision
                - recall
                - f1
                - support (количество наблюдений данного класса)
        """
        matrix = self.calculate_confusion_matrix(y_true, y_pred)
        metrics = {}

        for i, class_label in enumerate(self.classes_):
            tp = matrix[i, i]
            fp = np.sum(matrix[:, i]) - tp
            fn = np.sum(matrix[i, :]) - tp
            tn = np.sum(matrix) - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[class_label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }

        return metrics

    def calculate_precision(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """
        Вычисляет среднюю точность (precision) по классам.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Метод усреднения:
                - 'macro': равное среднее по всем классам,
                - 'weighted': среднее с учетом поддержки (support),
                - 'micro': глобальная точность по всем классам.

        Возвращает:
            float: Значение метрики precision.
        """
        metrics = self._calculate_class_metrics(y_true, y_pred)
        precisions = [m["precision"] for m in metrics.values()]
        supports = [m["support"] for m in metrics.values()]

        if average == "macro":
            return np.mean(precisions)
        elif average == "weighted":
            return np.average(precisions, weights=supports)
        elif average == "micro":
            matrix = self.calculate_confusion_matrix(y_true, y_pred)
            tp = np.sum(np.diag(matrix))
            fp = np.sum(matrix, axis=0) - np.diag(matrix)
            return tp / (tp + np.sum(fp))
        else:
            raise ValueError("Неподдерживаемый тип усреднения")

    def calculate_recall(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """
        Вычисляет среднюю полноту (recall) по классам.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Метод усреднения:
                - 'macro': равное среднее по всем классам,
                - 'weighted': среднее с учетом поддержки (support),
                - 'micro': глобальная полнота по всем классам.

        Возвращает:
            float: Значение метрики recall.
        """
        metrics = self._calculate_class_metrics(y_true, y_pred)
        recalls = [m["recall"] for m in metrics.values()]
        supports = [m["support"] for m in metrics.values()]

        if average == "macro":
            return np.mean(recalls)
        elif average == "weighted":
            return np.average(recalls, weights=supports)
        elif average == "micro":
            matrix = self.calculate_confusion_matrix(y_true, y_pred)
            tp = np.sum(np.diag(matrix))
            fn = np.sum(matrix, axis=1) - np.diag(matrix)
            return tp / (tp + np.sum(fn))
        else:
            raise ValueError("Неподдерживаемый тип усреднения")

    def calculate_f1(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """
        Вычисляет среднее значение F1-меры по классам.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Метод усреднения ('macro', 'weighted', 'micro').

        Возвращает:
            float: Значение F1-меры.
        """
        precision = self.calculate_precision(y_true, y_pred, average)
        recall = self.calculate_recall(y_true, y_pred, average)
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

    def get_metrics_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> Dict[str, float]:
        """
        Генерирует сводный отчёт по метрикам классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Тип усреднения для precision, recall и F1 ('macro', 'weighted', 'micro').

        Возвращает:
            Dict[str, float]: Словарь с метриками:
                - 'accuracy'
                - 'precision'
                - 'recall'
                - 'f1'
        """
        return {
            "accuracy": self.calculate_accuracy(y_true, y_pred),
            "precision": self.calculate_precision(y_true, y_pred, average),
            "recall": self.calculate_recall(y_true, y_pred, average),
            "f1": self.calculate_f1(y_true, y_pred, average),
        }


if __name__ == "__main__":
    manager = DatasetManager(source="sklearn")
    manager.preprocess()
    manager.remove_feature("total_phenols")
    manager.split_data(test_size=0.2, stratify=True)

    X_train, y_train = manager.get_training_data()
    X_test, y_test = manager.get_testing_data()

    knn = KNNCustom(n_neighbors=3, weights="distance", metric="euclidean")
    knn.fit(X_train.values, y_train.values)

    y_pred = knn.predict(X_test.values)

    report = knn.get_metrics_report(y_test.values, y_pred)
    print("\nОтчет о метриках классификации:")
    for metric, value in report.items():
        print(f"- {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(knn.calculate_confusion_matrix(y_test.values, y_pred))
