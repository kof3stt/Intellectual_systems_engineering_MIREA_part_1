import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from dataset_manager import DatasetManager


class KNNClassifier:
    def __init__(
        self, 
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'minkowski'
    ) -> None:
        """
        Классификатор k-ближайших соседей (KNN).
        
        Параметры:
            n_neighbors (int): Количество соседей (по умолчанию 5).
            weights (str): Стратегия взвешивания:
                - 'uniform': все соседи имеют равный вес
                - 'distance': вес обратно пропорционален расстоянию (по умолчанию 'uniform').
            metric (str): Метрика для расчета расстояний (по умолчанию 'minkowski').
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model: Optional[KNeighborsClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Обучение модели на обучающих данных.
        
        Параметры:
            X_train (DataFrame/ndarray): Матрица признаков обучающей выборки.
            y_train (Series/ndarray): Вектор целевых меток.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

    def predict(
        self, 
        X_test: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Предсказание классов для новых данных.
        
        Параметры:
            X_test (DataFrame/ndarray): Матрица признаков тестовой выборки.
            
        Возвращает:
            ndarray: Массив предсказанных меток.
            
        Исключения:
            RuntimeError: Если модель не обучена.
        """
        if self.model is None:
            raise RuntimeError("Сначала выполните обучение модели (fit()).")
        return self.model.predict(X_test)

    def calculate_accuracy(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: np.ndarray
    ) -> float:
        """
        Вычисление точности (Accuracy).
        
        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            
        Возвращает:
            float: Значение метрики Accuracy ∈ [0, 1].
        """
        return accuracy_score(y_true, y_pred)

    def calculate_precision(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """
        Вычисление точности (Precision).
        
        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения:
                - 'macro': среднее по классам
                - 'micro': глобальное усреднение
                - 'weighted': взвешенное среднее
                
        Возвращает:
            float: Значение метрики Precision.
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_recall(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """
        Вычисление полноты (Recall).

        Параметры:
            y_true (Series или ndarray): Истинные метки классов.
            y_pred (ndarray): Предсказанные моделью метки классов.
            average (str): Способ усреднения:
                - 'macro': среднее значение recall по всем классам;
                - 'micro': глобальная метрика по всем объектам;
                - 'weighted': среднее, взвешенное по количеству объектов в каждом классе.

        Возвращает:
            float: Значение метрики Recall в диапазоне [0, 1].
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_f1(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """
        Вычисление F1-меры (F1-score).

        Параметры:
            y_true (Series или ndarray): Истинные метки классов.
            y_pred (ndarray): Предсказанные моделью метки классов.
            average (str): Способ усреднения:
                - 'macro': F1-score по каждому классу, затем среднее;
                - 'micro': общее число TP, FP и FN;
                - 'weighted': среднее, взвешенное по количеству объектов каждого класса.

        Возвращает:
            float: Значение метрики F1-score в диапазоне [0, 1].
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Построение матрицы ошибок (confusion matrix), показывающей распределение предсказаний модели по классам.

        Параметры:
            y_true (Series или ndarray): Истинные метки классов.
            y_pred (ndarray): Предсказанные моделью метки классов.

        Возвращает:
            ndarray: Квадратная матрица размера [n_classes x n_classes], где
                    строки соответствуют истинным меткам, а столбцы — предсказанным.
        """
        return confusion_matrix(y_true, y_pred)

    def get_metrics_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Генерация сводного отчёта по основным метрикам классификации.

        Параметры:
            y_true (Series или ndarray): Истинные метки классов.
            y_pred (ndarray): Предсказанные моделью метки классов.
            average (str): Способ усреднения для precision, recall и F1:
                - 'macro': по всем классам одинаково,
                - 'micro': глобально по всем примерам,
                - 'weighted': с учётом долей классов в выборке.

        Возвращает:
            Dict[str, float]: Словарь, содержащий значения следующих метрик:
                - 'accuracy'
                - 'precision'
                - 'recall'
                - 'f1'
        """
        return {
            "accuracy": self.calculate_accuracy(y_true, y_pred),
            "precision": self.calculate_precision(y_true, y_pred, average),
            "recall": self.calculate_recall(y_true, y_pred, average),
            "f1": self.calculate_f1(y_true, y_pred, average)
        }


if __name__ == "__main__":
    manager = DatasetManager(source="sklearn")
    manager.preprocess()
    manager.remove_feature("total_phenols")
    manager.split_data(test_size=0.2, stratify=True) 

    X_train, y_train = manager.get_training_data()
    X_test, y_test = manager.get_testing_data()

    knn = KNNClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    report = knn.get_metrics_report(y_test, y_pred)
    print("\nОтчет о метриках классификации:")
    for metric, value in report.items():
        print(f"- {metric}: {value:.4f}")
        
    print("\nМатрица ошибок:")
    print(knn.calculate_confusion_matrix(y_test, y_pred))
