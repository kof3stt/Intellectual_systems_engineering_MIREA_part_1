import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from dataset_manager import DatasetManager


class DecisionTreeModel:
    def __init__(self, criterion: str = "gini", max_depth: Optional[int] = None) -> None:
        """
        Инициализирует классификатор на основе дерева решений.

        Параметры:
            criterion (str): Критерий для оценки качества разбиения:
                - 'gini': индекс Джини;
                - 'entropy': информация по Шеннону.
            max_depth (Optional[int]): Максимально допустимая глубина дерева.
                Если None, дерево строится до исчерпания выборки.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.model: Optional[DecisionTreeClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> None:
        """
        Обучает модель дерева решений по обучающим данным.

        Параметры:
            X_train (DataFrame | ndarray): Матрица признаков обучающей выборки.
            y_train (Series | ndarray): Вектор истинных меток классов.
        """
        self.model = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, random_state=42)
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Предсказывает метки классов для новых объектов.

        Параметры:
            X_test (DataFrame | ndarray): Матрица признаков тестовой выборки.

        Возвращает:
            ndarray: Предсказанные метки классов.

        Исключения:
            RuntimeError: если модель не обучена.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель с помощью fit().")
        return self.model.predict(X_test)

    def plot(self, feature_names: Optional[list] = None, class_names: Optional[list] = None) -> None:
        """
        Визуализирует структуру дерева решений.

        Параметры:
            feature_names (list): Названия признаков (столбцов).
            class_names (list): Названия классов (если есть).
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")
        plt.figure(figsize=(16, 10))
        plot_tree(self.model, filled=True, feature_names=feature_names, class_names=class_names)
        plt.title("Визуализация дерева решений")
        plt.show()

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисляет метрику Accuracy (долю правильных классификаций).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            float: Значение Accuracy в диапазоне [0, 1].
        """
        return accuracy_score(y_true, y_pred)

    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """
        Вычисляет метрику Precision (точность предсказания классов).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Тип усреднения:
                - 'macro', 'micro', 'weighted'.

        Возвращает:
            float: Значение Precision.
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """
        Вычисляет метрику Recall (полноту) — насколько хорошо модель находит положительные примеры.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Способ усреднения ('macro', 'micro', 'weighted').

        Возвращает:
            float: Значение Recall.
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """
        Вычисляет F1-меру — гармоническое среднее между точностью и полнотой.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Тип усреднения ('macro', 'micro', 'weighted').

        Возвращает:
            float: Значение F1.
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Строит матрицу ошибок (confusion matrix).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            ndarray: Матрица размера [n_classes, n_classes].
        """
        return confusion_matrix(y_true, y_pred)

    def get_metrics_report(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
        """
        Генерирует словарь с основными метриками классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Способ усреднения (macro, micro, weighted).

        Возвращает:
            Dict[str, float]: Метрики: accuracy, precision, recall, f1.
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
    manager.balance_classes()

    X_train, y_train = manager.get_training_data()
    X_test, y_test = manager.get_testing_data()

    decision_tree = DecisionTreeModel(criterion='gini', max_depth=None)
    decision_tree.fit(X_train, y_train)
    
    y_pred = decision_tree.predict(X_test)
    
    report = decision_tree.get_metrics_report(y_test, y_pred)
    print("\nОтчет о метриках классификации:")
    for metric, value in report.items():
        print(f"- {metric}: {value:.4f}")
        
    print("\nМатрица ошибок:")
    print(decision_tree.calculate_confusion_matrix(y_test, y_pred))

    decision_tree.plot(feature_names=X_train.columns.tolist(), class_names=[str(cls) for cls in decision_tree.classes_])
