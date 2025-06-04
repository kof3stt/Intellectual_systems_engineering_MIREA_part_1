import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from dataset_manager import DatasetManager


class AdaBoostModel:
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        base_estimator: Optional[DecisionTreeClassifier] = None,
        random_state: int = 42,
    ) -> None:
        """
        Инициализирует классификатор AdaBoost.

        Параметры:
            n_estimators (int): Количество слабых моделей (итераций) в ансамбле.
            learning_rate (float): Коэффициент снижения влияния каждой слабой модели.
            base_estimator (DecisionTreeClassifier, optional): Базовый алгоритм (слабый классификатор).
                Если None, по умолчанию используется дерево решений глубины 1 (stump).
            random_state (int): Начальное значение генератора случайных чисел для воспроизводимости.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else DecisionTreeClassifier(max_depth=1, random_state=random_state)
        )
        self.random_state = random_state
        self.model: Optional[AdaBoostClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Обучает модель AdaBoost по предоставленным обучающим данным.

        Параметры:
            X_train (DataFrame | ndarray): Матрица признаков обучающей выборки.
            y_train (Series | ndarray): Вектор истинных меток классов.
        """
        self.model = AdaBoostClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
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

    def plot_stage(
        self,
        stage_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Визуализирует структуру одного из слабых деревьев ансамбля AdaBoost.

        Параметры:
            stage_idx (int): Индекс слабого классификатора для отображения (0 ≤ idx < n_estimators).
            feature_names (list, optional): Список имён признаков (если доступно).
            class_names (list, optional): Список имён классов (если доступно).
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")
        if stage_idx < 0 or stage_idx >= len(self.model.estimators_):
            raise IndexError(f"stage_idx должно быть от 0 до {len(self.model.estimators_) - 1}.")
        plt.figure(figsize=(16, 10))
        plot_tree(
            self.model.estimators_[stage_idx],
            filled=True,
            feature_names=feature_names,
            class_names=class_names,
        )
        plt.title(f"AdaBoost: дерево №{stage_idx}")
        plt.show()

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Возвращает важность признаков, рассчитанную ансамблем AdaBoost.

        Возвращает:
            DataFrame: Таблица с признаками и их важностью, отсортированная по убыванию.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")
        importances = self.model.feature_importances_
        return (
            pd.DataFrame(
                {"feature": self.model.feature_names_in_, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисляет метрику Accuracy — долю правильно классифицированных объектов.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            float: Значение accuracy ∈ [0, 1].
        """
        return accuracy_score(y_true, y_pred)

    def calculate_precision(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """
        Вычисляет метрику Precision — точность предсказания классов.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения ('macro', 'micro', 'weighted').

        Возвращает:
            float: Значение precision.
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_recall(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """
        Вычисляет метрику Recall — полноту предсказания.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения ('macro', 'micro', 'weighted').

        Возвращает:
            float: Значение recall.
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_f1(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """
        Вычисляет F1-меру — гармоническое среднее точности и полноты.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения ('macro', 'micro', 'weighted').

        Возвращает:
            float: Значение F1-метрики.
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Строит матрицу ошибок (confusion matrix) по результатам классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            ndarray: Матрица размера [n_classes, n_classes].
        """
        return confusion_matrix(y_true, y_pred)

    def get_metrics_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> Dict[str, float]:
        """
        Возвращает сводный отчёт по метрикам классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения ('macro', 'micro', 'weighted').

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

    X_train, y_train = manager.get_training_data()
    X_test, y_test = manager.get_testing_data()

    ada = AdaBoostModel(
        n_estimators=50,
        learning_rate=0.5,
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        random_state=42,
    )
    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)

    report = ada.get_metrics_report(y_test, y_pred)
    print("\nОтчет о метриках классификации:")
    for metric, value in report.items():
        print(f"- {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(ada.calculate_confusion_matrix(y_test, y_pred))

    print("\nВажность признаков:")
    print(ada.get_feature_importance().to_string(index=False))

    ada.plot_stage(
        stage_idx=0,
        feature_names=X_train.columns.tolist(),
        class_names=[str(cls) for cls in ada.classes_],
    )
