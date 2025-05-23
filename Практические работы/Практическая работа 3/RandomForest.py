import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from dataset_manager import DatasetManager


class RandomForestModel:
    def __init__(
        self,
        n_estimators: int = 10,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        max_features: Optional[str] = "sqrt",
        random_state: int = 42
    ) -> None:
        """
        Инициализирует классификатор на основе случайного леса.

        Параметры:
            n_estimators (int): Количество деревьев в лесу.
            criterion (str): Критерий для оценки качества разбиения:
                - 'gini': индекс Джини;
                - 'entropy': информация по Шеннону.
            max_depth (Optional[int]): Максимальная глубина деревьев.
            max_features (str): Количество признаков для выбора при разделении:
                - 'sqrt': корень из числа признаков;
                - 'log2': логарифм по основанию 2;
                - int/float: конкретное количество или доля признаков.
            random_state (int): Начальное значение генератора случайных чисел.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Обучает модель случайного леса по предоставленным обучающим данным.

        Параметры:
            X_train (DataFrame | ndarray): Матрица признаков обучающей выборки.
            y_train (Series | ndarray): Вектор истинных меток классов.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

    def predict(
        self, 
        X_test: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
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

    def plot_tree(
        self, 
        tree_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Визуализирует структуру одного дерева из случайного леса.

        Параметры:
            tree_idx (int): Индекс дерева для отображения (по умолчанию 0).
            feature_names (list): Список имён признаков.
            class_names (list): Список имён классов.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")
            
        plt.figure(figsize=(16, 10))
        plot_tree(
            self.model.estimators_[tree_idx],
            filled=True,
            feature_names=feature_names,
            class_names=class_names
        )
        plt.title(f"Дерево №{tree_idx} случайного леса")
        plt.show()

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Возвращает важность признаков в обученной модели.

        Возвращает:
            DataFrame: Таблица с признаками и их важностью, отсортированная по убыванию.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")
            
        return pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

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

    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
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

    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
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

    def calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """
        Вычисляет F1-меру — гармоническое среднее точности и полноты.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Способ усреднения ('macro', 'micro', 'weighted').

        Возвращает:
            float: Значение F1-метрики.
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Строит матрицу ошибок (confusion matrix) по результатам классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            ndarray: Матрица размера [n_classes, n_classes].
        """
        return confusion_matrix(y_true, y_pred)

    def get_metrics_report(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
        """
        Возвращает сводный отчёт по метрикам классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения (macro, micro, weighted).

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

    rf = RandomForestModel(
        n_estimators=5,
        criterion='gini',
        max_depth=5,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    report = rf.get_metrics_report(y_test, y_pred)
    
    print("\nОтчет о метриках классификации:")
    for metric, value in report.items():
        print(f"- {metric}: {value:.4f}")
        
    print("\nМатрица ошибок:")
    print(rf.calculate_confusion_matrix(y_test, y_pred))
    
    print("\nВажность признаков:")
    print(rf.get_feature_importance().to_string(index=False))
    
    rf.plot_tree(
        tree_idx=0,
        feature_names=X_train.columns.tolist(),
        class_names=[str(cls) for cls in rf.classes_]
    )
