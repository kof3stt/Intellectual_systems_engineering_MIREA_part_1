import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset_manager import DatasetManager
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class VotingEnsemble:
    def __init__(
        self,
        estimators: List[Tuple[str, ClassifierMixin]],
        voting: str = "hard",
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        Простая реализация ансамбля на основе голосования.

        Параметры:
            estimators (List[Tuple[str, ClassifierMixin]]):
                Список кортежей вида (имя_модели, модель),
                где модель — объект, реализующий fit() и predict().
            voting (str): Тип голосования:
                - 'hard': большинство голосов (по умолчанию);
                - 'soft': усреднение предсказанных вероятностей (требует, чтобы базовые модели поддерживали predict_proba()).
            weights (Optional[List[float]]): Список весов для моделей при голосовании.
                Если None, все модели считаются равнозначными. Длина списка должна совпадать с числом моделей.
        """
        if voting not in ("hard", "soft"):
            raise ValueError("Параметр voting должен быть 'hard' или 'soft'.")
        if weights is not None and len(weights) != len(estimators):
            raise ValueError("Длина weights должна совпадать с числом estimators.")

        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.classes_: Optional[np.ndarray] = None
        self.fitted_estimators: List[ClassifierMixin] = []

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Обучает каждый базовый классификатор на переданных данных.

        Параметры:
            X_train (DataFrame | ndarray): Матрица признаков обучающей выборки.
            y_train (Series | ndarray): Вектор меток классов обучающей выборки.
        """
        X = np.array(X_train)
        y = np.array(y_train)
        self.classes_ = np.unique(y)
        self.fitted_estimators = []

        for _, estimator in self.estimators:
            model = clone(estimator)
            model.fit(X, y)
            self.fitted_estimators.append(model)

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Предсказывает метки классов для тестовых данных ансамблем.

        Параметры:
            X_test (DataFrame | ndarray): Матрица признаков тестовой выборки.

        Возвращает:
            ndarray: Вектор предсказанных меток классов.

        Исключения:
            RuntimeError: если ансамбль не был обучен (нет fitted_estimators).
        """
        if not self.fitted_estimators:
            raise RuntimeError("Модели не обучены. Вызовите метод fit() перед predict().")

        X = np.array(X_test)
        n_samples = X.shape[0]
        n_models = len(self.fitted_estimators)

        if self.voting == "hard":
            all_preds = np.zeros((n_models, n_samples), dtype=object)
            for idx, model in enumerate(self.fitted_estimators):
                all_preds[idx] = model.predict(X)

            predictions = []
            for j in range(n_samples):
                votes = {}
                for i in range(n_models):
                    label = all_preds[i, j]
                    weight = self.weights[i] if self.weights is not None else 1.0
                    votes[label] = votes.get(label, 0.0) + weight
                predicted_label = max(votes.items(), key=lambda x: x[1])[0]
                predictions.append(predicted_label)

            return np.array(predictions)

        else:
            probas = []
            for model in self.fitted_estimators:
                if not hasattr(model, "predict_proba"):
                    raise RuntimeError(f"Модель {model} не поддерживает predict_proba(), невозможно выполнить soft-голосование.")
                probas.append(model.predict_proba(X))

            avg_proba = np.zeros_like(probas[0])
            for idx, proba in enumerate(probas):
                w = self.weights[idx] if self.weights is not None else 1.0
                avg_proba += w * proba
            avg_proba /= (sum(self.weights) if self.weights is not None else n_models)

            return np.array([self.classes_[np.argmax(row)] for row in avg_proba])

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисляет метрику Accuracy (долю правильных классификаций).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            float: Значение accuracy ∈ [0, 1].
        """
        return accuracy_score(y_true, y_pred)

    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """
        Вычисляет метрику Precision (точность предсказания классов).

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
        Вычисляет метрику Recall (полноту предсказания).

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
            average (str): Стратегия усреднения ('macro', 'micro', 'weighted').

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

    def get_metrics_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro"
    ) -> Dict[str, float]:
        """
        Возвращает сводный отчёт по основным метрикам классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения ('macro', 'micro', 'weighted').

        Возвращает:
            Dict[str, float]: Словарь с метриками:
                - accuracy
                - precision
                - recall
                - f1
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

    knn = ("knn", KNeighborsClassifier(n_neighbors=3))
    dt = ("decision_tree", DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=42))
    rf = ("random_forest", RandomForestClassifier(n_estimators=5, random_state=42))

    voting = VotingEnsemble(
        estimators=[knn, dt, rf],
        voting="hard",
        weights=[1.0, 1.0, 1.0]
    )
    voting.fit(X_train, y_train)

    y_pred = voting.predict(X_test)

    report = voting.get_metrics_report(y_test, y_pred)
    print("\nОтчет о метриках классификации:")
    for metric, value in report.items():
        print(f"- {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(voting.calculate_confusion_matrix(y_test, y_pred))
