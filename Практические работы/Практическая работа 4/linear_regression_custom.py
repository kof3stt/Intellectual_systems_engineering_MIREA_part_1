import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional, Dict, Union
from dataset_manager import DatasetManager


class CustomLinearRegression:
    """
    Кастомная реализация многомерной линейной регрессии с использованием стохастического градиентного спуска.

    Параметры:
        learning_rate (float): Скорость обучения (по умолчанию 0.01)
        epochs (int): Количество эпох обучения (по умолчанию 1000)
        batch_size (int): Размер батча для мини-пакетного обучения (по умолчанию 32)
        random_state (int): Seed для воспроизводимости (по умолчанию None)
        early_stopping (bool): Останавливать ли обучение при ухудшении ошибки (по умолчанию True)
        patience (int): Количество эпох для ранней остановки (по умолчанию 10)

    Атрибуты:
        weights (np.ndarray): Веса модели (включая смещение)
        feature_names (list): Имена признаков
        errors (list): История ошибок в процессе обучения
        best_weights (np.ndarray): Лучшие веса (при использовании ранней остановки)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 32,
        random_state: Optional[int] = None,
        early_stopping: bool = True,
        patience: int = 10,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience

        if random_state is not None:
            np.random.seed(random_state)

        self.weights = None
        self.feature_names = None
        self.errors = []
        self.best_weights = None
        self.best_error = float("inf")

    def _initialize_weights(self, n_features: int) -> None:
        """Инициализирует веса случайными малыми значениями"""
        self.weights = np.random.randn(n_features + 1) * 0.01

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Добавляет столбец единиц для смещения (bias)"""
        return np.c_[np.ones(X.shape[0]), X]

    def _calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Вычисляет среднеквадратичную ошибку (MSE)"""
        predictions = X.dot(self.weights)
        return np.mean((predictions - y) ** 2)

    def _gradient_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """Выполняет один шаг градиентного спуска"""
        predictions = X_batch.dot(self.weights)
        error = predictions - y_batch
        gradient = X_batch.T.dot(error) / len(X_batch)
        self.weights -= self.learning_rate * gradient

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> None:
        """
        Обучает модель линейной регрессии на предоставленных данных.

        Параметры:
            X (Union[np.ndarray, pd.DataFrame]): Матрица признаков
            y (Union[np.ndarray, pd.Series]): Вектор целевых значений

        Исключения:
            ValueError: Если размеры X и y не совпадают
        """
        if len(X) != len(y):
            raise ValueError("Размеры X и y должны совпадать")

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X = np.array(X)
        y = np.array(y)

        self._initialize_weights(X.shape[1])
        X_with_bias = self._add_bias(X)

        no_improvement_count = 0
        self.errors = []
        self.best_weights = self.weights.copy()
        self.best_error = float("inf")

        for _ in range(self.epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]

            for i in range(0, len(X), self.batch_size):
                end = min(i + self.batch_size, len(X))
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]

                self._gradient_step(X_batch, y_batch)

            current_error = self._calculate_loss(X_with_bias, y)
            self.errors.append(current_error)

            if self.early_stopping:
                if current_error < self.best_error:
                    self.best_error = current_error
                    self.best_weights = self.weights.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.patience:
                    self.weights = self.best_weights
                    break

        if not self.early_stopping:
            self.best_weights = self.weights.copy()

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Предсказывает значения для новых данных.

        Параметры:
            X (Union[np.ndarray, pd.DataFrame]): Матрица признаков для предсказания

        Возвращает:
            np.ndarray: Предсказанные значения

        Исключения:
            RuntimeError: Если модель не обучена
        """
        if self.weights is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")

        X = np.array(X)
        X_with_bias = self._add_bias(X)
        return X_with_bias.dot(self.weights)

    def evaluate(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Оценивает модель на тестовых данных и возвращает комплекс метрик регрессии.

        Параметры:
            X (Union[np.ndarray, pd.DataFrame]): Матрица признаков
            y (Union[np.ndarray, pd.Series]): Вектор целевых значений

        Возвращает:
            Dict[str, float]: Словарь с метриками:
                - 'mse': Средняя квадратичная ошибка
                - 'rmse': Корень из средней квадратичной ошибки
                - 'mae': Средняя абсолютная ошибка
                - 'r2': Коэффициент детерминации (R²)
                - 'explained_variance': Объясненная дисперсия
        """
        y_pred = self.predict(X)
        y_true = np.array(y)

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_total)

        explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "explained_variance": explained_variance,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Возвращает важность признаков на основе абсолютных значений весов.

        Возвращает:
            Dict[str, float]: Словарь {имя_признака: важность}

        Исключения:
            RuntimeError: Если модель не обучена
        """
        if self.weights is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")

        feature_weights = self.weights[1:]

        abs_weights = np.abs(feature_weights)
        normalized_importance = 100 * abs_weights / np.sum(abs_weights)

        return dict(zip(self.feature_names, normalized_importance))

    def plot_learning_curve(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Визуализирует кривую обучения (ошибка vs эпохи).

        Исключения:
            RuntimeError: Если модель не обучена
        """
        if not self.errors:
            raise RuntimeError("История ошибок недоступна. Сначала обучите модель")

        plt.figure(figsize=figsize)
        plt.plot(self.errors)
        plt.title("Кривая обучения")
        plt.xlabel("Эпоха")
        plt.ylabel("MSE")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_predictions_vs_actual(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        figsize: Tuple[int, int] = (8, 6),
    ) -> None:
        """
        Визуализирует предсказанные значения vs фактические значения.

        Исключения:
            RuntimeError: Если модель не обучена
        """
        y_pred = self.predict(X)
        y_true = np.array(y)

        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2)

        plt.title("Предсказанные vs Фактические значения")
        plt.xlabel("Фактические значения")
        plt.ylabel("Предсказанные значения")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_feature_importance(
        self, figsize: Tuple[int, int] = (10, 6), top_n: Optional[int] = None
    ) -> None:
        """
        Визуализирует важность признаков на основе абсолютных значений весов.

        Параметры:
            top_n (int): Количество топ-признаков для отображения (по умолчанию все)

        Исключения:
            RuntimeError: Если модель не обучена
        """
        importance = self.get_feature_importance()

        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        if top_n is not None:
            sorted_importance = sorted_importance[:top_n]

        features, importances = zip(*sorted_importance)

        plt.figure(figsize=figsize)
        plt.barh(features, importances, color="skyblue")
        plt.title("Важность признаков")
        plt.xlabel("Важность (%)")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_coefficients(self) -> Dict[str, float]:
        """
        Возвращает коэффициенты модели для каждого признака.

        Возвращает:
            Dict[str, float]: Словарь {имя_признака: коэффициент}

        Исключения:
            RuntimeError: Если модель не обучена
        """
        if self.weights is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")

        feature_weights = self.weights[1:]
        return dict(zip(self.feature_names, feature_weights))


manager = DatasetManager(
    csv_path="Student_Performance.csv", target_column="Performance Index"
)

manager.preprocess()
manager.split_data()

X_train, y_train = manager.get_training_data()
X_test, y_test = manager.get_testing_data()

model = CustomLinearRegression(
    learning_rate=0.01,
    epochs=1000,
    batch_size=32,
    random_state=42,
    early_stopping=True,
    patience=20,
)
model.fit(X_train, y_train)

test_metrics = model.evaluate(X_test, y_test)

print("\nМетрики на тестовой выборке:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

model.plot_learning_curve()
model.plot_predictions_vs_actual(X_test, y_test)
model.plot_feature_importance(top_n=10)

coefficients = model.get_coefficients()
print("\nКоэффициенты модели:")
for feature, coef in coefficients.items():
    print(f"{feature}: {coef:.4f}")
