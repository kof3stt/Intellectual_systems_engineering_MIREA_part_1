import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
from dataset_manager import DatasetManager


class RegressionSklearn:
    """
    Класс для линейной регрессии (OLS) с использованием sklearn.

    Атрибуты:
        model (Optional[LinearRegression]): экземпляр обученной модели LinearRegression.
        feature_names (List[str]): список имён признаков, использованных при обучении.
        fitted (bool): флаг, указывающий, что модель обучена.

    Методы:
        fit(X_train, y_train) -> None:
            Обучает модель LinearRegression на тренировочных данных.
        evaluate(X_test, y_test) -> float:
            Вычисляет RMSE на тестовой выборке.
        plot_predicted_vs_actual(X, y, figsize=(8,6)) -> None:
            Строит scatter-диаграмму: предсказанные значения vs фактические.
    """

    def __init__(self) -> None:
        self.model: Optional[LinearRegression] = None
        self.feature_names: List[str] = []
        self.fitted: bool = False

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        """
        Обучает линейную регрессию (LinearRegression) на тренировочных данных.

        Параметры:
            X_train (DataFrame): DataFrame с признаками для обучения.
            y_train (Series): Series с целевой переменной для обучения.

        Исключения:
            ValueError: если X_train не DataFrame или y_train не Series,
                        либо их длины не совпадают.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train должен быть pandas.DataFrame")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train должен быть pandas.Series")
        if len(X_train) != len(y_train):
            raise ValueError("X_train и y_train должны быть одной длины")

        self.feature_names = list(X_train.columns)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.fitted = True

    def evaluate(self, X_test: DataFrame, y_test: Series) -> float:
        """
        Вычисляет RMSE (root mean squared error) на тестовой выборке.

        Параметры:
            X_test (DataFrame): DataFrame с признаками тестовой выборки.
            y_test (Series): Series с целевой переменной тестовой выборки.

        Возвращает:
            float: Значение RMSE на тестовых данных.

        Исключения:
            RuntimeError: если модель не обучена.
            ValueError: если X_test/y_test не тех типов или длины не совпадают.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test должен быть pandas.DataFrame")
        if not isinstance(y_test, pd.Series):
            raise ValueError("y_test должен быть pandas.Series")
        if len(X_test) != len(y_test):
            raise ValueError("X_test и y_test должны быть одной длины")

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        mdae = median_absolute_error(y_test, y_pred)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mdae": mdae,
            "r2": r2,
            "evs": evs,
        }

    def plot_predicted_vs_actual(
        self, X: DataFrame, y: Series, figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Строит scatter-диаграмму: предсказанные значения vs фактические.

        Параметры:
            X (DataFrame): DataFrame признаков для предсказания.
            y (Series): Series фактических значений целевой переменной.
            figsize (Tuple[int, int]): Размер фигуры (ширина, высота) в дюймах.

        Исключения:
            RuntimeError: если модель не обучена.
            ValueError: если X/y не тех типов или длины не совпадают.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X должен быть pandas.DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y должен быть pandas.Series")
        if len(X) != len(y):
            raise ValueError("X и y должны быть одной длины")

        preds = self.model.predict(X)
        plt.figure(figsize=figsize)
        plt.scatter(y.values, preds, alpha=0.6, edgecolor="k")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2)
        plt.title("Предсказанные значения vs фактические")
        plt.xlabel("Фактические значения")
        plt.ylabel("Предсказанные значения")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(
        self,
        figsize: Tuple[int, int] = (12, 8),
        importance_type: str = "coefficient",
        color: str = "skyblue",
    ) -> None:
        """
        Визуализирует важность признаков на основе коэффициентов линейной регрессии.

        Параметры:
            figsize (Tuple[int, int]): Размер графика (ширина, высота).
            importance_type (str): Тип важности:
                'coefficient' - исходные коэффициенты
                'absolute' - абсолютные значения коэффициентов
                'normalized' - нормализованные абсолютные значения (сумма=100%)
            color (str): Цвет столбцов диаграммы.
        """

        features = self.model.feature_names_in_
        coefs = self.model.coef_

        if importance_type == "coefficient":
            importance = coefs
            ylabel = "Коэффициент"
        elif importance_type == "absolute":
            importance = np.abs(coefs)
            ylabel = "Абсолютное значение коэффициента"
        elif importance_type == "normalized":
            importance = 100 * np.abs(coefs) / np.sum(np.abs(coefs))
            ylabel = "Относительная важность (%)"
        else:
            raise ValueError(
                f"Неизвестный тип важности: {importance_type}. "
                "Допустимые значения: 'coefficient', 'absolute', 'normalized'"
            )

        importance_df = pd.DataFrame({"feature": features, "importance": importance})

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=True if importance_type == "coefficient" else False,
        )

        plt.figure(figsize=figsize)
        bars = plt.barh(
            importance_df["feature"],
            importance_df["importance"],
            color=color,
            edgecolor="black",
        )

        for bar in bars:
            width = bar.get_width()
            plt.annotate(
                f"{width:.4f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
            )

        plt.xlabel(ylabel)
        plt.title("Важность признаков в линейной регрессии")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    manager = DatasetManager(
        csv_path="Student_Performance.csv", target_column="Performance Index"
    )

    manager.preprocess()
    manager.split_data()
    X_train, y_train = manager.get_training_data()
    X_test, y_test = manager.get_testing_data()

    reg = RegressionSklearn()
    reg.fit(X_train, y_train)
    metrics = reg.evaluate(X_test, y_test)

    reg.plot_predicted_vs_actual(X_test, y_test)
    reg.plot_feature_importance(importance_type="coefficient")

    print("Результаты оценки модели:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.2f}")
