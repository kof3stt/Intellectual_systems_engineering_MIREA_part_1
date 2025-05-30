import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
from pandas import DataFrame, Series
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
from dataset_manager import DatasetManager


class LassoRegressionSklearn:
    def __init__(self) -> None:
        """
        Инициализирует пустые атрибуты модели.
        """
        self.model: Optional[Lasso] = None
        self.feature_names: List[str] = []
        self.fitted: bool = False

    def fit(
        self,
        X_train: DataFrame,
        y_train: Series,
        alpha: float = 1.0,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Обучает модель Lasso (sklearn.linear_model.Lasso) на тренировочных данных.

        Параметры:
            X_train (DataFrame): DataFrame с признаками для обучения.
            y_train (Series): Series с целевой переменной для обучения.
            alpha (float): Коэффициент регуляризации L1 (по умолчанию 1.0).
            max_iter (int): Максимальное число итераций для оптимизации (по умолчанию 1000).
            random_state (Optional[int]): Сид для воспроизводимости (по умолчанию None).

        Исключения:
            ValueError: если X_train не pandas.DataFrame или y_train не pandas.Series,
                        либо их длины не совпадают.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train должен быть pandas.DataFrame")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train должен быть pandas.Series")
        if len(X_train) != len(y_train):
            raise ValueError("X_train и y_train должны быть одной длины")

        self.feature_names = list(X_train.columns)
        self.model = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
        self.model.fit(X_train, y_train)
        self.fitted = True

    def evaluate(self, X_test: DataFrame, y_test: Series) -> Dict[str, float]:
        """
        Вычисляет основные метрики качества на тестовой выборке:
            - MSE (mean squared error)
            - RMSE (root mean squared error)
            - MAE (mean absolute error)
            - MdAE (median absolute error)
            - R2 score (коэффициент детерминации)
            - EVS (explained variance score)

        Параметры:
            X_test (DataFrame): DataFrame с признаками тестовой выборки.
            y_test (Series): Series с целевой переменной тестовой выборки.

        Возвращает:
            Dict[str, float]: Словарь с ключами ["mse", "rmse", "mae", "mdae", "r2", "evs"].

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
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mdae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

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
            X (DataFrame): DataFrame с признаками для предсказания.
            y (Series): Series с фактическими значениями целевой переменной.
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

        y_pred = self.model.predict(X)
        plt.figure(figsize=figsize)
        plt.scatter(y.values, y_pred, alpha=0.6, edgecolor="k")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2)
        plt.title("Предсказанные значения vs фактические (Lasso)")
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
        Визуализирует важность признаков на основе коэффициентов Lasso-модели.

        Параметры:
            figsize (Tuple[int, int]): Размер графика (ширина, высота).
            importance_type (str): Тип важности:
                - 'coefficient'  : исходные коэффициенты (с учётом знака)
                - 'absolute'     : абсолютные значения коэффициентов
                - 'normalized'   : относительные абсолютные (в сумме = 100%)
            color (str): Цвет столбцов диаграммы.

        Исключения:
            RuntimeError: если модель не обучена.
            ValueError: если importance_type задан неверно.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")

        coefs = self.model.coef_
        features = np.array(self.feature_names)

        if importance_type == "coefficient":
            importance = coefs
            ylabel = "Коэффициент"
        elif importance_type == "absolute":
            importance = np.abs(coefs)
            ylabel = "Абсолютное значение коэффициента"
        elif importance_type == "normalized":
            abs_vals = np.abs(coefs)
            if abs_vals.sum() == 0:
                importance = abs_vals
            else:
                importance = 100 * abs_vals / abs_vals.sum()
            ylabel = "Относительная важность (%)"
        else:
            raise ValueError(
                f"Неизвестный тип важности: {importance_type}. "
                "Допустимые: 'coefficient', 'absolute', 'normalized'"
            )

        importance_df = pd.DataFrame({"feature": features, "importance": importance})

        ascending = True if importance_type == "coefficient" else False
        importance_df = importance_df.sort_values(by="importance", ascending=ascending)

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
        plt.title("Важность признаков в Lasso-регрессии")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    manager = DatasetManager(
        csv_path="Student_Performance.csv", target_column="Performance Index"
    )

    manager.preprocess()
    manager.split_data(test_size=0.2, random_state=42)
    X_train, y_train = manager.get_training_data()
    X_test, y_test = manager.get_testing_data()

    lasso_reg = LassoRegressionSklearn()
    lasso_reg.fit(X_train, y_train, alpha=0.1, max_iter=1000, random_state=42)

    metrics = lasso_reg.evaluate(X_test, y_test)
    print("Результаты оценки Lasso-модели:")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    lasso_reg.plot_predicted_vs_actual(X_test, y_test)

    lasso_reg.plot_feature_importance(importance_type="absolute", color="teal")
