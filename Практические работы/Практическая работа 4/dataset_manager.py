import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Tuple
from pandas import DataFrame, Series


class DatasetManager:
    """
    Менеджер датасета для задач регрессии: загрузка из CSV, анализ, предобработка и визуализация.

    Параметры:
        csv_path (str): Путь к CSV-файлу с данными.
        target_column (str): Название столбца с целевой переменной.
    Атрибуты:
        df (DataFrame): Полный исходный DataFrame.
        features (DataFrame): DataFrame с только признаками (без целевой переменной).
        target (Series): Серия с целевой переменной.
        scaled_features (DataFrame): DataFrame с масштабированными признаками.
        stats (Dict[str, DataFrame]): Словарь рассчитанных статистик.
    """

    def __init__(self, csv_path: str, target_column: str) -> None:
        self.csv_path: str = csv_path
        self.target_column: str = target_column
        self.df: Optional[DataFrame] = None
        self.features: Optional[DataFrame] = None
        self.target: Optional[Series] = None
        self.scaled_features: Optional[DataFrame] = None
        self.stats: Dict[str, DataFrame] = {}

        self._load_data()
        self._dataset_custom_preprocess()
        self._extract_features_target()

        if self.csv_path == 'housing_dataset.csv':
            self.remove_feature('Location')
            self.remove_feature('City')

    def _dataset_custom_preprocess(self):
        if self.csv_path == 'Student_Performance.csv':
            self.df["Extracurricular Activities"] = self.df["Extracurricular Activities"].map({"Yes":1 , "No":0})
        else:
            cols_to_check = [col for col in self.df.columns if col != "No. of Bedrooms"]
            mask_no_nines = (self.df[cols_to_check] != 9).all(axis=1)
            self.df = self.df[mask_no_nines].reset_index(drop=True)

    def _load_data(self) -> None:
        """
        Загружает данные из CSV-файла в self.df.

        Выбрасывает:
            FileNotFoundError: если файл по пути csv_path не найден.
            pd.errors.EmptyDataError: если CSV пустой или неверный формат.
        """
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Файл не найден по пути: {self.csv_path}") from e
        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError(
                f"Пустой или некорректный CSV: {self.csv_path}"
            ) from e

        print(
            f"Данные загружены: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов"
        )

    def _extract_features_target(self) -> None:
        """
        Разделяет DataFrame на признаки и целевую переменную.

        После выполнения:
            - self.features содержит все столбцы, кроме target_column.
            - self.target содержит Series с данными целевой переменной.
        Выбрасывает:
            ValueError: если target_column отсутствует в self.df.
        """
        if self.df is None:
            raise RuntimeError("Данные не загружены. Сначала вызовите _load_data().")

        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Целевая переменная '{self.target_column}' не найдена в данных."
            )
        self.target = self.df[self.target_column].copy()
        self.features = self.df.drop(columns=[self.target_column]).copy()

    def compute_basic_statistics(self) -> Dict[str, DataFrame]:
        """
        Вычисляет базовые статистики по признакам и сохраняет их в self.stats.

        Сохраняются:
            - "describe": описательные статистики (mean, std, min, max, квартили) для каждого признака.
            - "correlation_matrix": матрица корреляций между признаками.
            - "target_distribution": описательные статистики целевой переменной.

        Возвращает:
            Dict[str, DataFrame]: Словарь со статистиками.
        Выбрасывает:
            RuntimeError: если признаки или целевая переменная не выделены.
        """
        if self.features is None or self.target is None:
            raise RuntimeError(
                "Признаки или целевая переменная не выделены. Вызовите _extract_features_target()."
            )

        desc_features = self.features.describe().T
        self.stats["describe"] = desc_features

        corr_features = self.features.corr()
        self.stats["correlation_matrix"] = corr_features

        desc_target = self.target.describe().to_frame(name="target_stats")
        self.stats["target_distribution"] = desc_target

        return self.stats

    def preprocess(
        self,
        drop_duplicates: bool = True,
        drop_outliers: bool = True,
        z_thresh: float = 3.0,
    ) -> None:
        """
        Предобработка данных:
            1. Удаление дубликатов (если drop_duplicates=True).
            2. Удаление выбросов по Z-оценке (если drop_outliers=True).
            3. Масштабирование признаков StandardScaler.

        Параметры:
            drop_duplicates (bool): Удалять ли полные дубликаты строк.
            drop_outliers (bool): Удалять ли выбросы по Z-оценке для каждого признака.
            z_thresh (float): Порог Z-оценки; строки, у которых |z_score| > z_thresh хотя бы по одному признаку, удаляются.

        После выполнения:
            - self.df обновляется без дубликатов и выбросов.
            - self.features и self.target обновляются согласно очищенному DataFrame.
            - self.scaled_features заполняется DataFrame с масштабированными признаками.
        Выбрасывает:
            RuntimeError: если self.df не инициализирован.
        """
        if self.df is None:
            raise RuntimeError("Данные не загружены. Сначала вызовите _load_data().")

        df_proc = self.df.copy()

        if drop_duplicates:
            before_dupes = df_proc.shape[0]
            df_proc = df_proc.drop_duplicates().reset_index(drop=True)
            after_dupes = df_proc.shape[0]
            print(f"Удалено дубликатов: {before_dupes - after_dupes}")

        if drop_outliers:
            df_no_target = df_proc.drop(columns=[self.target_column], errors="ignore")
            means = df_no_target.mean()
            stds = df_no_target.std(ddof=0)
            z_scores = (df_no_target - means) / stds
            mask = (z_scores.abs() <= z_thresh).all(axis=1)
            before_out = df_proc.shape[0]
            df_proc = df_proc[mask].reset_index(drop=True)
            after_out = df_proc.shape[0]
            print(f"Удалено выбросов: {before_out - after_out}")

        self.df = df_proc
        self.target = df_proc[self.target_column].copy()
        self.features = df_proc.drop(columns=[self.target_column]).copy()

        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(self.features)
        self.scaled_features = pd.DataFrame(
            scaled_array, columns=self.features.columns, index=self.features.index
        )

        print(
            "Предобработка завершена: дубликаты и выбросы удалены (если указано), признаки масштабированы."
        )

    def visualize_distributions(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Строит гистограммы распределений каждого признака (до масштабирования).

        Параметры:
            figsize (Tuple[int, int]): Размер фигуры (ширина, высота) в дюймах.
        """
        if self.features is None:
            raise RuntimeError(
                "Признаки не выделены. Сначала вызовите _extract_features_target()."
            )

        n = len(self.features.columns)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(self.features.columns):
            axes[i].hist(self.features[col], bins=15, edgecolor="black")
            axes[i].set_title(col)
        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_correlation_heatmap(
        self, figsize: Tuple[int, int] = (18, 12)
    ) -> None:
        """
        Строит тепловую карту корреляций между признаками с помощью seaborn.

        Параметры:
            figsize (Tuple[int, int]): Размер фигуры (ширина, высота) в дюймах.
        Выбрасывает:
            RuntimeError: если self.features не инициализирован.
        """
        if self.features is None:
            raise RuntimeError(
                "Признаки не выделены. Вызовите _extract_features_target()."
            )

        plt.figure(figsize=figsize)
        sns.heatmap(
            self.features.corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.7},
        )
        plt.title("Матрица корреляции признаков")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Разделяет данные на обучающую и тестовую выборки.

        Параметры:
            test_size (float): Доля тестовой выборки (по умолчанию 0.2).
            random_state (int): Seed для воспроизводимости (по умолчанию 42).
        Выбрасывает:
            RuntimeError: если self.scaled_features или self.target не инициализированы.
        """
        if self.scaled_features is None or self.target is None:
            raise RuntimeError("Данные не предобработаны. Вызовите preprocess().")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.scaled_features,
            self.target,
            test_size=test_size,
            random_state=random_state,
        )

        print(
            f"Данные разделены:\n"
            f"- Обучающая выборка: {self.X_train.shape[0]} образцов\n"
            f"- Тестовая выборка: {self.X_test.shape[0]} образцов"
        )

    def get_training_data(self) -> Tuple[DataFrame, Series]:
        """
        Возвращает обучающие данные (признаки и целевую переменную).

        Возвращает:
            Tuple[DataFrame, Series]: (X_train, y_train).
        Выбрасывает:
            RuntimeError: если split_data не был вызван.
        """
        if not hasattr(self, "X_train") or not hasattr(self, "y_train"):
            raise RuntimeError("Данные не разделены. Вызовите split_data().")
        return self.X_train, self.y_train

    def get_testing_data(self) -> Tuple[DataFrame, Series]:
        """
        Возвращает тестовые данные (признаки и целевую переменную).

        Возвращает:
            Tuple[DataFrame, Series]: (X_test, y_test).
        Выбрасывает:
            RuntimeError: если split_data не был вызван.
        """
        if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
            raise RuntimeError("Данные не разделены. Вызовите split_data().")
        return self.X_test, self.y_test

    def remove_feature(self, feature_name: str) -> None:
        """
        Удаляет указанный признак из текущего набора данных.

        Параметры:
            feature_name (str): Название удаляемого признака.
        Исключения:
            ValueError: если feature_name не строка.
            KeyError: если признак отсутствует в self.features.
            RuntimeError: если self.features не инициализирован.
        """
        if self.features is None:
            raise RuntimeError(
                "Набор признаков пуст. Вызовите _extract_features_target()."
            )
        if not isinstance(feature_name, str):
            raise ValueError(
                f"Имя признака должно быть строкой, получено {type(feature_name).__name__}"
            )
        if feature_name not in self.features.columns:
            raise KeyError(
                f"Признак '{feature_name}' отсутствует в текущем наборе признаков."
            )

        self.features.drop(columns=[feature_name], inplace=True)
        if self.df is not None and feature_name in self.df.columns:
            self.df.drop(columns=[feature_name], inplace=True)
        if (
            self.scaled_features is not None
            and feature_name in self.scaled_features.columns
        ):
            self.scaled_features.drop(columns=[feature_name], inplace=True)

        print(f"Признак '{feature_name}' успешно удалён из набора данных.")


    def visualize_target_distribution(
        self, 
        figsize: Tuple[int, int] = (10, 6), 
        bins: int = 30, 
        kde: bool = True,
        log_scale: bool = False
    ) -> None:
        """
        Строит гистограмму распределения целевой переменной (цены дома).

        Параметры:
            figsize (Tuple[int, int]): Размер графика (ширина, высота).
            bins (int): Количество бинов для гистограммы.
            kde (bool): Отображать ли кривую оценки плотности.
            log_scale (bool): Использовать логарифмическую шкалу для оси Y.
        
        Выбрасывает:
            RuntimeError: Если целевая переменная не загружена.
        """
        if self.target is None:
            raise RuntimeError("Целевая переменная не загружена. Сначала вызовите _extract_features_target().")
        
        plt.figure(figsize=figsize)
        
        sns.histplot(
            self.target, 
            bins=bins, 
            kde=kde, 
            color='skyblue', 
            edgecolor='black', 
            linewidth=1.2,
            alpha=0.7
        )
        
        mean_val = self.target.mean()
        median_val = self.target.median()
        std_val = self.target.std()
        
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Медиана: {median_val:.2f}')
        plt.axvline(mean_val - std_val, color='purple', linestyle=':', linewidth=1.5, label=f'±1 STD')
        plt.axvline(mean_val + std_val, color='purple', linestyle=':', linewidth=1.5)
        
        if log_scale:
            plt.yscale('log')
            plt.ylabel('Частота (log scale)')
        else:
            plt.ylabel('Частота')
        
        plt.title(f'Распределение целевой переменной: {self.target_column}')
        plt.xlabel(self.target_column)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        stats_text = (
            f"Минимум: {self.target.min():.2f}\n"
            f"Максимум: {self.target.max():.2f}\n"
            f"Среднее: {mean_val:.2f}\n"
            f"Медиана: {median_val:.2f}\n"
            f"Станд. отклонение: {std_val:.2f}\n"
        )
        plt.annotate(
            stats_text, 
            xy=(0.98, 0.7), 
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            horizontalalignment='right',
            fontsize=10
        )
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # manager = DatasetManager(csv_path="Student_Performance.csv", target_column="Performance Index")

    # stats = manager.compute_basic_statistics()
    # print("Описание признаков:")
    # print(stats["describe"])
    # print("\nКорреляционная матрица признаков:")
    # print(stats["correlation_matrix"])
    # print("\nСтатистика по целевой переменной:")
    # print(stats["target_distribution"])

    # manager.visualize_distributions()
    # manager.visualize_correlation_heatmap()
    # manager.visualize_target_distribution()
    
    # manager.preprocess(drop_duplicates=True, drop_outliers=True, z_thresh=2.0)
    # manager.split_data(test_size=0.2, random_state=42)

    # X_train, y_train = manager.get_training_data()
    # X_test, y_test = manager.get_testing_data()

    test_manager = DatasetManager('housing_dataset.csv', 'Price')
    test_manager.preprocess(z_thresh=5)
