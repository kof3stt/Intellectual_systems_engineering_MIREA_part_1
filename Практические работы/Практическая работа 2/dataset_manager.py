import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import Optional, Dict, Tuple, List
from pandas import DataFrame, Series


class DatasetManager:
    def __init__(
        self,
        source: str = "sklearn",
        csv_path: Optional[str] = None,
    ) -> None:
        """
        Инициализирует менеджер датасета для загрузки, анализа, предобработки и визуализации.

        Параметры:
            source (str): Источник данных.
                - "sklearn": загружаем встроенный датасет Wine из sklearn.
                - "csv": читаем CSV-файл по пути csv_path.
            csv_path (Optional[str]): Путь к CSV-файлу при source="csv".
                Если source="sklearn", игнорируется.
        """
        self.source: str = source
        self.csv_path: Optional[str] = csv_path
        self.df: Optional[DataFrame] = None
        self.features: Optional[DataFrame] = None
        self.target: Optional[Series] = None
        self.scaled_features: Optional[DataFrame] = None
        self.stats: Dict[str, DataFrame] = {}

        self._load_data()
        self._extract_features_target()

    def _load_data(self) -> None:
        """
        Загружает исходный датасет в self.df.

        При source="sklearn" загружается Wine-датасет из sklearn.
        При source="csv" загружается CSV-файл по пути csv_path.

        Выбрасывает:
            ValueError: если source="csv" и csv_path не указан или source не равен "sklearn"/"csv".
        """
        if self.source == "sklearn":
            raw = load_wine(as_frame=True)
            df0 = raw.frame.copy()
            self.df = df0
        elif self.source == "csv":
            if self.csv_path is None:
                raise ValueError("При source='csv' необходимо указать путь csv_path")
            self.df = pd.read_csv(self.csv_path)
        else:
            raise ValueError("source должен быть 'sklearn' или 'csv'")

        print(
            f"Данные загружены: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов"
        )

    def _extract_features_target(self) -> None:
        """
        Разделяет DataFrame на признаки и метку (если столбец 'target' присутствует).

        После выполнения:
            - self.features будет содержать DataFrame только с признаками.
            - self.target будет содержать Series с метками классов (или None, если 'target' отсутствует).
        """
        if self.df is None:
            raise RuntimeError("Данные не загружены. Сначала вызовите _load_data().")

        if "target" in self.df.columns:
            self.target = self.df["target"].copy()
            self.features = self.df.drop(columns=["target"]).copy()
        else:
            self.target = None
            self.features = self.df.copy()

    def compute_basic_statistics(self) -> Dict[str, DataFrame]:
        """
        Вычисляет базовые статистики по признакам и сохраняет их в self.stats.

        Сохраняются:
            - "describe": описательные статистики (mean, std, min, max, квартили) для каждого признака.
            - "correlation_matrix": матрица корреляций между признаками.
            - "class_distribution": распределение по классам (если есть self.target).

        Возвращает:
            Dict[str, DataFrame]: Словарь с DataFrame-статистиками.
        """
        if self.features is None:
            raise RuntimeError(
                "Признаки не выделены. Сначала вызовите _extract_features_target()."
            )

        desc = self.features.describe().T
        self.stats["describe"] = desc

        corr = self.features.corr()
        self.stats["correlation_matrix"] = corr

        if self.target is not None:
            class_counts: Series = self.target.value_counts().sort_index()
            self.stats["class_distribution"] = class_counts.to_frame(name="count")

        return self.stats

    def preprocess(
        self,
        drop_duplicates: bool = True,
        drop_outliers: bool = True,
        z_thresh: float = 3.0,
    ) -> None:
        """
        Полная предобработка данных:
            1. Удаление дубликатов.
            2. Удаление выбросов по Z-оценке (если drop_outliers=True).
            3. Масштабирование признаков StandardScaler.

        Параметры:
            drop_duplicates (bool): Удалять ли полные дубликаты строк (True/False).
            drop_outliers (bool): Удалять ли выбросы по Z-оценке (True/False).
            z_thresh (float): Порог Z-оценки; объекты, у которых хотя бы один признак
                              имеет |z_score| > z_thresh, считаются выбросами.

        После выполнения:
            - self.df обновляется без дубликатов и выбросов.
            - self.features обновляются (признаки из очищенного DataFrame).
            - self.target обновляется (метки из очищенного DataFrame).
            - self.scaled_features заполняется DataFrame-ом масштабированных признаков.
        """
        if self.df is None:
            raise RuntimeError("Данные не загружены. Сначала вызовите _load_data().")

        df_proc: DataFrame = self.df.copy()

        if drop_duplicates:
            before = df_proc.shape[0]
            df_proc = df_proc.drop_duplicates().reset_index(drop=True)
            after = df_proc.shape[0]
            print(f"Удалено дубликатов: {before - after}")

        if drop_outliers:
            df_no_target = df_proc.drop(columns=["target"], errors="ignore")
            means = df_no_target.mean()
            stds = df_no_target.std(ddof=0)
            z_scores = (df_no_target - means) / stds
            mask = (z_scores.abs() <= z_thresh).all(axis=1)
            before_out = df_proc.shape[0]
            df_proc = df_proc[mask].reset_index(drop=True)
            after_out = df_proc.shape[0]
            print(f"Удалено выбросов: {before_out - after_out}")

        scaler = StandardScaler()
        feat: DataFrame = df_proc.drop(columns=["target"], errors="ignore")
        scaled_array = scaler.fit_transform(feat)
        scaled_df = pd.DataFrame(scaled_array, columns=feat.columns, index=feat.index)

        self.df = df_proc
        if "target" in df_proc.columns:
            self.target = df_proc["target"].copy()
            self.features = df_proc.drop(columns=["target"]).copy()
        else:
            self.target = None
            self.features = df_proc.copy()

        self.scaled_features = scaled_df
        print(
            "Предобработка завершена: дубликаты и выбросы (если указано) удалены, признаки масштабированы."
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

    def visualize_scatter_matrix(
        self,
        with_target: bool = True,
        figsize: Tuple[int, int] = (10, 10),
    ) -> None:
        """
        Строит матрицу рассеяния (pairplot) для первых 5–7 признаков.

        Параметры:
            with_target (bool): Если True и self.target определён, раскрашивает точки по классам.
            figsize (Tuple[int, int]): Размер фигуры (ширина, высота) в дюймах.
        """
        if self.features is None:
            raise RuntimeError(
                "Признаки не выделены. Сначала вызовите _extract_features_target()."
            )

        num_to_plot = min(7, len(self.features.columns))
        df_plot = self.features.iloc[:, :num_to_plot].copy()

        if with_target and self.target is not None:
            df_plot["target"] = self.target.values
            colors = {0: "red", 1: "green", 2: "blue"}
            scatter_matrix(
                df_plot,
                figsize=figsize,
                diagonal="hist",
                color=df_plot["target"].map(colors),
                alpha=0.5,
            )
        else:
            scatter_matrix(df_plot, figsize=figsize, diagonal="hist", alpha=0.5)

        plt.suptitle("Матрица рассеяния признаков", y=1.02)
        plt.show()

    def visualize_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Строит «приятную» тепловую карту корреляций между признаками с помощью seaborn.

        Параметры:
            figsize (Tuple[int, int]): Размер фигуры (ширина, высота) в дюймах.
        """
        if self.features is None:
            raise RuntimeError(
                "Признаки не выделены. Сначала вызовите _extract_features_target()."
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

    def get_preprocessed_data(self) -> Tuple[DataFrame, Optional[Series]]:
        """
        Возвращает масштабированные признаки и метки (если есть) для дальнейшего анализа/кластеризации.

        Возвращает:
            Tuple[DataFrame, Optional[Series]]:
                - DataFrame: self.scaled_features (масштабированные признаки).
                - Series или None: self.target (метки классов, если были изначально).

        Выбрасывает:
            RuntimeError: если self.scaled_features ещё не вычислены (не вызван preprocess()).
        """
        if self.scaled_features is None:
            raise RuntimeError(
                "Данные ещё не предобработаны. Сначала вызовите preprocess()."
            )
        return self.scaled_features, self.target

    def visualize_class_distribution(
        self,
        figsize: Tuple[int, int] = (8, 8),
        title: str = "Распределение по классам",
        colors: Optional[List[str]] = None,
        autopct: str = "%1.1f%%",
        startangle: int = 90,
    ) -> None:
        """
        Строит круговую диаграмму распределения объектов по классам.

        Параметры:
            figsize (Tuple[int, int]): Размер фигуры (ширина, высота) в дюймах.
            title (str): Заголовок диаграммы.
            colors (Optional[List[str]]): Список цветов для секторов.
            autopct (str): Формат отображения процентных значений.
            startangle (int): Угол начала первой секции.
        """
        if "class_distribution" not in self.stats:
            raise RuntimeError(
                "Распределение по классам не вычислено. Вызовите compute_basic_statistics()."
            )

        class_dist = self.stats["class_distribution"]
        labels = class_dist.index.astype(str).tolist()
        sizes = class_dist["count"].tolist()

        if not colors:
            colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

        plt.figure(figsize=figsize)
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=autopct,
            startangle=startangle,
            textprops={"fontsize": 12},
            wedgeprops={"edgecolor": "black", "linewidth": 0.5},
        )
        plt.title(title, fontsize=14, pad=20)
        plt.axis("equal")
        plt.show()

    def remove_feature(self, feature_name: str) -> None:
        """
        Удаляет признак из текущего набора данных по его имени.

        Параметры:
            feature_name (str): Название удаляемого признака.

        Исключения:
            ValueError: если feature_name не является строкой.
            KeyError: если признака с таким именем нет в self.features.
            RuntimeError: если self.features ещё не инициализирован (нет данных).
        """
        if self.features is None:
            raise RuntimeError(
                "Набор признаков пуст. Сначала выполните загрузку данных и метод _extract_features_target()."
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

    def visualize_elbow_method(self, k_range: range = range(1, 11)) -> None:
        """
        Реализует метод локтя для определения оптимального числа кластеров.

        Для каждого k в заданном диапазоне вычисляется WCSS (within-cluster sum of squares).
        Строится график зависимости WCSS от количества кластеров k.

        Параметры:
            k_range (range): Диапазон значений k (число кластеров) для анализа.

        Исключения:
            RuntimeError: если данные ещё не были предобработаны.
        """
        if self.scaled_features is None:
            raise RuntimeError(
                "Сначала вызовите preprocess(), чтобы получить масштабированные признаки."
            )

        wcss = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(self.scaled_features)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), wcss, marker="o")
        plt.title("Метод локтя: выбор числа кластеров")
        plt.xlabel("Число кластеров (k)")
        plt.ylabel("Сумма внутрикластерных расстояний (WCSS)")
        plt.xticks(list(k_range))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_silhouette_analysis(self, n_clusters: int = 3) -> None:
        """
        Строит силуэтную диаграмму для оценки качества кластеризации методом k-средних.

        Параметры:
            n_clusters (int): Число кластеров для KMeans.

        Исключения:
            RuntimeError: если не были предобработаны признаки.
        """
        if self.scaled_features is None:
            raise RuntimeError(
                "Сначала вызовите preprocess(), чтобы получить масштабированные признаки."
            )

        X = self.scaled_features.values

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title(f"Силуэтная диаграмма для {n_clusters} кластеров")
        ax1.set_xlabel("Коэффициент силуэта")
        ax1.set_ylabel("Номер кластера")

        ax1.axvline(
            x=silhouette_avg,
            color="red",
            linestyle="--",
            label=f"Среднее значение: {silhouette_avg:.2f}",
        )
        ax1.set_xlim([-0.1, 1.0])
        ax1.set_yticks([])
        ax1.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    manager = DatasetManager(source="sklearn")
    stats = manager.compute_basic_statistics()

    manager.visualize_class_distribution(
        title="Распределение вин по классам",
        colors=["#ff9999", "#66b3ff", "#99ff99"],
        autopct="%1.1f%%",
    )

    print("Описание признаков:")
    print(stats["describe"])
    if "class_distribution" in stats:
        print("\nРаспределение по классам:")
        print(stats["class_distribution"])

    manager.visualize_distributions()
    manager.visualize_scatter_matrix()
    manager.visualize_correlation_heatmap()

    manager.preprocess()

    manager.remove_feature("total_phenols")
    manager.visualize_elbow_method()
    manager.visualize_silhouette_analysis()

    X_scaled, y = manager.get_preprocessed_data()
