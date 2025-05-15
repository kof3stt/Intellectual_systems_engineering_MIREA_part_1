from dataset_generator import (
    DatasetGenerator,
    GenerationMode,
    EmptyGenerationSetException,
    GenerationException,
)
import matplotlib.pyplot as plt
import pandas as pd


class DatasetManager:
    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Инициализирует менеджер статистик по датасету.

        Параметры:
            dataset (pd.DataFrame): DataFrame, где строки — транзакции, столбцы — позиции товаров.
        """
        self.dataset: pd.DataFrame = dataset

    def show_dataset_info(self) -> None:
        """
        Выводит основную информацию о датасете:
        число транзакций, число признаков и подробную сводку pandas.
        """
        print(f"Число транзакций: {len(self.dataset)}")
        print(f"Число признаков: {dataset.columns.size}")
        self.dataset.info()

    def show_top_n_items(self, n: int) -> None:
        """
        Строит столбчатую диаграмму для первых N товаров по числу транзакций,
        в которых они встречаются.

        Параметры:
            n (int): количество топ-товаров для отображения.
        """
        freq: pd.Series = (self.dataset > 0).sum(axis=0)
        top: pd.Series = freq.sort_values(ascending=False).head(n)

        plt.figure(figsize=(8, 4))
        plt.bar(top.index, top.values)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Товар")
        plt.ylabel("Число транзакций")
        plt.title(f"Топ {n} товаров по встречаемости в транзакциях")
        plt.tight_layout()
        plt.show()

        print(top)

    def plot_transaction_length_distribution(self) -> None:
        """
        Строит круговую (pie) диаграмму распределения размеров корзин (сумма по строке).
        """
        transaction_lengths: pd.Series = self.dataset.sum(axis=1).astype(int)
        size_counts: pd.Series = transaction_lengths.value_counts().sort_index()

        labels = [
            f"{size} товар{'а' if size < 5 else 'ов'}" for size in size_counts.index
        ]

        plt.figure(figsize=(8, 4))
        plt.pie(
            size_counts,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.Paired.colors,
            wedgeprops={"edgecolor": "black", "linewidth": 0.5},
        )

        plt.title("Распределение размеров корзин")
        plt.tight_layout()
        plt.show()

    def show_basket_stats(self) -> None:
        """
        Выводит основные статистики по размерам корзин:
        среднее, медиану, моду, минимум и максимум.
        """
        transaction_lengths: pd.Series = self.dataset.sum(axis=1)
        print(f"Средний размер корзины: {transaction_lengths.mean():.2f}")
        print(f"Медианный размер: {transaction_lengths.median()}")
        print(f"Мода: {transaction_lengths.mode().iat[0]}")
        print(
            f"Минимум/Максимум: {transaction_lengths.min()}/{transaction_lengths.max()}"
        )

    def check_duplicates(self) -> None:
        """
        Подсчитывает и выводит число полностью дублирующихся транзакций в датасете.
        """
        duplicates: int = self.dataset.duplicated().sum()
        print(f"Число дубликатов транзакций: {duplicates}")


dataset_generator = DatasetGenerator(
    "unifood.json",
    min_order_items=2,
    max_order_items=5,
    max_order_price=1000,
    allow_duplicates=False,
    mode=GenerationMode.UNTIL_SOLD,
)

dataset = dataset_generator.generate_dataset()
dataset_manager = DatasetManager(dataset)

dataset_manager.show_dataset_info()
dataset_manager.show_top_n_items(20)
dataset_manager.plot_transaction_length_distribution()
dataset_manager.show_basket_stats()
dataset_manager.check_duplicates()
