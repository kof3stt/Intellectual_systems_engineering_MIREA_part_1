from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from math import inf
from typing import Optional, List, Set

import pandas as pd

from dataset_generator import (
    DatasetGenerator,
    GenerationMode,
    EmptyGenerationSetException,
    GenerationException,
)


class Metric(str, Enum):
    """Метрики качества ассоциативных правил."""

    SUPPORT = "support"
    CONFIDENCE = "confidence"
    CONVICTION = "conviction"
    LIFT = "lift"
    LEVERAGE = "leverage"


@dataclass(frozen=True)
class RuleFilter:
    """
    Описывает одно условие фильтрации:
      metric — по какой метрике фильтруем,
      min_threshold — минимально допустимое значение (или None),
      max_threshold — максимально допустимое значение (или None).
    """

    metric: Metric
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None


class Eclat:
    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Инициализирует объект для поиска частых наборов методом Eclat.

        Параметры:
            dataset (pd.DataFrame): one-hot–кодированный DataFrame транзакций,
                где столбцы — товары, строки — транзакции, значения 0/1 или False/True.
        """
        self.dataset = dataset > 0

    def eclat(
        self, min_support: float = 0.25, max_len: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Находит частые наборы элементов методом Eclat (вертикальное хранение).

        Параметры:
            min_support (float): нижняя граница поддержки в [0.0, 1.0].
            max_len (Optional[int]): максимальная длина наборов (None — без ограничения).

        Возвращает:
            pd.DataFrame с колонками:
                - itemsets (frozenset): частый набор элементов
                - support (float): доля транзакций, содержащих набор
        """
        if not isinstance(min_support, (float, int)):
            raise TypeError("min_support должен быть числом")
        if not 0 <= min_support <= 1:
            raise ValueError("min_support должен быть в диапазоне [0;1]")
        if max_len is not None:
            if not isinstance(max_len, int):
                raise TypeError("max_len должен быть целым или None")
            if max_len < 1:
                raise ValueError("max_len должен быть ≥1 или None")

        n_transactions = len(self.dataset)
        tid_lists = {}
        for col in self.dataset.columns:
            tids = set(self.dataset.index[self.dataset[col]].tolist())
            sup = len(tids) / n_transactions
            if sup >= min_support:
                tid_lists[col] = tids

        support_data = {}
        for item, tids in tid_lists.items():
            support_data[frozenset([item])] = len(tids) / n_transactions

        def dfs(prefix: List[str], prefix_tids: Set[int], items: List[str]) -> None:
            """
            prefix — текущий набор (список товаров),
            prefix_tids — пересечённый TID-list,
            items — оставшиеся кандидаты для расширения
            """
            for i, item in enumerate(items):
                new_prefix = prefix + [item]
                new_tids = prefix_tids & tid_lists[item] if prefix else tid_lists[item]
                sup = len(new_tids) / n_transactions
                if sup < min_support:
                    continue
                fs = frozenset(new_prefix)
                support_data[fs] = sup
                if max_len is None or len(new_prefix) < max_len:
                    dfs(new_prefix, new_tids, items[i + 1 :])

        items = sorted(tid_lists.keys())
        dfs([], set(), items)

        result = pd.DataFrame(
            [
                {"itemsets": itemset, "support": support}
                for itemset, support in support_data.items()
            ]
        )
        return result

    def association_rules(
        self, frequent_itemsets: pd.DataFrame, filters: List[RuleFilter]
    ) -> pd.DataFrame:
        """
        Строит ассоциативные правила из частых наборов и фильтрует их.

        Параметры:
            frequent_itemsets (pd.DataFrame): результат apriori,
                колонки itemsets (frozenset) и support (float).
            filters (List[RuleFilter]): список условий фильтрации по метрикам.

        Возвращает:
            pd.DataFrame с колонками:
                - antecedents (frozenset)
                - consequents (frozenset)
                - support (float)
                - confidence (float)
                - lift (float)
                - leverage (float)
                - conviction (float)
            и только теми строками, которые проходят все фильтры.
        """
        df_bool = self.dataset

        support_map = {
            frozenset(row["itemsets"]): float(row["support"])
            for _, row in frequent_itemsets.iterrows()
        }

        records = []

        for itemset, support_AB in support_map.items():
            if len(itemset) < 2:
                continue
            for r in range(1, len(itemset)):
                for antecedent in combinations(itemset, r):
                    X = frozenset(antecedent)
                    Y = itemset - X
                    support_A = support_map.get(X)
                    support_B = support_map.get(Y)
                    if support_B is None:
                        mask_B = df_bool[list(Y)].all(axis=1)
                        support_B = float(mask_B.sum()) / len(df_bool)
                    confidence = support_AB / support_A
                    lift = confidence / support_B
                    leverage = support_AB - support_A * support_B
                    conviction = (
                        (1 - support_B) / (1 - confidence) if confidence != 1 else inf
                    )

                    records.append(
                        {
                            "antecedents": X,
                            "consequents": Y,
                            "support": support_AB,
                            "confidence": confidence,
                            "lift": lift,
                            "leverage": leverage,
                            "conviction": conviction,
                        }
                    )

        rules_df = pd.DataFrame(records)

        for f in filters:
            col = f.metric.value
            if f.min_threshold is not None:
                rules_df = rules_df[rules_df[col] >= f.min_threshold]
            if f.max_threshold is not None:
                rules_df = rules_df[rules_df[col] <= f.max_threshold]

        return rules_df


dataset_generator = DatasetGenerator(
    "unifood.json",
    min_order_items=2,
    max_order_items=5,
    max_order_price=1000,
    allow_duplicates=False,
    mode=GenerationMode.UNTIL_SOLD,
)

dataset = dataset_generator.generate_dataset()

eclat = Eclat(dataset)
frequent_itemsets = eclat.eclat(min_support=0.01)

frequent_itemsets.sort_values(["support"], ascending=[False])
print(frequent_itemsets.sort_values(["support"], ascending=[False]))

filters = [RuleFilter(Metric.LIFT, min_threshold=1)]

rules = eclat.association_rules(frequent_itemsets, filters)
print(rules)
