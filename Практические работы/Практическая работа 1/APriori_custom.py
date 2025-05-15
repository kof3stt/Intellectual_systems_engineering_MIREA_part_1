from dataset_generator import (
    DatasetGenerator,
    GenerationMode,
    EmptyGenerationSetException,
    GenerationException,
)
from enum import Enum
from math import inf
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Set, FrozenSet, List
import pandas as pd


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


class APriori:
    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Инициализирует датасет для поиска частых наборов и построения правил.

        Параметры:
            dataset (pd.DataFrame): one-hot–кодированный DataFrame транзакций,
                где столбцы — товары, строки — транзакции, значения 0/1 или False/True.
        """
        self.dataset = dataset

    def apriori(
        self, min_support: float = 0.25, max_len: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Находит частые наборы элементов методом Apriori.

        Параметры:
            min_support (float): минимальная поддержка в диапазоне [0.0, 1.0].
            max_len (Optional[int]): максимальный размер наборов (None — без ограничения).

        Возвращает:
            pd.DataFrame с колонками:
                - itemsets (frozenset): частый набор элементов,
                - support (float): доля транзакций, содержащих этот набор.
        """
        if not isinstance(min_support, (float, int)):
            raise TypeError("Минимальное значение support должно быть числом")
        if min_support < 0 or min_support > 1:
            raise ValueError(
                "Минимальное значение support должно находиться в диапазоне [0;1]"
            )
        if max_len is not None:
            if not isinstance(max_len, int):
                raise TypeError("Значение max_len должно быть целым числом")
            if max_len < 0:
                raise ValueError("Значение max_len должно быть положительным числом")

        df = self.dataset
        n_transactions = len(df)

        support_data = {}
        L = []
        for col in df.columns:
            sup = df[col].sum() / n_transactions
            if sup >= min_support:
                itemset = frozenset([col])
                support_data[itemset] = sup
                L.append({col})

        k = 2
        prev_L = L

        while prev_L and (max_len is None or k <= max_len):
            Ck = set()
            for i in range(len(prev_L)):
                for j in range(i + 1, len(prev_L)):
                    union_set = prev_L[i] | prev_L[j]
                    if len(union_set) == k:
                        subsets = combinations(union_set, k - 1)
                        if all(
                            frozenset(sub) in map(frozenset, prev_L) for sub in subsets
                        ):
                            Ck.add(frozenset(union_set))

            next_L = []
            for candidate in Ck:
                mask = df[list(candidate)].all(axis=1)
                sup = mask.sum() / n_transactions
                if sup >= min_support:
                    support_data[candidate] = sup
                    next_L.append(set(candidate))

            prev_L = next_L
            k += 1

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

apriori = APriori(dataset)
frequent_itemsets = apriori.apriori(min_support=0.01)
frequent_itemsets.sort_values(["support"], ascending=[False])

print(frequent_itemsets.sort_values(["support"], ascending=[False]))

filters = [RuleFilter(Metric.LIFT, min_threshold=1)]

rules = apriori.association_rules(frequent_itemsets, filters)

print(rules)
