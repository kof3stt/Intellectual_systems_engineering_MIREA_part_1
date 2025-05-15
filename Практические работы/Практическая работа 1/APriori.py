from dataset_generator import (
    DatasetGenerator,
    GenerationMode,
    EmptyGenerationSetException,
    GenerationException,
)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


def hot_encode(x):
    if x == 0:
        return False
    return True


dataset_generator = DatasetGenerator(
    "unifood.json",
    min_order_items=2,
    max_order_items=5,
    max_order_price=1000,
    allow_duplicates=False,
    mode=GenerationMode.UNTIL_SOLD,
)

dataset = dataset_generator.generate_dataset()

dataset = dataset.map(hot_encode)
print(dataset.head())

frq_items = apriori(dataset, min_support=0.01, use_colnames=True)
frq_items.sort_values(["support"], ascending=[False])

print(frq_items.sort_values(["support"], ascending=[False]))

rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules.sort_values(["lift"], ascending=[False])

print(rules.sort_values(["lift"], ascending=[False]))

rules_random = rules.sample(10, random_state=42)
rules_lift = rules_random[["lift"]].to_numpy()
rules_lift = (rules_lift / rules_lift.max()).transpose()[0]
rules_conf = rules_random[["confidence"]].to_numpy()
rules_conf = (rules_conf / rules_conf.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(8, 4), dpi=200)

plt.bar(np.arange(len(rules_random)) - 0.2, rules_lift, width, color="black")
plt.bar(
    np.arange(len(rules_random)) + 0.2,
    rules_conf,
    width,
    hatch="//",
    edgecolor="black",
    facecolor="white",
)
plt.xlabel("Instance index")
plt.ylabel("Normalized metric value")
plt.legend(["lift", "confidence"])
plt.xticks(range(0, 10))
plt.show()
