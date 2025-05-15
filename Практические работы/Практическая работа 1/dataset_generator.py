import json
import random
import pandas as pd
from copy import deepcopy
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict, Optional, Any


class GenerationMode(Enum):
    """Режимы работы генератора."""

    UNTIL_SOLD = auto()
    FIXED_ROWS = auto()


class EmptyGenerationSetException(Exception):
    """Нет ни одной комбинации, удовлетворяющей заданным параметрам генерации."""

    pass


class GenerationException(Exception):
    """Ошибка во время процесса генерации."""

    pass


class DatasetGenerator:
    """Генератор случайных «заказов» на основе JSON-файла с товарами."""

    def __init__(
        self,
        json_name: str,
        min_order_price: Optional[float | int] = None,
        max_order_price: Optional[float | int] = None,
        min_order_items: Optional[int] = None,
        max_order_items: Optional[int] = None,
        allow_duplicates: bool = True,
        mode: GenerationMode = GenerationMode.UNTIL_SOLD,
        num_rows: Optional[int] = None,
    ) -> None:
        """
        Инициализирует генератор заказов на основе JSON-файла.

        Параметры:
            json_name (str): Путь к JSON-файлу с описанием товаров.
            min_order_price (Optional[float | int]): Минимальная сумма одного заказа.
            max_order_price (Optional[float | int]): Максимальная сумма одного заказа.
            min_order_items (Optional[int]): Минимальное число позиций в одном заказе.
            max_order_items (Optional[int]): Максимальное число позиций в одном заказе.
            allow_duplicates (bool): Разрешить ли несколько единиц одного товара в заказе.
            mode (GenerationMode): Режим генерации (UNTIL_SOLD или FIXED_ROWS).
            num_rows (Optional[int]): Число строк (заказов) при режиме FIXED_ROWS.
        """

        self.json_name = json_name
        self.min_order_price = min_order_price
        self.max_order_price = max_order_price
        self.min_order_items = min_order_items
        self.max_order_items = max_order_items

        self._validate_min_max()

        self.allow_duplicates = allow_duplicates
        self.mode = mode
        self.num_rows = num_rows

        self._validate_mode_and_num_rows()

        self._load_data()

    @property
    def json_name(self) -> str:
        """Путь к JSON-файлу с описанием товаров."""
        return self._json_name

    @json_name.setter
    def json_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("Имя файла должно быть строкой")
        if not Path(value).exists():
            raise FileNotFoundError(f"Файл '{value}' не найден")
        if not value.lower().endswith(".json"):
            raise ValueError("Файл должен иметь расширение .json")
        self._json_name = value

    @property
    def min_order_price(self) -> Optional[float]:
        """Нижняя граница суммы заказа."""
        return self._min_order_price

    @min_order_price.setter
    def min_order_price(self, value: Optional[float | int]) -> None:
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError("min_order_price должен быть числом")
            if value < 0:
                raise ValueError("min_order_price не может быть отрицательным")
        self._min_order_price = float(value) if value is not None else None

    @property
    def max_order_price(self) -> Optional[float]:
        """Верхняя граница суммы заказа."""
        return self._max_order_price

    @max_order_price.setter
    def max_order_price(self, value: Optional[float | int]) -> None:
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError("max_order_price должен быть числом")
            if value < 0:
                raise ValueError("max_order_price не может быть отрицательным")
        self._max_order_price = float(value) if value is not None else None

    @property
    def min_order_items(self) -> Optional[int]:
        """Минимальное число товарных позиций в заказе."""
        return self._min_order_items

    @min_order_items.setter
    def min_order_items(self, value: Optional[int]) -> None:
        if value is not None:
            if not isinstance(value, int):
                raise TypeError("min_order_items должен быть целым числом")
            if value < 0:
                raise ValueError("min_order_items не может быть отрицательным")
        self._min_order_items = int(value) if value is not None else None

    @property
    def max_order_items(self) -> Optional[int]:
        """Максимальное число товарных позиций в заказе."""
        return self._max_order_items

    @max_order_items.setter
    def max_order_items(self, value: Optional[int]) -> None:
        if value is not None:
            if not isinstance(value, int):
                raise TypeError("max_order_items должен быть целым числом")
            if value < 0:
                raise ValueError("max_order_items не может быть отрицательным")
        self._max_order_items = int(value) if value is not None else None

    @property
    def allow_duplicates(self) -> bool:
        """Флаг: можно ли несколько единиц одного товара в заказе."""
        return self._allow_duplicates

    @allow_duplicates.setter
    def allow_duplicates(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("allow_duplicates должен быть булевым")
        self._allow_duplicates = value

    @property
    def mode(self) -> GenerationMode:
        """Режим генерации."""
        return self._mode

    @mode.setter
    def mode(self, value: GenerationMode) -> None:
        if not isinstance(value, GenerationMode):
            raise ValueError("mode должен быть элементом GenerationMode")
        self._mode = value

    @property
    def num_rows(self) -> Optional[int]:
        """Число строк для режима FIXED_ROWS."""
        return self._num_rows

    @num_rows.setter
    def num_rows(self, value: Optional[int]) -> None:
        if value is not None:
            if not isinstance(value, int):
                raise TypeError("num_rows должен быть целым числом")
            if value < 0:
                raise ValueError("num_rows не может быть отрицательным")
        self._num_rows = value

    def _validate_min_max(self) -> None:
        """
        Проверяет, что min_order_price < max_order_price и
        min_order_items < max_order_items (если оба заданы).

        Исключения:
            ValueError
        """
        if self.min_order_price is not None and self.max_order_price is not None:
            if self.min_order_price >= self.max_order_price:
                raise ValueError(
                    f"min_order_price должен быть меньше max_order_price: {self.min_order_price} < {self.max_order_price} - False"
                )
        if self.min_order_items is not None and self.max_order_items is not None:
            if self.min_order_items >= self.max_order_items:
                raise ValueError(
                    f"min_order_items должен быть меньше max_order_items: {self.min_order_items} < {self.max_order_items} - False"
                )

    def _validate_mode_and_num_rows(self) -> None:
        """
        Проверяет согласованность mode и num_rows:
        - UNTIL_SOLD требует num_rows=None
        - FIXED_ROWS требует num_rows заданным

        Исключения:
            ValueError
        """
        if self.mode == GenerationMode.UNTIL_SOLD and self.num_rows is not None:
            raise ValueError(
                "Для режима генерации UNTIL_SOLD параметр num_rows должен опущен"
            )
        if self.mode == GenerationMode.FIXED_ROWS and self.num_rows is None:
            raise ValueError(
                "Для режима генерации FIXED_ROWS параметр num_rows должен быть задан"
            )

    def _load_data(self) -> None:
        """
        Загружает данные из JSON:
        - self._original_data: список всех товаров (List[Dict])
        - self._headers: список их названий (List[str])

        Исключения:
            ValueError, RuntimeError
        """
        try:
            with open(self._json_name, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON-файла: {e}")
        except Exception as e:
            raise RuntimeError(
                f"Непредвиденная ошибка при чтении {self._json_name}: {e}"
            )

        if not isinstance(raw, dict):
            raise ValueError(
                "Ожидался словарь со строками в ключах и списками словарей в значениях"
            )

        self._original_data = []
        self._headers = []
        for items in raw.values():
            if 'teremok' in self.json_name:
                for item in items:
                    item['Цена'] = float(item['Цена'].rstrip('₽'))
            self._original_data.extend(items)
            for item in items:
                if "Позиция" in item:
                    self._headers.append(item["Позиция"])
                elif "Название продукта" in item:
                    self._headers.append(item["Название продукта"])

    def generate_dataset(self) -> pd.DataFrame:
        """
        Формирует DataFrame заказов в зависимости от режима.

        Возвращает:
            pd.DataFrame: строки — заказы, столбцы — позиции.
        """
        if self.mode == GenerationMode.FIXED_ROWS:
            rows = self._generate_fixed_rows()
        elif self.mode == GenerationMode.UNTIL_SOLD:
            rows = self._generate_until_sold()
        return pd.DataFrame(rows, columns=self._headers)

    def _generate_fixed_rows(self) -> List[List[int]]:
        """
        Составляет ровно num_rows заказов, каждый удовлетворяет параметрам.

        Возвращает:
            List[List[int]]: список векторов количеств по каждому товару.

        Исключения:
            GenerationException
        """
        data_pool = self._validate_params()
        rows = []
        for _ in range(self.num_rows):
            for attempt in range(10000):
                min_items = self.min_order_items or 1
                max_items = self.max_order_items or len(data_pool)
                num_items = random.randint(min_items, max_items)

                if self.allow_duplicates:
                    chosen = random.choices(data_pool, k=num_items)
                else:
                    chosen = random.sample(data_pool, k=num_items)

                total_price = sum(item["Цена"] for item in chosen)

                if (
                    self.min_order_price is not None
                    and total_price < self.min_order_price
                ):
                    continue
                if (
                    self.max_order_price is not None
                    and total_price > self.max_order_price
                ):
                    continue

                row = [0] * len(self._original_data)
                for item in chosen:
                    idx = self._original_data.index(item)
                    row[idx] += 1

                rows.append(row)
                break
            else:
                raise GenerationException(
                    f"Не удалось сгенерировать строку {_ + 1}/{self.num_rows} "
                    f"за {attempt + 1} попыток"
                )
        return rows

    def _validate_params(self) -> List[Dict[str, Any]]:
        """
        Отбирает товары по max_order_price и проверяет min_order_items
        (при allow_duplicates=False).

        Возвращает:
            List[Dict]: список отфильтрованных товаров.

        Исключения:
            EmptyGenerationSetException
        """
        data = deepcopy(self._original_data)
        if self.max_order_price is not None:
            data = list(
                filter(
                    lambda item: item["Цена"] <= self.max_order_price,
                    self._original_data,
                )
            )
            if not data:
                raise EmptyGenerationSetException(
                    f"Невозможно сгенерировать датасет, не найдено позиций, удовлетворяющих параметрам генерации: max_order_price={self.max_order_price}"
                )
        if self.min_order_items is not None and not self.allow_duplicates:
            total_unique = len(data)
            if self.min_order_items > len(data):
                raise EmptyGenerationSetException(
                    f"Невозможно сгенерировать датасет, не найдено позиций, удовлетворяющих параметрам генерации: min_order_items={self.min_order_items}, allow_duplicates={self.allow_duplicates}. Невозможно выбрать {self.min_order_items} уникальных товаров из {total_unique}"
                )
        return data

    def _generate_until_sold(self) -> List[List[int]]:
        """
        Генерирует заказы, пока есть остатки товаров (поле 'Продано').

        Возвращает:
            List[List[int]]: список векторов количеств по каждому товару.
        """
        data_pool = self._validate_params()
        remaining = [item["Продано"] for item in data_pool]
        rows = []
        while any(rem > 0 for rem in remaining):
            for _ in range(10000):
                min_items = self.min_order_items or 1
                max_items = self.max_order_items or len(data_pool)
                num_items = random.randint(min_items, max_items)

                available_indices = [
                    idx for idx, rem in enumerate(remaining) if rem > 0
                ]
                if not available_indices:
                    return rows

                if self.allow_duplicates:
                    chosen_idxs = random.choices(available_indices, k=num_items)
                else:
                    if num_items > len(available_indices):
                        continue
                    chosen_idxs = random.sample(available_indices, k=num_items)

                cnt = {}
                for i in chosen_idxs:
                    cnt[i] = cnt.get(i, 0) + 1
                    if cnt[i] > remaining[i]:
                        break

                else:
                    total_price = sum(
                        self._original_data[i]["Цена"] * cnt[i] for i in cnt
                    )
                    if (
                        self.min_order_price is not None
                        and total_price < self.min_order_price
                    ):
                        continue
                    if (
                        self.max_order_price is not None
                        and total_price > self.max_order_price
                    ):
                        continue

                    for i, q in cnt.items():
                        remaining[i] -= q

                    row = [0] * len(self._original_data)
                    for i, q in cnt.items():
                        row[i] = q
                    rows.append(row)
                    break
            else:
                remaining_items = [
                    (self._original_data[i]["Позиция"], rem)
                    for i, rem in enumerate(remaining)
                    if rem > 0
                ]
                print(
                    "\033[91m"
                    + f"Не удалось сгенерировать очередную строку в режиме UNTIL_SOLD. "
                    f"Сгенерировано: {len(rows)} строк. Остатки по позициям:"
                    + "\033[0m"
                )
                for name, qty in remaining_items:
                    print(f"  • {name}: {qty}")
                return rows

        return rows
