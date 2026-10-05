"""Методы рекомендаций для маркетплейса

Данные - действия покупателей: просмотр, избранное, корзина, заказ

Классические подходы:
    content_based    — товары из любимых категорий покупателя в привычном ценовом диапазоне
    item_based       — коллаборативная фильтрация по товарам: что выбирали с похожей историей
Гибрид: hybrid — item_based; если истории мало — добавляем content_based, если покупатель новый - популярное
Эвристики (без модели):
    popular          — «Популярное»: больше всего заказов за последнюю неделю
    recently_viewed  — «Вы недавно смотрели»
    bought_together  — «С этим товаром покупают» на карточке товара
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ORDER, FAVORITE, VIEW, CART = 0, 1, 2, 3
WEIGHTS = {VIEW: 1, FAVORITE: 2, CART: 3, ORDER: 4}
TEST_DAYS = 28


def load(path="data/sample.parquet"):
    return pd.read_parquet(path)


def split(df):
    """История - всё до последних 4 недель, проверка = последние 4 недели"""
    border = df["datetime"].max() - pd.Timedelta(days=TEST_DAYS)
    return df[df["datetime"] < border], df[df["datetime"] >= border]


class Recommender:
    def __init__(self, df):
        self.user_index = {u: i for i, u in enumerate(df["user_id"].unique())}
        self.items = df["item_id"].unique()
        self.item_index = {item: j for j, item in enumerate(self.items)}

        rows = df["user_id"].map(self.user_index).to_numpy()
        cols = df["item_id"].map(self.item_index).to_numpy()
        weights = df["event"].map(WEIGHTS).to_numpy(dtype=float)
        self.X = csr_matrix((weights, (rows, cols)), shape=(len(self.user_index), len(self.items)))
        self.X.data = np.log1p(self.X.data)
        self.XT = self.X.T.tocsr()

        orders = df[df["event"] == ORDER]
        self.ordered = orders.groupby("user_id")["item_id"].agg(set).to_dict()

        o_rows = orders["user_id"].map(self.user_index).to_numpy()
        o_cols = orders["item_id"].map(self.item_index).to_numpy()
        self.O = csr_matrix((np.ones(len(orders)), (o_rows, o_cols)), shape=self.X.shape)
        self.O.data[:] = 1
        self.OT = self.O.T.tocsr()

        buyers = orders.groupby("item_id")["user_id"].nunique()
        self.popularity = buyers.reindex(self.items, fill_value=0).to_numpy(dtype=float)
        week_ago = df["datetime"].max() - pd.Timedelta(days=7)
        last_week = orders[orders["datetime"] >= week_ago]
        self.popular_items = (last_week.groupby("item_id")["user_id"].nunique()
                              .sort_values(ascending=False).index.tolist())
а
        info = df.groupby("item_id").agg(category=("category_id", "first"), price=("price", "median"))
        self.item_price = info["price"].to_dict()
        top = buyers.rename("buyers").to_frame().join(info["category"]).sort_values("buyers", ascending=False)
        top = top.groupby("category").head(50)
        self.category_top = {cat: group.index.tolist() for cat, group in top.groupby("category")}

        weighted = df.assign(weight=df["event"].map(WEIGHTS))
        by_category = (weighted.groupby(["user_id", "category_id"])["weight"].sum()
                       .reset_index().sort_values("weight", ascending=False))
        self.user_categories = by_category.groupby("user_id")["category_id"].apply(lambda s: s.head(3).tolist()).to_dict()
        self.user_price = df.groupby("user_id")["price"].median().to_dict()

        views = df[df["event"] == VIEW].sort_values("datetime", ascending=False)
        views = views.drop_duplicates(["user_id", "item_id"])
        self.last_viewed = views.groupby("user_id")["item_id"].apply(lambda s: s.head(20).tolist()).to_dict()

    def _best(self, scores, k, exclude=()):
        """k товаров с наибольшей оценкой, кроме исключённых"""
        n = min(len(scores), k + len(exclude) + 1)
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        result = []
        for j in top:
            if scores[j] <= 0 or len(result) == k:
                break
            if self.items[j] not in exclude:
                result.append(self.items[j])
        return result

    # эвристики

    def popular(self, user=None, k=10):
        exclude = self.ordered.get(user, set())
        return [item for item in self.popular_items if item not in exclude][:k]

    def recently_viewed(self, user, k=10):
        exclude = self.ordered.get(user, set())
        return [item for item in self.last_viewed.get(user, []) if item not in exclude][:k]

    def bought_together(self, item, k=10):
        """Товары, которые чаще всего заказывали те же люди, что заказали этот"""
        if item not in self.item_index:
            return []
        j = self.item_index[item]
        scores = (self.OT[j] @ self.O).toarray().ravel()
        scores[j] = 0
        return self._best(scores, k)

    # классические подходы

    def content_based(self, user, k=10):
        """Сначала любимая категория покупателя, внутри неё - товары ближе к привычной цене"""
        exclude = self.ordered.get(user, set())
        price = self.user_price.get(user)
        result = []
        for category in self.user_categories.get(user, []):
            candidates = [item for item in self.category_top.get(category, []) if item not in exclude]
            if price:
                candidates.sort(key=lambda item: abs(np.log1p(self.item_price[item]) - np.log1p(price)))
            for item in candidates:
                if item not in result:
                    result.append(item)
                if len(result) == k:
                    return result
        return result

    def item_based(self, user, k=10):
        """Коллаборативная фильтрация по товарам

        Оценка товара - насколько часто он встречается вместе с товарами покупателя
        в истории других людей
        """
        if user not in self.user_index:
            return []
        u = self.user_index[user]
        similarity = (self.X @ self.X[u].T).toarray().ravel()
        similarity[u] = 0  # сам себе не сосед
        scores = np.asarray(self.XT @ similarity).ravel()
        scores = scores / np.sqrt(self.popularity + 1)  # приглушаем товары, которые популярны у всех
        return self._best(scores, k, self.ordered.get(user, set()))

    # гибрид

    def hybrid(self, user, k=10):
        """Персональные рекомендации с запасными вариантами. Возвращает (товары, метод)"""
        if user not in self.user_index:
            return self.popular(user, k), "popular"  # холодный старт: истории нет
        result = self.item_based(user, k)
        source = "item_based"
        if len(result) < k:  # истории мало - добираем по категориям
            if not result:
                source = "content_based"
            result += [item for item in self.content_based(user, k) if item not in result][:k - len(result)]
        if len(result) < k:
            if not result:
                source = "popular"
            result += [item for item in self.popular(user, k) if item not in result][:k - len(result)]
        return result, source
